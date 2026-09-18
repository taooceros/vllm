"""In-process asynchronous serving driver over vLLM V1's synchronous engine.

``AsyncLLM`` unconditionally starts ``EngineCore`` in a separate process
(``vllm/v1/engine/async_llm.py:145`` calls ``make_async_mp_client``), and
``EngineCoreClient.make_client`` refuses ``asyncio_mode`` without
multiprocessing.  That would put the scheduler, the KV-cache manager and every
migration seam behind a ZMQ boundary in another process.

This driver keeps ``EngineCore`` in the actor process by using ``LLMEngine``
with ``multiprocess_mode=False`` and running its step loop on a dedicated
thread, delivering outputs back into the actor's asyncio loop.  Engine mutation
(``add_request``/``abort``/``step``) is serialised under one lock, which is also
the fence used later to quiesce a single request for migration.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from typing import Any, Callable, Dict, Iterator, List, Optional


class RequestStream:
    """Per-request delivery channel from the step thread to the event loop."""

    __slots__ = ("request_id", "future", "outputs", "first_token_s", "created_at",
                 "prefix_tokens", "prefix_chars", "full_output_text",
                 "qualification_top32", "first_token_id", "first_token_logprobs")

    def __init__(self, request_id: str, loop: asyncio.AbstractEventLoop) -> None:
        self.request_id = request_id
        self.future: asyncio.Future = loop.create_future()
        self.outputs: Optional[Any] = None
        self.first_token_s: Optional[float] = None
        self.created_at = time.time()
        self.prefix_tokens = 0
        self.prefix_chars = 0
        self.full_output_text = ""
        self.qualification_top32 = False
        self.first_token_id: Optional[int] = None
        self.first_token_logprobs: Optional[Dict[int, float]] = None


class InprocAsyncEngine:
    """Drives an in-process ``LLMEngine`` from an asyncio actor."""

    def __init__(self, engine: Any, loop: asyncio.AbstractEventLoop) -> None:
        self.engine = engine
        self.loop = loop
        self.lock = threading.RLock()
        # Turnstile in front of ``lock``.  Python locks barge: the step thread
        # re-acquires immediately after releasing and can starve the actor's
        # event loop for tens of seconds.  Every acquirer must pass the
        # turnstile first, and the step thread releases it before its long
        # critical section, so a waiter always wins the next round.
        self._turnstile = threading.Lock()
        self._capacity_snapshot: Dict[str, Any] = {}
        self._streams: Dict[str, RequestStream] = {}
        self._wake = threading.Event()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._step_count = 0
        self._last_step_s = 0.0
        self._error: Optional[BaseException] = None
        self._iteration_observers: List[Callable[[List[Any]], None]] = []

    # ----------------------------------------------------------- lifecycle

    @property
    def engine_core(self) -> Any:
        return self.engine.engine_core.engine_core

    @property
    def scheduler(self) -> Any:
        return self.engine_core.scheduler

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._step_loop, name="omp-engine-step", daemon=True
        )
        self._thread.start()

    def stop(self, timeout_s: float = 30.0) -> None:
        self._running = False
        self._wake.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout_s)
        self._thread = None

    # -------------------------------------------------------------- stepping

    @contextlib.contextmanager
    def priority_lock(self) -> Iterator[None]:
        """Acquires the engine lock ahead of the step thread's next iteration."""
        with self._turnstile:
            self.lock.acquire()
        try:
            yield
        finally:
            self.lock.release()

    def capacity_snapshot(self) -> Dict[str, Any]:
        """Lock-free view published by the step thread after every iteration."""
        return self._capacity_snapshot

    def _publish_snapshot(self) -> None:
        scheduler = self.engine_core.scheduler
        pool = scheduler.kv_cache_manager.block_pool
        self._capacity_snapshot = {
            "total_blocks": int(pool.num_gpu_blocks),
            "free_blocks": int(pool.get_num_free_blocks()),
            "num_running": len(scheduler.running),
            "num_waiting": len(scheduler.waiting),
            "published_at": time.time(),
            "step_count": self._step_count,
        }

    def _step_loop(self) -> None:
        while self._running:
            try:
                with self._turnstile:
                    self.lock.acquire()
                try:
                    # Client outputs can finish before the native async queue:
                    # step_with_batch_queue submits N+1 before retiring N. Keep
                    # stepping its tail and finished-request cleanup even after
                    # the output processor has delivered the last user token.
                    core = self.engine_core
                    has_work = (
                        self.engine.has_unfinished_requests()
                        or bool(core.batch_queue)
                        or core.scheduler.has_requests()
                    )
                    outputs = self.engine.step() if has_work else []
                    self._step_count += 1 if has_work else 0
                    self._publish_snapshot()
                finally:
                    self.lock.release()
                if outputs:
                    self._last_step_s = time.time()
                    self.loop.call_soon_threadsafe(self._deliver, outputs)
                if not has_work:
                    self._wake.wait(timeout=0.05)
                    self._wake.clear()
            except BaseException as exc:  # noqa: BLE001 - surfaced to callers
                self._error = exc
                self._fail_all(exc)
                self._running = False

    def _deliver(self, outputs: List[Any]) -> None:
        # The frontend detokenizer is seeded for byte/token boundary correctness,
        # but consumers must never receive the already committed prefix again.
        for output in outputs:
            stream = self._streams.get(output.request_id)
            if stream is None:
                continue
            for completion in output.outputs:
                if stream.prefix_tokens:
                    stream.full_output_text = completion.text
                    completion.token_ids = completion.token_ids[stream.prefix_tokens:]
                    completion.text = completion.text[stream.prefix_chars:]
                if (stream.qualification_top32 and stream.first_token_logprobs is None
                        and completion.token_ids and completion.logprobs):
                    # Only newly sampled positions have native logprobs. The
                    # seeded detokenizer prefix has no synthetic logprob rows.
                    stream.first_token_id = int(completion.token_ids[0])
                    stream.first_token_logprobs = {
                        int(token): float(value.logprob)
                        for token, value in completion.logprobs[0].items()
                    }
        for observer in self._iteration_observers:
            try:
                observer(outputs)
            except Exception:  # pragma: no cover - observers must not break serving
                pass
        for output in outputs:
            stream = self._streams.get(output.request_id)
            if stream is None:
                continue
            if stream.first_token_s is None and output.outputs and output.outputs[0].token_ids:
                stream.first_token_s = time.time() - stream.created_at
            stream.outputs = output
            if output.finished:
                self._streams.pop(output.request_id, None)
                if not stream.future.done():
                    stream.future.set_result(output)

    def _fail_all(self, exc: BaseException) -> None:
        def _do() -> None:
            for stream in list(self._streams.values()):
                if not stream.future.done():
                    stream.future.set_exception(exc)
            self._streams.clear()

        try:
            self.loop.call_soon_threadsafe(_do)
        except RuntimeError:  # pragma: no cover - loop already closed
            pass

    # --------------------------------------------------------------- serving

    async def generate(self, request_id: str, prompt: Any, params: Any) -> Any:
        """Submits one request and awaits its final output."""
        if self._error is not None:
            raise RuntimeError(f"engine step loop has failed: {self._error}")
        stream = RequestStream(request_id, self.loop)
        with self.priority_lock():
            if request_id in self._streams:
                raise RuntimeError("request already has a frontend stream")
            self.engine.add_request(request_id, prompt, params)
            self._streams[request_id] = stream
        self._wake.set()
        try:
            output = await stream.future
        except asyncio.CancelledError:
            self.abort(request_id)
            raise
        return output, stream.first_token_s

    def prepare_continuation(self, request_id: str, prompt: Any, params: Any,
                             generated_token_ids: List[int]) -> Any:
        """Build a real Request without registering frontend or scheduler work."""
        from vllm.v1.request import RequestStatus

        if self._error is not None:
            raise RuntimeError(f"engine step loop has failed: {self._error}")
        with self.priority_lock():
            frontend = self.engine.input_processor.process_inputs(
                request_id, prompt, params,
                supported_tasks=self.engine.get_supported_tasks(),
            )
            self.engine.input_processor.assign_request_id(frontend)
            if frontend.request_id != request_id:
                raise RuntimeError("continuation requires disabled request ID randomization")
            native, wave = self.engine_core.preprocess_add_request(frontend)
            native.append_output_token_ids(generated_token_ids)
            # NewRequestData carries no output history. Workers are explicitly
            # seeded before admission, and the native resumed path carries all
            # token IDs, the prompt boundary and original generation budget.
            native.status = RequestStatus.PREEMPTED
            native.num_preemptions = 1
            return frontend, native, wave

    def activate_continuation(self, frontend: Any, native: Any, wave: int) -> RequestStream:
        """Register and admit prepared state; caller holds the native step lock."""
        request_id = frontend.request_id
        prior = self._streams.get(request_id)
        if prior is not None:
            return prior
        if self._error is not None:
            raise RuntimeError(f"engine step loop has failed: {self._error}")
        stream = RequestStream(request_id, self.loop)
        stream.qualification_top32 = bool(getattr(native, "omp_qualification_top32", False))
        processor = self.engine.output_processor
        processor.add_request(frontend, None, None, 0)
        try:
            detokenizer = processor.request_states[request_id].detokenizer
            detokenizer.update(list(native.output_token_ids), False)
            stream.prefix_tokens = len(native.output_token_ids)
            stream.prefix_chars = len(detokenizer.get_next_output_text(False, False))
            self._streams[request_id] = stream
            self.engine_core.add_request(native, wave)
        except BaseException:
            if self.scheduler.requests.get(request_id) is not native:
                self._streams.pop(request_id, None)
                processor.abort_requests([request_id], internal=True)
                stream.future.cancel()
            raise
        self._wake.set()
        return stream

    def detach(self, request_id: str) -> bool:
        """Resolves a stream with its latest partial output (migrated away).

        The gateway owns the logical request across the handoff; the source's
        awaiting caller must not hang after its KV is released.
        """
        stream = self._streams.pop(request_id, None)
        if stream is None:
            return False
        if not stream.future.done():
            stream.future.set_result(stream.outputs)
        return True

    def abort(self, request_id: str) -> None:
        with self.priority_lock():
            self.engine.abort_request([request_id])
        stream = self._streams.pop(request_id, None)
        if stream is not None and not stream.future.done():
            stream.future.cancel()

    # ------------------------------------------------------------- telemetry

    def add_iteration_observer(self, observer: Callable[[List[Any]], None]) -> None:
        self._iteration_observers.append(observer)

    def clear_iteration_observers(self) -> None:
        self._iteration_observers.clear()

    def stats(self) -> Dict[str, Any]:
        return {
            "steps": self._step_count,
            "inflight_streams": len(self._streams),
            "running": self._running,
            "last_output_at_s": self._last_step_s,
            "error": repr(self._error) if self._error else None,
        }
