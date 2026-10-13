"""Per-instance HTTP serving surface.

Each logical serving instance exposes exactly one reachable endpoint, bound on
its coordinator host.  Ray worker pods run with ``hostNetwork: true``, so
``podIP:8000`` is this host's address and the KubeRay readiness probe
(``GET :8000/health``) only passes once a real engine is serving.  That is what
turns "a Ray worker node is alive" into "a model endpoint is ready", and it is
what the gateway's pod-selector discovery resolves to a logical instance.

The same surface is the llm-d EPP contract: ``/metrics`` carries the vLLM
names the core metrics extractor reads plus ``omp:kv_*`` block facts, and
``/v1/completions`` accepts the standard OpenAI ``prompt`` as one int array of
token ids.  Each response choice also carries ``token_ids`` (the delivered
output token ids, as vLLM's ``return_token_ids``) and an ``omp`` object with
the serving instance, so a client behind Envoy can account every token exactly.
"""

from __future__ import annotations

import asyncio
import time
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from typing import Any, Awaitable, Callable, Dict, Optional


async def _client_disconnected(request: Request) -> None:
    """Returns once the client connection is gone; the request body is already read."""
    while (await request.receive())["type"] != "http.disconnect":
        pass


def build_app(engine: Any) -> Any:
    """Builds the FastAPI app for one engine actor."""
    app = FastAPI(title="omp-hetero-instance", docs_url=None, redoc_url=None)

    @app.get("/health")
    async def health() -> Any:
        status = engine.status()
        ready = status["state"] == "ready"
        return JSONResponse(
            status_code=200 if ready else 503,
            content={
                "status": "ok" if ready else status["state"],
                "instance_id": engine.instance_id,
                "generation": engine.generation,
                "layout_id": engine.layout_id,
            },
        )

    @app.get("/omp/status")
    async def omp_status() -> Any:
        return engine.status()

    @app.get("/omp/identity")
    async def omp_identity() -> Any:
        return engine.identity()

    @app.get("/metrics")
    async def metrics() -> Any:
        return PlainTextResponse(_render_metrics(engine))

    @app.get("/v1/models")
    async def models() -> Any:
        model = engine.settings.model
        return {
            "object": "list",
            "data": [{
                "id": model, "object": "model", "owned_by": "omp-hetero", "root": model,
                "max_model_len": int(engine.settings.max_model_len),
            }],
        }

    @app.post("/v1/completions")
    async def completions(request: Request) -> Any:
        body = await request.json()
        prompt = body.get("prompt")
        prompt_token_ids = body.get("prompt_token_ids")
        if isinstance(prompt, list):
            # OpenAI allows a batch (list of strings/arrays); only one token array is served.
            if prompt_token_ids is not None or not prompt or not all(
                    isinstance(t, int) and not isinstance(t, bool) for t in prompt):
                raise HTTPException(status_code=400,
                                    detail="prompt must be a string or one int array of token ids")
            prompt, prompt_token_ids = None, prompt
        if prompt is None and prompt_token_ids is None:
            raise HTTPException(status_code=400, detail="prompt or prompt_token_ids required")
        if body.get("stream"):
            raise HTTPException(status_code=400, detail="streaming is not supported")
        started = time.time()
        generation = asyncio.ensure_future(engine.generate(
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            max_tokens=int(body.get("max_tokens", 16)),
            temperature=float(body.get("temperature", 0.0)),
            seed=body.get("seed"),
            request_id=body.get("request_id"),
            ignore_eos=bool(body.get("ignore_eos", False)),
        ))
        disconnect = asyncio.ensure_future(_client_disconnected(request))
        try:
            await asyncio.wait({generation, disconnect}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            # Client gone (or this handler cancelled): cancelling the generate
            # coroutine aborts the native request, as for any cancelled caller,
            # so an abandoned request never keeps decoding on the device while
            # the router has already released its charge.
            disconnect.cancel()
            if not generation.done():
                generation.cancel()
                await asyncio.gather(generation, return_exceptions=True)
        if generation.cancelled():
            return Response(status_code=499)
        try:
            result = generation.result()
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {
            "id": result["request_id"],
            "object": "text_completion",
            "created": int(started),
            "model": body.get("model") or engine.settings.model,
            "choices": [
                {
                    "index": 0,
                    "text": result["text"],
                    "token_ids": result["output_token_ids"],
                    "finish_reason": result["finish_reason"],
                }
            ],
            "usage": {
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["output_tokens"],
                "total_tokens": result["prompt_tokens"] + result["output_tokens"],
            },
            "omp": {
                "instance_id": result["instance_id"],
                "generation": engine.generation,
                "ttft_s": result["ttft_s"],
                "elapsed_s": result["elapsed_s"],
                "started_at_s": result["started_at_s"],
                "finished_at_s": result["finished_at_s"],
            },
        }

    return app


def _render_metrics(engine: Any) -> str:
    status = engine.status()
    total = max(1, int(status.get("total_blocks", 1)))
    free = int(status.get("free_blocks", total))
    usage = 1.0 - (free / total)
    labels = f'{{instance="{engine.instance_id}",layout="{engine.layout_id}"}}'
    lines = [
        "# TYPE vllm:num_requests_running gauge",
        f"vllm:num_requests_running{labels} {status.get('num_running', 0)}",
        "# TYPE vllm:num_requests_waiting gauge",
        f"vllm:num_requests_waiting{labels} {status.get('num_waiting', 0)}",
        "# TYPE vllm:kv_cache_usage_perc gauge",
        f"vllm:kv_cache_usage_perc{labels} {usage:.6f}",
        "# TYPE omp:kv_free_blocks gauge",
        f"omp:kv_free_blocks{labels} {free}",
        "# TYPE omp:kv_total_blocks gauge",
        f"omp:kv_total_blocks{labels} {total}",
        "# TYPE omp:kv_block_size gauge",
        f"omp:kv_block_size{labels} {int(status.get('block_size', 0))}",
        "# TYPE omp:inflight_requests gauge",
        f"omp:inflight_requests{labels} {status.get('inflight', 0)}",
        "# TYPE omp:served_requests counter",
        f"omp:served_requests{labels} {status.get('served', 0)}",
    ]
    return "\n".join(lines) + "\n"


class ServingEndpoint:
    """Owns the uvicorn server task for one engine actor."""

    def __init__(self, engine: Any, host: str = "0.0.0.0", port: int = 8000) -> None:
        self.engine = engine
        self.host = host
        self.port = port
        self._server: Optional[Any] = None
        self._task: Optional[asyncio.Task] = None

    async def start(self, timeout_s: float = 60.0) -> Dict[str, Any]:
        import uvicorn

        if self._task is not None:
            return {"started": False, "reason": "already_running", "port": self.port}
        config = uvicorn.Config(
            build_app(self.engine),
            host=self.host,
            port=self.port,
            log_level="warning",
            access_log=False,
            timeout_keep_alive=600,
        )
        self._server = uvicorn.Server(config)
        self._task = asyncio.create_task(self._server.serve())
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if getattr(self._server, "started", False):
                return {"started": True, "host": self.host, "port": self.port}
            if self._task.done():
                raise RuntimeError(f"serving endpoint failed to start: {self._task.exception()}")
            await asyncio.sleep(0.1)
        raise RuntimeError(f"serving endpoint did not bind {self.host}:{self.port} in {timeout_s}s")

    async def stop(self, timeout_s: float = 30.0) -> None:
        if self._server is not None:
            self._server.should_exit = True
        task = self._task
        self._task = None
        self._server = None
        if task is not None:
            try:
                await asyncio.wait_for(task, timeout=timeout_s)
            except Exception:  # pragma: no cover - best effort
                task.cancel()
