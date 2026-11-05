#!/usr/bin/env python3
"""TPUFlex changes to vLLM's own source (vllm 0.29.0 as shipped in vllm-torchtpu 0.29.0).

Each patch replaces one exact block of an installed file. New paths require a PM_* environment variable
or an explicit prepare_cpu_only/launcher handoff, so ordinary calls retain the stock path.
Most TPUFlex mechanisms live in the plugin (pm_torchtpu_plugin.py); a patch is used where a plugin hook
cannot reach, e.g. reordering statements inside a constructor.

  pm_source_patches.py apply SITE_PACKAGES      apply every patch not yet applied (idempotent)
  pm_source_patches.py count                    changed lines per patch (for the LoC table)
"""
import difflib
import pathlib
import sys

PATCHES = [
    {
        'name': 'overlap_renderer',
        'file': 'vllm/v1/engine/async_llm.py',
        'superseded_by': 'prepare_cpu_renderer',
        'old': '''\
        self.renderer = renderer = renderer_from_config(self.vllm_config)

        # Convert EngineInput --> EngineCoreRequest.
''',
        'new': '''\
        overlap_renderer = os.environ.get("PM_OVERLAP_RENDERER") == "1"
        if overlap_renderer:
            # TPUFlex (Overlap): build the renderer (tokenizer load plus the tokenizer pool's deep
            # copies, ~9 s for Gemma 4) while the engine core starts, not before launching it.
            import concurrent.futures

            pool = concurrent.futures.ThreadPoolExecutor(1, thread_name_prefix="pm-renderer")
            renderer_future = pool.submit(renderer_from_config, self.vllm_config)
            try:
                self.engine_core = EngineCoreClient.make_async_mp_client(
                    vllm_config=vllm_config,
                    executor_class=executor_class,
                    log_stats=self.log_stats,
                    client_addresses=client_addresses,
                    client_count=client_count,
                    client_index=client_index,
                )
            except BaseException:
                pool.shutdown(wait=False)
                raise
            self.renderer = renderer = renderer_future.result()
            pool.shutdown()
        else:
            self.renderer = renderer = renderer_from_config(self.vllm_config)

        # Convert EngineInput --> EngineCoreRequest.
''',
    },
    {
        'name': 'overlap_renderer',
        'file': 'vllm/v1/engine/async_llm.py',
        'superseded_by': 'prepare_cpu_resume',
        'old': '''\
        # EngineCore (starts the engine in background process).
        self.engine_core = EngineCoreClient.make_async_mp_client(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=self.log_stats,
            client_addresses=client_addresses,
            client_count=client_count,
            client_index=client_index,
        )
''',
        'new': '''\
        # EngineCore (starts the engine in background process).
        if not overlap_renderer:
            self.engine_core = EngineCoreClient.make_async_mp_client(
                vllm_config=vllm_config,
                executor_class=executor_class,
                log_stats=self.log_stats,
                client_addresses=client_addresses,
                client_count=client_count,
                client_index=client_index,
            )
''',
    },
    {
        'name': 'fast_replay',
        'file': 'vllm_torchtpu/compilation/tpu_compiler.py',
        'old': '''\
        logger.info("[TpuCompilerAdaptor] Loading compiled artifact from %s", path)

        # Same deepcopy + output-tuple rewrite as `compile`, so the graph the
        # artifact is checked against is shaped the way it was when saved.
        graph = copy.deepcopy(graph)
        graph, was_wrapped = _ensure_tuple_output(graph)

        compiled_fn = self._replay(path, graph, was_wrapped)
        if compiled_fn is None:
''',
        'new': '''\
        logger.info("[TpuCompilerAdaptor] Loading compiled artifact from %s", path)

        compiled_fn = None
        if os.environ.get("PM_FAST_REPLAY") == "1":
            # TPUFlex (Cache): a replay only checks the placeholders and whether the output is a
            # single value, so check the artifact against the graph as is. The deepcopy and the
            # recompile of the tuple rewrite are needed only if it must be compiled after all.
            out = next((n for n in graph.graph.nodes if n.op == "output"), None)
            was_wrapped = out is not None and not isinstance(out.args[0], (tuple, list))
            compiled_fn = self._replay(path, graph, was_wrapped)
            if compiled_fn is None and example_inputs is None:
                return None  # the caller compiles, with real example inputs
        if compiled_fn is None:
            # Same deepcopy + output-tuple rewrite as `compile`, so the graph the
            # artifact is checked against is shaped the way it was when saved.
            graph = copy.deepcopy(graph)
            graph, was_wrapped = _ensure_tuple_output(graph)
            compiled_fn = self._replay(path, graph, was_wrapped)
        if compiled_fn is None:
''',
    },
    {
        'name': 'fast_replay',
        'file': 'vllm/compilation/piecewise_backend.py',
        'old': '''\
            self._log_compile_start(range_entry.compile_range)

            if range_entry.compile_range.is_single_size():
''',
        'new': '''\
            self._log_compile_start(range_entry.compile_range)

            import os

            if os.environ.get("PM_FAST_REPLAY") == "1":
                # TPUFlex (Cache): try the compile cache before building concrete example
                # inputs (~0.36 s per range); only a real compile needs them.
                runnable = self.vllm_backend.compiler_manager.load(
                    self.graph, None, self.piecewise_compile_index, range_entry.compile_range
                )
                if runnable is not None:
                    range_entry.runnable = runnable
                    range_entry.compiled = True
                    continue

            if range_entry.compile_range.is_single_size():
''',
    },
    {
        'name': 'skip_mm_probe',
        'file': 'vllm/multimodal/registry.py',
        'old': '''\
        mm_config = model_config.get_multimodal_config()
        try:
            info = self._create_processing_info(model_config, tokenizer=None)
''',
        'new': '''\
        mm_config = model_config.get_multimodal_config()
        import os

        if (
            os.environ.get("PM_SKIP_MM_PROBE") == "1"
            and mm_config.language_model_only
            and not mm_config.enable_mm_embeds
        ):
            # TPUFlex: --language-model-only sets every modality limit to 0, so the answer below
            # is text-only; skip building the processing info (it loads the tokenizer, ~1.3 s).
            logger.info_once(
                "All limits of multimodal modalities supported by the model "
                "are set to 0, running in text-only mode."
            )
            return False
        try:
            info = self._create_processing_info(model_config, tokenizer=None)
''',
    },
    {
        'name': 'rewind_deferred_tpu_init',
        'file': 'vllm_torchtpu/worker/tpu_worker.py',
        'old': '''\
        if not self.devices:
            self.devices = [torch.device("tpu")]
''',
        'new': '''\
        if os.environ.get("PM_REWIND_DEFER_TPU_INIT") == "1":
            # CPU templates disable autoload. Bind the real SDK only here, after
            # init_device has installed this worker's final rank/topology env.
            from torch_tpu import _loader as pm_tpu_loader

            if os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") != "0":
                raise RuntimeError("rewind requires autoload disabled before worker binding")
            if (pm_tpu_loader._LOADED or pm_tpu_loader._DEVICE_INITIALIZED
                    or torch.distributed.is_initialized()
                    or torch.tpu.is_initialized() or torch.tpu._is_in_bad_fork()):
                raise RuntimeError("TorchTPU initialized/registered before final worker binding")
            pm_tpu_loader.load()
            if not (pm_tpu_loader._LOADED and pm_tpu_loader._DEVICE_INITIALIZED):
                raise RuntimeError("real TorchTPU backend registration did not complete")
            logger.info(
                "PM_REWIND_TPU_INIT rank=%s pid=%s client_initialized=%s",
                self.rank, os.getpid(), torch.tpu.is_initialized(),
            )
        if not self.devices:
            self.devices = [torch.device("tpu")]
''',
    },
    {
        'name': 'prepare_cpu_constructor',
        'file': 'vllm/v1/engine/async_llm.py',
        'old': '''\
        client_index: int = 0,
    ) -> None:
        """
        Create an AsyncLLM.
''',
        'new': '''\
        client_index: int = 0,
        *,
        prepare_cpu_only: bool = False,
    ) -> None:
        """
        Create an AsyncLLM.
''',
    },
    {
        'name': 'prepare_cpu_resources',
        'file': 'vllm/v1/engine/async_llm.py',
        'old': '''\
        # Ensure we can serialize custom transformer configs
        maybe_register_config_serialize_by_value()
''',
        'new': '''\
        # TPUFlex: this is a quiescent native-snapshot boundary, not a
        # serving engine or a portable Python serialization format. Keep the
        # real config/renderer/tokenizers intact; never warm their async pools
        # before capture. The caller owns final binding and resource checks.
        self._pm_cpu_state = "preparing" if prepare_cpu_only else None
        self.engine_core = None
        self.output_handler = None
        self.logger_manager = None
        self.profiler = None
        if prepare_cpu_only:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                pass
            else:
                raise RuntimeError("CPU preparation must run outside an event loop")
            self._pm_validate_cpu_config(
                vllm_config, client_addresses, client_count, client_index
            )
            if stat_loggers or mm_registry is not MULTIMODAL_REGISTRY:
                raise ValueError("CPU preparation rejects custom loggers/MM registries")
            # Inspect metadata only: loading a plugin factory can itself
            # allocate resources, even before a logger instance is created.
            from importlib.metadata import entry_points

            if entry_points(group="vllm.stat_logger_plugins"):
                raise ValueError("CPU preparation rejects stat logger plugins")

        # Ensure we can serialize custom transformer configs
        maybe_register_config_serialize_by_value()
''',
    },
    {
        'name': 'prepare_cpu_stat_plugins',
        'file': 'vllm/v1/engine/async_llm.py',
        'old': '''\
        custom_stat_loggers = list(stat_loggers or [])
        custom_stat_loggers.extend(load_stat_logger_plugin_factories())
''',
        'new': '''\
        custom_stat_loggers = list(stat_loggers or [])
        if not prepare_cpu_only:
            custom_stat_loggers.extend(load_stat_logger_plugin_factories())
''',
    },
    {
        'name': 'prepare_cpu_renderer',
        'file': 'vllm/v1/engine/async_llm.py',
        'old': '''\
        overlap_renderer = os.environ.get("PM_OVERLAP_RENDERER") == "1"
        if overlap_renderer:
            # TPUFlex (Overlap): build the renderer (tokenizer load plus the tokenizer pool's deep
            # copies, ~9 s for Gemma 4) while the engine core starts, not before launching it.
            import concurrent.futures

            pool = concurrent.futures.ThreadPoolExecutor(1, thread_name_prefix="pm-renderer")
            renderer_future = pool.submit(renderer_from_config, self.vllm_config)
            try:
                self.engine_core = EngineCoreClient.make_async_mp_client(
                    vllm_config=vllm_config,
                    executor_class=executor_class,
                    log_stats=self.log_stats,
                    client_addresses=client_addresses,
                    client_count=client_count,
                    client_index=client_index,
                )
            except BaseException:
                pool.shutdown(wait=False)
                raise
            self.renderer = renderer = renderer_future.result()
            pool.shutdown()
        else:
            self.renderer = renderer = renderer_from_config(self.vllm_config)

        # Convert EngineInput --> EngineCoreRequest.
''',
        'new': '''\
        overlap_renderer = (
            not prepare_cpu_only and os.environ.get("PM_OVERLAP_RENDERER") == "1"
        )
        if overlap_renderer:
            # TPUFlex (Overlap): build the renderer (tokenizer load plus the tokenizer pool's deep
            # copies, ~9 s for Gemma 4) while the engine core starts, not before launching it.
            import concurrent.futures

            pool = concurrent.futures.ThreadPoolExecutor(1, thread_name_prefix="pm-renderer")
            renderer_future = pool.submit(renderer_from_config, self.vllm_config)
            try:
                self.engine_core = EngineCoreClient.make_async_mp_client(
                    vllm_config=vllm_config,
                    executor_class=executor_class,
                    log_stats=self.log_stats,
                    client_addresses=client_addresses,
                    client_count=client_count,
                    client_index=client_index,
                )
            except BaseException:
                pool.shutdown(wait=False)
                raise
            self.renderer = renderer = renderer_future.result()
            pool.shutdown()
        else:
            self.renderer = renderer = renderer_from_config(self.vllm_config)

        # Convert EngineInput --> EngineCoreRequest.
''',
    },
    {
        'name': 'prepare_cpu_resume',
        'file': 'vllm/v1/engine/async_llm.py',
        'old': '''\
        # EngineCore (starts the engine in background process).
        if not overlap_renderer:
            self.engine_core = EngineCoreClient.make_async_mp_client(
                vllm_config=vllm_config,
                executor_class=executor_class,
                log_stats=self.log_stats,
                client_addresses=client_addresses,
                client_count=client_count,
                client_index=client_index,
            )
''',
        'new': '''\
        if prepare_cpu_only:
            self._pm_cpu_options = (
                executor_class, log_stats, custom_stat_loggers,
                aggregate_engine_logging, client_addresses, client_count, client_index,
            )
            self._pm_cpu_state = "prepared"
            return

        self._pm_start_engine(
            executor_class, log_stats, custom_stat_loggers, aggregate_engine_logging,
            client_addresses, client_count, client_index,
            overlap_renderer=overlap_renderer,
        )

    @staticmethod
    def _pm_validate_cpu_executor(backend) -> None:
        # TPU platform normalization has already imported this exact SDK
        # class. A bare "ray" would select vLLM's generic executor instead.
        # Check identity without importing a custom executor/plugin.
        from sys import modules

        module = modules.get(
            "vllm_torchtpu.executors.ray_distributed_executor_v2"
        )
        if (
            not isinstance(backend, type) or module is None
            or backend is not getattr(module, "RayDistributedExecutorV2", None)
        ):
            raise ValueError("CPU preparation requires the SDK RayDistributedExecutorV2")

    @staticmethod
    def _pm_validate_cpu_config(
        vllm_config, client_addresses, client_count, client_index
    ) -> None:
        # The fixed Gemma4 TP8 Ray template has one API and no PP/DP
        # coordination. General plugins and model-file fingerprinting are the
        # prefix factory's job; reject resource-producing frontend hooks here.
        parallel = vllm_config.parallel_config
        if (
            client_count != 1 or client_index != 0 or client_addresses is not None
            or parallel.tensor_parallel_size != 8
            or parallel.pipeline_parallel_size != 1
            or parallel.data_parallel_size != 1
            or parallel.data_parallel_backend != "mp"
            or parallel.data_parallel_size_local != 1
            or parallel.data_parallel_rank != 0
            or parallel.data_parallel_external_lb
            or parallel.data_parallel_hybrid_lb
            or parallel.enable_elastic_ep
            or parallel._api_process_count != 1
            or parallel._api_process_rank != 0
        ):
            raise ValueError("CPU preparation requires TP8/PP1/DP1 and one local API")
        AsyncLLM._pm_validate_cpu_executor(parallel.distributed_executor_backend)
        if vllm_config.observability_config.otlp_traces_endpoint is not None:
            raise ValueError("CPU preparation rejects tracing")
        if vllm_config.profiler_config.profiler is not None:
            raise ValueError("CPU preparation rejects profiling")
        model = vllm_config.model_config
        if model.trust_remote_code:
            raise ValueError("CPU preparation rejects remote model/tokenizer code")
        mm = model.multimodal_config
        if mm is not None and (
            mm.enable_mm_embeds
            or any(
                mm.get_limit_per_prompt(modality) != 0
                for modality in {"image", "video", "audio"} | mm.limit_per_prompt.keys()
            )
            or mm.mm_processor_cache_type == "shm" or mm.mm_ipc_gpu_memory_gb != 0
        ):
            # Gemma4's MM config may exist with all input limits zero. That
            # is the real SDK text-only path, not a reason to replace it.
            # Active MM with SHM caching instead constructs a real
            # ShmObjectStoreSenderCache before the engine exists.
            raise ValueError("CPU preparation requires text-only, non-IPC rendering")
        from vllm.renderers.registry import RENDERER_REGISTRY
        from vllm.tokenizers.registry import TokenizerRegistry, tokenizer_args_from_config

        renderer_mode, *_ = tokenizer_args_from_config(model)
        if (
            renderer_mode != "hf"
            or RENDERER_REGISTRY.renderers.get("hf")
            != ("vllm.renderers.hf", "HfRenderer")
            or TokenizerRegistry.tokenizers.get("hf")
            != ("vllm.tokenizers.hf", "CachedHfTokenizer")
        ):
            raise ValueError("CPU preparation requires the built-in HF renderer/tokenizer")

    def resume_cpu(self) -> None:
        # One shot, on the restored API event loop. The caller must use
        # shutdown in a finally, including when the real engine start fails.
        # Do not mutate the retained config, renderer, or processor objects
        # between preparation and resume; no re-construction/comparison occurs.
        if getattr(self, "_pm_cpu_state", None) != "prepared":
            raise RuntimeError("resume_cpu requires an unused CPU-prepared AsyncLLM")
        self._pm_cpu_state = "starting"
        try:
            asyncio.get_running_loop()
            options = self._pm_cpu_options
            self._pm_validate_cpu_config(self.vllm_config, *options[-3:])
            if self.engine_core is not None or self.output_handler is not None:
                raise RuntimeError("CPU-prepared AsyncLLM already owns engine resources")
            self._pm_start_engine(*options, overlap_renderer=False)
            # The stock suffix catches RuntimeError around output-handler
            # startup. A failed handler must not count as a successful resume.
            if self.output_handler is None:
                raise RuntimeError("CPU resume did not start the output handler")
        except BaseException:
            self._pm_cpu_state = "failed"
            raise
        else:
            self._pm_cpu_state = "started"
            del self._pm_cpu_options

    def _pm_start_engine(
        self, executor_class, log_stats, custom_stat_loggers,
        aggregate_engine_logging, client_addresses, client_count, client_index,
        *, overlap_renderer,
    ) -> None:
        # Both ordinary construction and resume use the SAME original suffix:
        # real EngineCoreClient, logger manager, output handler and profiler.
        vllm_config = self.vllm_config
        # EngineCore (starts the engine in background process).
        if not overlap_renderer:
            self.engine_core = EngineCoreClient.make_async_mp_client(
                vllm_config=vllm_config,
                executor_class=executor_class,
                log_stats=self.log_stats,
                client_addresses=client_addresses,
                client_count=client_count,
                client_index=client_index,
            )
''',
    },
    {
        'name': 'prepare_cpu_from_config',
        'file': 'vllm/v1/engine/async_llm.py',
        'old': '''\
        client_index: int = 0,
    ) -> "AsyncLLM":
        # Create the LLMEngine.
''',
        'new': '''\
        client_index: int = 0,
        *,
        prepare_cpu_only: bool = False,
    ) -> "AsyncLLM":
        # Reject custom executor imports before Executor.get_class can resolve
        # a plugin qualname; worker resource initialization remains in resume.
        if prepare_cpu_only:
            cls._pm_validate_cpu_executor(
                vllm_config.parallel_config.distributed_executor_backend
            )
        # Create the LLMEngine.
''',
    },
    {
        'name': 'prepare_cpu_from_config_forward',
        'file': 'vllm/v1/engine/async_llm.py',
        'old': '''\
            client_index=client_index,
        )

    @classmethod
    def from_engine_args(
''',
        'new': '''\
            client_index=client_index,
            prepare_cpu_only=prepare_cpu_only,
        )

    @classmethod
    def from_engine_args(
''',
    },
    {
        'name': 'prepare_cpu_shutdown',
        'file': 'vllm/v1/engine/async_llm.py',
        'old': '''\
        """Shutdown, cleaning up the background proc and IPC."""
        shutdown_prometheus()
''',
        'new': '''\
        """Shutdown, cleaning up the background proc and IPC."""
        if getattr(self, "_pm_cpu_state", None) is not None:
            # A rejected/CPU-only constructor owns no Prometheus engine state.
            # Close only its actual resources; every cleanup is attempted even
            # if an earlier one raises. Mark closed before calling callbacks so
            # explicit shutdown followed by __del__ is safe and one-shot.
            if self._pm_cpu_state == "closed":
                return
            self._pm_cpu_state = "closed"
            try:
                if getattr(self, "engine_core", None) is not None:
                    shutdown_prometheus()
            finally:
                try:
                    if renderer := getattr(self, "renderer", None):
                        renderer.shutdown()
                finally:
                    try:
                        if engine_core := getattr(self, "engine_core", None):
                            engine_core.shutdown(timeout=timeout)
                    finally:
                        handler = getattr(self, "output_handler", None)
                        if handler is not None:
                            cancel_task_threadsafe(handler)
            return
        shutdown_prometheus()
''',
    },
    {
        'name': 'prepare_cpu_launcher_handoff',
        'file': 'vllm/entrypoints/launchers/api_server/entry.py',
        'old': '''\
    if os.getenv("VLLM_WORKER_MULTIPROC_METHOD") == "forkserver":
''',
        'new': '''\
    prepared = getattr(args, "_pm_rewind_cpu", None)
    if prepared is not None:
        # The retained tuple is a one-shot ownership handoff, not a config
        # cache. Do not reconstruct engine args, VllmConfig, or the renderer.
        if not isinstance(prepared, tuple) or len(prepared) != 2:
            raise ValueError("_pm_rewind_cpu must be (engine_args, async_llm)")
        engine_args, prepared_engine = prepared
        if prepared_engine is None:
            raise ValueError("_pm_rewind_cpu requires a prepared AsyncLLM")
        # Reject an explicit scale-out request before resuming; this guard is
        # outside the inner owner, so it must release the prepared object too.
        if getattr(args, "api_server_count", None) not in (None, 1):
            try:
                raise ValueError("prepared handoff requires api_server_count=1")
            finally:
                prepared_engine.shutdown(
                    timeout=prepared_engine.vllm_config.shutdown_timeout
                )
        async with build_async_engine_client_from_engine_args(
            engine_args,
            usage_context=usage_context,
            client_config=client_config,
            prepared_engine=prepared_engine,
        ) as engine:
            yield engine
        return

    if os.getenv("VLLM_WORKER_MULTIPROC_METHOD") == "forkserver":
''',
    },
    {
        'name': 'prepare_cpu_launcher_resume',
        'file': 'vllm/entrypoints/launchers/api_server/entry.py',
        'old': '''\
async def build_async_engine_client_from_engine_args(
    engine_args: AsyncEngineArgs,
    *,
    usage_context: UsageContext = UsageContext.OPENAI_API_SERVER,
    client_config: dict[str, Any] | None = None,
) -> AsyncIterator[EngineClient]:
    """
    Create EngineClient, either:
        - in-process using the AsyncLLMEngine Directly
        - multiprocess using AsyncLLMEngine RPC

    Returns the Client or None if the creation failed.
    """

    # Create the EngineConfig (determines if we can use V1).
    vllm_config = engine_args.create_engine_config(usage_context=usage_context)

    from vllm.v1.engine.async_llm import AsyncLLM

    async_llm: AsyncLLM | None = None

    # Don't mutate the input client_config
    client_config = dict(client_config) if client_config else {}
    client_count = client_config.pop("client_count", 1)
    client_index = client_config.pop("client_index", 0)

    try:
        async_llm = AsyncLLM.from_vllm_config(
            vllm_config=vllm_config,
            usage_context=usage_context,
            enable_log_requests=engine_args.enable_log_requests,
            aggregate_engine_logging=engine_args.aggregate_engine_logging,
            disable_log_stats=engine_args.disable_log_stats,
            client_addresses=client_config,
            client_count=client_count,
            client_index=client_index,
        )

        # Don't keep the dummy data in memory
        assert async_llm is not None
        await async_llm.reset_mm_cache()

        yield async_llm
    finally:
        if async_llm:
            async_llm.shutdown(timeout=vllm_config.shutdown_timeout)
''',
        'new': '''\
async def build_async_engine_client_from_engine_args(
    engine_args: AsyncEngineArgs,
    *,
    usage_context: UsageContext = UsageContext.OPENAI_API_SERVER,
    client_config: dict[str, Any] | None = None,
    prepared_engine: EngineClient | None = None,
) -> AsyncIterator[EngineClient]:
    """
    Create EngineClient, either:
        - in-process using the AsyncLLMEngine Directly
        - multiprocess using AsyncLLMEngine RPC

    Returns the Client or None if the creation failed.
    """

    # Own the prepared instance BEFORE any validation or resume can fail.
    # Its real shutdown must run even if real engine construction raises.
    async_llm = prepared_engine
    try:
        # Create the EngineConfig only on the ordinary, non-prepared path.
        vllm_config = (
            prepared_engine.vllm_config if prepared_engine is not None
            else engine_args.create_engine_config(usage_context=usage_context)
        )

        from vllm.v1.engine.async_llm import AsyncLLM

        # Don't mutate the input client_config
        client_config = dict(client_config) if client_config else {}
        client_count = client_config.pop("client_count", 1)
        client_index = client_config.pop("client_index", 0)

        if prepared_engine is not None:
            if type(prepared_engine) is not AsyncLLM:
                raise TypeError("prepared_engine must be a real AsyncLLM")
            if type(engine_args) is not AsyncEngineArgs:
                raise TypeError("prepared handoff requires real AsyncEngineArgs")
            if (
                client_count != 1 or client_index != 0 or client_config
                or engine_args._api_process_count != 1
                or engine_args._api_process_rank != 0
            ):
                raise ValueError("prepared handoff supports only a single local API")
            prepared_engine.resume_cpu()
        else:
            async_llm = AsyncLLM.from_vllm_config(
                vllm_config=vllm_config,
                usage_context=usage_context,
                enable_log_requests=engine_args.enable_log_requests,
                aggregate_engine_logging=engine_args.aggregate_engine_logging,
                disable_log_stats=engine_args.disable_log_stats,
                client_addresses=client_config,
                client_count=client_count,
                client_index=client_index,
            )

        # Don't keep the dummy data in memory
        assert async_llm is not None
        await async_llm.reset_mm_cache()

        yield async_llm
    finally:
        if async_llm is not None:
            async_llm.shutdown(timeout=async_llm.vllm_config.shutdown_timeout)
''',
    },
]


def apply(site):
    for p in PATCHES:
        path = pathlib.Path(site) / p['file']
        text = path.read_text()
        # Only the full exact successor block proves an older overlap patch
        # has already been incorporated. A marker/name alone is insufficient.
        successor = p.get('superseded_by')
        if successor is not None and any(
            q['name'] == successor and q['file'] == p['file']
            and q['new'] in text for q in PATCHES
        ):
            print(f"PM_PATCH {p['name']} {p['file']}: already applied")
            continue
        if p['new'] in text:
            print(f"PM_PATCH {p['name']} {p['file']}: already applied")
            continue
        n = text.count(p['old'])
        if n != 1:
            raise SystemExit(f"PM_PATCH {p['name']} {p['file']}: expected the original block once, found {n}")
        path.write_text(text.replace(p['old'], p['new']))
        print(f"PM_PATCH {p['name']} {p['file']}: applied")


def count():
    per = {}
    for p in PATCHES:
        added = removed = 0
        for line in difflib.ndiff(p['old'].splitlines(), p['new'].splitlines()):
            body = line[2:].strip()
            if not body or body.startswith('#'):
                continue
            added += line.startswith('+ ')
            removed += line.startswith('- ')
        a, r = per.get(p['name'], (0, 0))
        per[p['name']] = (a + added, r + removed)
    for name, (a, r) in per.items():
        print(f'{name}: +{a} -{r} code lines')


if __name__ == '__main__':
    if sys.argv[1] == 'apply':
        apply(sys.argv[2])
    else:
        count()
