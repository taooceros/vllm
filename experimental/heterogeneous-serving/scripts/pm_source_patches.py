#!/usr/bin/env python3
"""TPUFlex changes to vLLM's own source (vllm 0.29.0 as shipped in vllm-torchtpu 0.29.0).

Each patch replaces one exact block of an installed file. Every new code path is gated by a PM_* environment
variable, so a patched venv behaves like stock vLLM unless the variable is set (stock arms stay stock).
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
]


def apply(site):
    for p in PATCHES:
        path = pathlib.Path(site) / p['file']
        text = path.read_text()
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
