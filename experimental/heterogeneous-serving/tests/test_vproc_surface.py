"""C3: surface scanner, VllmEngine mesh order and guards, vLLM view executor (no vLLM, no TPU)."""
import dataclasses
import glob
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from concurrent.futures import Future
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import vproc_surface_check as scan  # noqa: E402
from hetero.vproc import engine_vllm, vllm_executor  # noqa: E402
from hetero.vproc.types import EngineSpec  # noqa: E402

FORKS = Path.home() / "heterogeneous-serving-tpu"


def classes(source, module="tpu_inference.fixture", **kwargs):
    return [(s["api"], s["class"]) for s in scan.scan_source(source, module, **kwargs)]


class ScannerFixtures(unittest.TestCase):
    def test_public_api_is_covered_and_key_removal_uncovers(self):
        src = "import jax\nd = jax.devices()\nn = jax.process_count()\nf = jax.process_index\n"
        self.assertEqual(classes(src), [("jax.devices", "covered"), ("jax.process_count", "covered"),
                                        ("jax.process_index", "covered")])
        reduced = set(scan.PINNED_VIRTUALIZED) - {"jax.process_count"}
        self.assertIn(("jax.process_count", "UNCOVERED"), classes(src, virtualized=reduced))

    def test_alias_and_nested_calls(self):
        src = "import jax as j\nx = j.devices('cpu')[:1]\ny = j.local_devices(process_index=3)\n"
        self.assertEqual(classes(src), [("jax.devices", "covered"), ("jax.local_devices", "covered")])

    def test_local_device_count_is_harmless_when_not_virtualized(self):
        src = "import jax\nn = jax.local_device_count()\n"
        self.assertEqual(classes(src, virtualized=scan.PINNED_VIRTUALIZED), [("jax.local_device_count", "harmless")])

    def test_from_import_binding_is_uncovered_at_module_level_only(self):
        module_level = "from jax import devices\ndef f():\n    return devices()\n"
        rows = classes(module_level)
        self.assertEqual(rows, [("jax.devices", "UNCOVERED"), ("jax.devices", "UNCOVERED")])
        function_level = "def f():\n    from jax import devices\n    return devices()\n"
        self.assertEqual(classes(function_level), [("jax.devices", "covered")])

    def test_xla_bridge_proxy_only_in_tpu_inference_utils(self):
        src = "from jax._src import xla_bridge as xb\na = xb.devices()\nb = xb.process_index()\nc = xb.local_devices()\n"
        self.assertEqual(classes(src, module="tpu_inference.utils"),
                         [("tpu_inference.utils.xb.devices", "covered"),
                          ("tpu_inference.utils.xb.process_index", "UNCOVERED"),
                          ("tpu_inference.utils.xb.local_devices", "harmless")])
        other = classes(src, module="tpu_inference.other")
        self.assertEqual([c for _, c in other], ["UNCOVERED", "UNCOVERED", "harmless"])

    def test_device_process_index_is_covered_only_inside_the_wrapped_function(self):
        src = "def get_device_topology_order_id(local, global_):\n    return [d.process_index for d in global_]\n"
        self.assertEqual(classes(src, module="tpu_inference.distributed.utils"), [("d.process_index", "covered")])
        self.assertEqual(classes(src, module="tpu_inference.elsewhere"), [("d.process_index", "UNCOVERED")])
        call = "import jax\nbad = jax.devices()[0].process_index\n"
        self.assertEqual(dict(classes(call)), {"jax.devices": "covered", "<expr>.process_index": "UNCOVERED"})

    def test_distributed_and_coordination(self):
        src = ("import jax\nfrom jax._src import distributed\n"
               "jax.distributed.initialize()\n"
               "distributed.global_state.client.wait_at_barrier('b', 1)\n"
               "pid = distributed.global_state.process_id\n")
        rows = dict(classes(src))
        self.assertEqual(rows["jax.distributed.initialize"], "guarded")
        self.assertEqual(rows["jax._src.distributed.global_state.client.wait_at_barrier"], "covered")
        self.assertEqual(rows["jax._src.distributed.global_state.process_id"], "UNCOVERED")
        missing = {name: False for name in scan.ADAPTER_GUARDS}
        self.assertEqual(dict(classes(src, guards=missing))["jax.distributed.initialize"], "UNCOVERED")

    def test_multihost_utils_only_the_scoped_functions(self):
        src = ("from jax.experimental import multihost_utils\n"
               "multihost_utils.process_allgather(1)\nmultihost_utils.host_local_array_to_global_array(1, 2, 3)\n")
        self.assertEqual([c for _, c in classes(src)], ["covered", "UNCOVERED"])

    def test_tpu_env_writes(self):
        src = ("import os\nfrom os import environ\n"
               "class W:\n    def init_device(self):\n        os.environ[\n            'TPU_VISIBLE_CHIPS'] = '0'\n"
               "def other():\n    os.environ['TPU_PROCESS_BOUNDS'] = '1,1,1'\n"
               "    os.environ.update({'LIBTPU_INIT_ARGS': 'x', 'HARMLESS': '1'})\n"
               "    environ.setdefault('CLOUD_TPU_TASK_ID', '0')\n"
               "    os.environ['VLLM_FOO'] = '1'\n"
               "    os.environ[name] = 'dynamic'\n"
               "    del os.environ['XLA_FLAGS']\n")
        got = scan.scan_source(src, "tpu_inference.worker.tpu_worker")
        rows = [(s["line"], s["api"], s["class"]) for s in got]
        self.assertEqual(rows, [
            (5, "os.environ['TPU_VISIBLE_CHIPS']", "guarded"),
            (8, "os.environ['TPU_PROCESS_BOUNDS']", "UNCOVERED"),
            (9, "os.environ.update['LIBTPU_INIT_ARGS']", "UNCOVERED"),
            (10, "os.environ.setdefault['CLOUD_TPU_TASK_ID']", "UNCOVERED"),
            (12, "os.environ[<dynamic key>]", "UNCOVERED"),
            (13, "os.environ['XLA_FLAGS']", "UNCOVERED"),
        ])

    def test_env_override_import_time_writes_are_guarded_by_preinit(self):
        src = "import os\nos.environ['LIBTPU_INIT_ARGS'] = 'x ' + os.environ.get('LIBTPU_INIT_ARGS', '')\n"
        self.assertEqual(classes(src, module="tpu_inference.env_override"),
                         [("os.environ['LIBTPU_INIT_ARGS']", "guarded")])

    def test_cache_config_clear_caches_and_implicit_devices(self):
        src = ("import jax\n"
               "jax.config.update('jax_compilation_cache_dir', 'x')\n"
               "jax.config.update('jax_persistent_cache_min_compile_time_secs', 0)\n"
               "jax.config.update('jax_default_matmul_precision', 'highest')\n"
               "jax.clear_caches()\n"
               "m1 = jax.make_mesh((1, 8), ('a', 'b'))\n"
               "m2 = jax.make_mesh((1, 8), ('a', 'b'), devices=devs)\n")
        self.assertEqual(classes(src), [
            ("jax.config.update('jax_compilation_cache_dir')", "guarded"),
            ("jax.config.update('jax_persistent_cache_min_compile_time_secs')", "harmless"),
            ("jax.clear_caches", "harmless"),
            ("jax.make_mesh", "UNCOVERED")])

    def test_unrelated_code_has_no_sites(self):
        src = "import jax, os\nx = jax.numpy.ones(3)\ny = os.environ.get('TPU_X')\nz = os.environ['PATH']\n"
        self.assertEqual(classes(src), [])


class ScannerOnForks(unittest.TestCase):
    @unittest.skipUnless((FORKS / "tpu-inference/tpu_inference").is_dir() and (FORKS / "vllm/vllm").is_dir(),
                         "forks not checked out")
    def test_real_forks_write_the_report(self):
        out = REPO / "results/vproc/surface-check.json"
        code = scan.main(["--out", str(out)])
        report = json.loads(out.read_text())
        for name in ("tpu-inference", "vllm"):
            sha = subprocess.run(["git", "-C", str(FORKS / name), "rev-parse", "HEAD"], capture_output=True,
                                 text=True, check=True).stdout.strip()
            self.assertEqual(report["forks"][name]["sha"], sha)
        self.assertEqual(code, 1 if report["summary"]["uncovered_engine_path"] else 0)
        self.assertTrue(all(g["present"] for g in report["adapter_guards"].values()))
        by_loc = {(s["file"], s["line"], s["api"]): s for s in report["sites"]}
        key = ("tpu_inference/worker/tpu_worker.py", 410, "jax.devices")
        self.assertEqual(by_loc[key]["class"], "covered")
        self.assertEqual(by_loc[key]["fork_sha"], report["forks"]["tpu-inference"]["sha"])
        self.assertEqual(by_loc[("tpu_inference/worker/tpu_worker.py", 299, "os.environ['TPU_VISIBLE_CHIPS']")]["class"],
                         "guarded")
        self.assertEqual(by_loc[("tpu_inference/utils.py", 249, "tpu_inference.utils.xb.devices")]["class"], "covered")
        self.assertEqual(by_loc[("tpu_inference/layers/common/utils.py", 174, "jax.devices")]["class"], "covered")
        self.assertEqual(by_loc[("tpu_inference/env_override.py", 21, "os.environ['LIBTPU_INIT_ARGS']")]["class"],
                         "guarded")
        for site in report["sites"]:
            self.assertEqual(set(site) >= {"location", "snippet", "class", "fork_sha", "engine_path"}, True)
            self.assertIn(site["class"], {"covered", "guarded", "harmless", "UNCOVERED"})

    def test_exit_code_is_1_for_an_uncovered_engine_site(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "ti/tpu_inference").mkdir(parents=True)
            (root / "ti/tpu_inference/bad.py").write_text("from jax import devices\nprint(devices())\n")
            (root / "ti/tpu_inference/executors").mkdir()
            (root / "ti/tpu_inference/executors/ray.py").write_text("from jax import devices\n")
            (root / "vl/vllm").mkdir(parents=True)
            out = root / "out.json"
            self.assertEqual(scan.main(["--tpu-inference", str(root / "ti"), "--vllm", str(root / "vl"),
                                        "--out", str(out)]), 1)
            report = json.loads(out.read_text())
            self.assertEqual(report["summary"]["uncovered_engine_path"], 2)
            self.assertEqual(report["summary"]["by_class"], {"UNCOVERED": 3})  # off-path site is reported too
            (root / "ti/tpu_inference/bad.py").write_text("import jax\nprint(jax.devices())\n")
            self.assertEqual(scan.main(["--tpu-inference", str(root / "ti"), "--vllm", str(root / "vl"),
                                        "--out", str(out)]), 0)


class ImportsWithoutVllm(unittest.TestCase):
    def test_modules_import_and_executor_needs_vllm_lazily(self):
        code = (
            "import sys\n"
            "for name in ('vllm', 'tpu_inference', 'flax', 'torch'):\n"
            "    sys.modules[name] = None\n"
            "import hetero.vproc.engine_vllm as e, hetero.vproc.vllm_executor as x\n"
            "assert 'vllm' not in [m for m in sys.modules if sys.modules[m] is not None and m.startswith('vllm')]\n"
            "try:\n"
            "    x.VllmViewExecutor\n"
            "except ImportError:\n"
            "    print('lazy-ok')\n"
        )
        run = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True,
                             env={**os.environ, "JAX_PLATFORMS": "cpu"})
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("lazy-ok", run.stdout)


# --------------------------------------------------------------- mesh_device_ids


@dataclasses.dataclass(frozen=True)
class FakeTpu:
    id: int
    process_index: int
    coords: tuple
    core_on_chip: int = 0
    platform: str = "tpu"
    device_kind: str = "TPU v5 lite"

    def __repr__(self):
        return f"D{self.id}"


# deviceless v5e:4x4 (libtpu 0.0.43): chip (x, y) -> id; host = x // 2 + 2 * (y // 2)
V5E_ID = {(0, 0): 0, (1, 0): 1, (0, 1): 4, (1, 1): 5, (2, 0): 2, (3, 0): 3, (2, 1): 6, (3, 1): 7,
          (0, 2): 8, (1, 2): 9, (0, 3): 12, (1, 3): 13, (2, 2): 10, (3, 2): 11, (2, 3): 14, (3, 3): 15}


def v5e_4x4():
    devices = [FakeTpu(V5E_ID[x, y], x // 2 + 2 * (y // 2), (x, y, 0)) for y in range(4) for x in range(4)]
    return sorted(devices, key=lambda d: d.id)


def view_devices(hosts):
    """What jax.devices() shows under a view: the hosts' devices in physical (id) order."""
    return [d for d in v5e_4x4() if d.process_index in hosts]


# make_mesh on deviceless v5e:4x4 devices, jax 0.10.2 and 0.11.2 (identical): physical ring order
EXPECTED = {
    (0, 1): (0, 1, 2, 3, 7, 6, 5, 4),
    (2, 3): (8, 9, 10, 11, 15, 14, 13, 12),
    (0, 3): (0, 1, 4, 5, 15, 14, 11, 10),
    (1, 2): (2, 3, 6, 7, 13, 12, 9, 8),
    (0, 1, 2, 3): (0, 1, 2, 3, 7, 6, 5, 9, 10, 11, 15, 14, 13, 12, 8, 4),
}


def libtpu_dir():
    configured = os.environ.get("HETERO_LIBTPU_DIR")
    if configured and (Path(configured) / "libtpu/libtpu.so").exists():
        return configured
    for pattern in ("~/.cache/uv/archive-v0/*/libtpu-0.0.43*.dist-info",):
        for hit in sorted(glob.glob(os.path.expanduser(pattern))):
            root = Path(hit).parent
            if (root / "libtpu/libtpu.so").exists():
                return str(root)
    return None


class MeshDeviceIds(unittest.TestCase):
    def ids(self, hosts, tp=None, devices=None, **env):
        spec = EngineSpec("e", "vllm", tuple(hosts), tp or 4 * len(hosts))
        return engine_vllm.tp_mesh_device_ids(spec.tp_size, view_devices(hosts) if devices is None else devices, env) \
            if env else engine_vllm.VllmEngine.mesh_device_ids(spec, view_devices(hosts) if devices is None else devices)

    def test_block_layouts_on_fake_v5e_devices(self):
        for hosts, expected in EXPECTED.items():
            self.assertEqual(self.ids(hosts), expected, hosts)

    def test_block_a_order_differs_from_physical_order(self):
        physical = tuple(d.id for d in view_devices((0, 1)))  # jax.devices() order: by id
        self.assertEqual(physical, (0, 1, 2, 3, 4, 5, 6, 7))
        self.assertNotEqual(self.ids((0, 1)), physical)
        self.assertEqual(sorted(self.ids((0, 1))), sorted(physical))

    def test_engine_devices_are_the_first_tp_of_the_view(self):
        everything = v5e_4x4()  # tpu_worker.py:410 jax.devices()[:tp]
        spec = EngineSpec("e", "vllm", (0, 1), 8)
        self.assertEqual(engine_vllm.VllmEngine.mesh_device_ids(spec, everything), EXPECTED[(0, 1)])

    def test_input_order_within_the_engine_devices_does_not_matter(self):
        devices = list(reversed(view_devices((2, 3))))
        self.assertEqual(self.ids((2, 3), devices=devices), EXPECTED[(2, 3)])

    def test_new_model_design_sorted_by_coords(self):
        got = self.ids((0, 1), NEW_MODEL_DESIGN="1", TPU_MESH_SORT_BY_COORDS="1")
        self.assertEqual(got, (0, 1, 2, 3, 4, 5, 6, 7))  # (z, y, x) order

    def test_new_model_design_physical_mesh(self):
        got = self.ids((0, 1), NEW_MODEL_DESIGN="1")
        self.assertEqual(sorted(got), sorted(EXPECTED[(0, 1)]))

    def test_v6e_8_chip_ring(self):
        devices = [FakeTpu(i, 0, (x, y, 0), device_kind="TPU v6 lite")
                   for i, (x, y) in enumerate((x, y) for x in range(2) for y in range(4))]
        got = engine_vllm.tp_mesh_device_ids(8, devices, {})
        by_id = {d.id: d.coords[:2] for d in devices}
        self.assertEqual([by_id[i] for i in got],
                         [(0, 0), (0, 1), (0, 2), (0, 3), (1, 3), (1, 2), (1, 1), (1, 0)])

    def test_errors(self):
        with self.assertRaises(ValueError):
            engine_vllm.tp_mesh_device_ids(16, view_devices((0, 1)), {})
        import jax
        with self.assertRaises(ValueError):  # CPU devices have no coords: tpu-inference fails there too
            engine_vllm.tp_mesh_device_ids(1, jax.devices("cpu")[:1], {})
        with self.assertRaises(NotImplementedError):
            engine_vllm.tp_mesh_device_ids(8, view_devices((0, 1)), {"NUM_SLICES": "2"})

    @unittest.skipUnless(libtpu_dir(), "libtpu 0.0.43 not found (set HETERO_LIBTPU_DIR)")
    def test_matches_deviceless_v5e_4x4_topology(self):
        code = (
            "import json, sys\n"
            "from jax.experimental import topologies\n"
            "from hetero.vproc import engine_vllm\n"
            "from hetero.vproc.types import EngineSpec\n"
            "t = topologies.get_topology_desc(topology_name='v5e:4x4', platform='tpu')\n"
            "ds = sorted(t.devices, key=lambda d: d.id)\n"
            "out = {}\n"
            "for hosts in [(0, 1), (2, 3), (0, 3), (1, 2), (0, 1, 2, 3)]:\n"
            "    view = [d for d in ds if d.process_index in hosts]\n"
            "    spec = EngineSpec('e', 'vllm', hosts, 4 * len(hosts))\n"
            "    out[','.join(map(str, hosts))] = engine_vllm.VllmEngine.mesh_device_ids(spec, view)\n"
            "print('RESULT' + json.dumps(out))\n")
        run = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, timeout=280,
                             env={**os.environ, "JAX_PLATFORMS": "cpu", "PYTHONPATH": libtpu_dir()})
        self.assertEqual(run.returncode, 0, run.stderr[-2000:])
        line = [l for l in run.stdout.splitlines() if l.startswith("RESULT")][0]
        got = json.loads(line[len("RESULT"):])
        self.assertEqual({k: tuple(v) for k, v in got.items()},
                         {",".join(map(str, hosts)): ids for hosts, ids in EXPECTED.items()})
        # the fake devices above encode the same topology, so the unit tests above are the deviceless result



class ZeroCopyRewrap(unittest.TestCase):
    """rewrap_resident keeps every device buffer (8 forced CPU devices, subprocess)."""

    SCRIPT = r"""
import json
import numpy as np
import jax, jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from hetero.vproc import engine_vllm as e

def pointers(tree):
    return {jax.tree_util.keystr(path): tuple(sorted(int(sh.data.unsafe_buffer_pointer())
                                                     for sh in array.addressable_shards))
            for path, array in jax.tree_util.tree_flatten_with_path(tree)[0]}

devs = jax.devices()
order = [devs[i] for i in (0, 1, 2, 3, 7, 6, 5, 4)]          # a non-id TP order, as on v5e
mesh1 = Mesh(np.array(order), ("model",))                      # ResidentStore handover mesh
mesh2 = Mesh(np.array(order).reshape(1, 8), ("data", "model"))  # runner mesh
specs = {"w": P(None, "model"), "v": P("model"), "b": P()}
shapes = {"w": (4, 16), "v": (16,), "b": (3,)}
leaves = {"['%s']" % k: jax.device_put(jnp.arange(np.prod(shapes[k]), dtype=jnp.float32).reshape(shapes[k]),
                                       NamedSharding(mesh1, specs[k])) for k in specs}
abstract = {k: jax.ShapeDtypeStruct(shapes[k], jnp.float32) for k in specs}
wanted = {k: NamedSharding(mesh2, specs[k]) for k in specs}
out = {}
before = pointers({k: leaves["['%s']" % k] for k in specs})
wrapped = e.rewrap_resident(abstract, wanted, leaves)   # leaves keyed as in production
after = pointers(wrapped)
out["n_leaves"] = len(after)
out["same_pointers"] = before == after
out["on_runner_mesh"] = all(wrapped[k].sharding == wanted[k] for k in specs)
out["values_equal"] = all(bool((np.asarray(wrapped[k]) == np.asarray(leaves["['%s']" % k])).all()) for k in specs)
out["sharded_shards"] = len(wrapped["w"].addressable_shards)
# a runner mesh with a different device order must be refused (shard index changes)
bad = Mesh(np.array(order[::-1]).reshape(1, 8), ("data", "model"))
try:
    e.rewrap_resident(abstract, {k: NamedSharding(bad, specs[k]) for k in specs}, leaves)
    out["reversed"] = "accepted"
except ValueError as error:
    out["reversed"] = str(error)
# a runner mesh using a device the leaves do not live on
half = Mesh(np.array(devs[:4]).reshape(1, 4), ("data", "model"))
small = {k: jax.device_put(jnp.zeros(shapes[k], jnp.float32), NamedSharding(Mesh(np.array(devs[4:]), ("model",)), specs[k])) for k in specs}
try:
    e.rewrap_resident(abstract, {k: NamedSharding(half, specs[k]) for k in specs}, {"['%s']" % k: v for k, v in small.items()})
    out["foreign"] = "accepted"
except ValueError as error:
    out["foreign"] = str(error)
try:
    e.rewrap_resident(abstract, wanted, {"['w']": leaves["['w']"]})
    out["missing"] = "accepted"
except ValueError as error:
    out["missing"] = str(error)
print("RESULT" + json.dumps(out))
"""

    def test_rewrap_preserves_buffers_and_validates_indices(self):
        run = subprocess.run([sys.executable, "-c", self.SCRIPT], cwd=REPO, capture_output=True, text=True,
                             timeout=120, env={**os.environ, "JAX_PLATFORMS": "cpu",
                                               "XLA_FLAGS": "--xla_force_host_platform_device_count=8"})
        self.assertEqual(run.returncode, 0, run.stderr[-2000:])
        out = json.loads([l for l in run.stdout.splitlines() if l.startswith("RESULT")][0][len("RESULT"):])
        self.assertEqual(out["n_leaves"], 3)
        self.assertTrue(out["same_pointers"])       # no buffer moved
        self.assertTrue(out["on_runner_mesh"])      # and the arrays carry the runner's shardings
        self.assertTrue(out["values_equal"])
        self.assertEqual(out["sharded_shards"], 8)
        self.assertIn("changes device shard index", out["reversed"])
        self.assertIn("no shard on engine device", out["foreign"])
        self.assertIn("missing", out["missing"])


# ------------------------------------------------------------------------ guards


class Guards(unittest.TestCase):
    def test_forbid_distributed_restores_every_name(self):
        import jax.distributed
        from jax._src import distributed
        originals = (jax.distributed.initialize, distributed.initialize, distributed.shutdown)
        with engine_vllm.forbid_distributed() as attempts:
            with self.assertRaises(engine_vllm.EngineGuardError):
                jax.distributed.initialize("localhost:1", 1, 0)
            with self.assertRaises(engine_vllm.EngineGuardError):
                distributed.global_state.initialize("localhost:1", 1, 0)
            with self.assertRaises(engine_vllm.EngineGuardError):
                jax.distributed.shutdown()
        self.assertEqual(attempts, ["initialize", "initialize", "shutdown"])
        self.assertEqual((jax.distributed.initialize, distributed.initialize, distributed.shutdown), originals)
        self.assertNotIn("initialize", vars(distributed.global_state))

    def test_frozen_tpu_env_restores_and_raises(self):
        os.environ["TPU_VISIBLE_CHIPS"] = "0,1,2,3"
        os.environ.pop("TPU_PROCESS_BOUNDS", None)
        try:
            with self.assertRaises(engine_vllm.EngineGuardError) as caught:
                with engine_vllm.frozen_tpu_env():
                    os.environ["TPU_VISIBLE_CHIPS"] = "0"
                    os.environ["TPU_PROCESS_BOUNDS"] = "1,1,1"
                    os.environ["UNRELATED_VAR"] = "x"
            self.assertIn("TPU_VISIBLE_CHIPS", str(caught.exception))
            self.assertEqual(os.environ["TPU_VISIBLE_CHIPS"], "0,1,2,3")
            self.assertNotIn("TPU_PROCESS_BOUNDS", os.environ)
            with engine_vllm.frozen_tpu_env():  # no TPU change: fine
                os.environ["UNRELATED_VAR"] = "y"
        finally:
            os.environ.pop("TPU_VISIBLE_CHIPS", None)
            os.environ.pop("UNRELATED_VAR", None)

    def test_preinit_environment_on_a_dict(self):
        env = {"LIBTPU_INIT_ARGS": "--a"}
        got = engine_vllm.preinit_environment(env)
        self.assertEqual(env["TPU_MULTIHOST_BACKEND"], "ray")
        self.assertEqual(got, {"LIBTPU_INIT_ARGS": "--a", "TPU_MULTIHOST_BACKEND": "ray", "TPU_MULTIPROCESS_DP": "0"})
        env["TPU_MULTIHOST_BACKEND"] = "other"
        engine_vllm.preinit_environment(env)
        self.assertEqual(env["TPU_MULTIHOST_BACKEND"], "other")  # setdefault: never overrides

    def test_preinit_shares_the_vllm_xla_cache(self):
        """tpu-inference moves JAX's cache to VLLM_XLA_CACHE_PATH (pod-local by default); only one
        process per engine writes, so the other hosts must read the shared JAX cache dir."""
        env = {"JAX_COMPILATION_CACHE_DIR": "/models/jax_cache/x"}
        engine_vllm.preinit_environment(env)
        self.assertEqual(env["VLLM_XLA_CACHE_PATH"], "/models/jax_cache/x")
        env = {"JAX_COMPILATION_CACHE_DIR": "/models/jax_cache/x", "VLLM_XLA_CACHE_PATH": "/shared/y"}
        engine_vllm.preinit_environment(env)
        self.assertEqual(env["VLLM_XLA_CACHE_PATH"], "/shared/y")
        env = {}
        engine_vllm.preinit_environment(env)
        self.assertNotIn("VLLM_XLA_CACHE_PATH", env)

    def test_build_requires_preinit(self):
        engine = engine_vllm.VllmEngine(EngineSpec("e", "vllm", (0, 1), 8))
        with self.assertRaises(engine_vllm.EngineGuardError):
            engine.build(None)  # tpu_inference.env_override was not imported
        with self.assertRaises(ValueError):
            engine_vllm.VllmEngine(EngineSpec("e", "synthetic", (0, 1), 8))

    def test_state_from_leaves_is_strict_and_keeps_structure(self):
        import jax
        import jax.numpy as jnp
        abstract = {"model": {"a": jax.ShapeDtypeStruct((2, 3), jnp.float32),
                              "b": jax.ShapeDtypeStruct((4,), jnp.float32)}}
        leaves = {"['model']['a']": jnp.ones((2, 3)), "['model']['b']": jnp.zeros((4,))}
        state = engine_vllm._state_from_leaves(abstract, leaves)
        self.assertEqual(jax.tree_util.tree_structure(state), jax.tree_util.tree_structure(abstract))
        self.assertEqual(float(state["model"]["a"].sum()), 6.0)
        with self.assertRaises(ValueError):
            engine_vllm._state_from_leaves(abstract, {"['model']['a']": leaves["['model']['a']"]})
        with self.assertRaises(ValueError):
            engine_vllm._state_from_leaves(abstract, {**leaves, "['model']['b']": jnp.zeros((5,))})

    def test_leaf_specs_report_tp_axis_and_fused_parts(self):
        import jax
        import jax.numpy as jnp
        import numpy as np
        from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
        mesh = Mesh(np.array(jax.devices("cpu")[:1]), ("model",))
        put = lambda shape, spec: jax.device_put(jnp.zeros(shape, jnp.float32), NamedSharding(mesh, spec))
        state = {"model": {"qkv_proj": {"weight": {"value": put((4, 12), P(None, "model"))}},
                           "gate_up_proj": {"weight": {"value": put((4, 8), P(None, "model"))}},
                           "norm": {"scale": {"value": put((4,), P())}}}}
        packing = {("model", "qkv_proj", "weight"): (6, 3, 3), ("model", "gate_up_proj", "weight"): (4, 4)}
        specs = {s.path: s for s in engine_vllm._leaf_specs(state, packing, tp=1)}
        qkv = specs["['model']['qkv_proj']['weight']['value']"]
        self.assertEqual((qkv.fused, qkv.fused_sizes, qkv.tp_axis, qkv.shape), ("qkv", (6, 3, 3), 1, (4, 12)))
        self.assertEqual(specs["['model']['gate_up_proj']['weight']['value']"].fused, "gate_up")
        norm = specs["['model']['norm']['scale']['value']"]
        self.assertEqual((norm.fused, norm.tp_axis), (None, None))
        with self.assertRaises(ValueError):
            engine_vllm._leaf_specs(state, packing, tp=4)  # 6 % 4 != 0


# ---------------------------------------------------------------------- executor


class FakeHost:
    """Stands in for a HostKernel actor: engine_call("step", envelope) -> {"result": ...}."""

    def __init__(self, rank, log):
        self.rank, self.log = rank, log

    def step(self, envelope):
        method = envelope["rpc"]
        self.log.append((self.rank, method, envelope["args"], envelope["kwargs"]))
        if method == "execute_model":
            result = f"output-from-{self.rank}" if self.rank == 0 else None
        elif method == "compile_or_warm_up_model":
            result = types.SimpleNamespace(language_model=1.5 + self.rank, encoder=0.5)
        else:
            result = (method, self.rank)
        return {"result": result, "tokens": [], "seconds": 0.0}


class InlineTransport:
    def submit(self, host, envelope):
        return host.step(envelope)

    def gather(self, refs, timeout):
        return list(refs)

    def health(self, hosts, timeout):
        return [{"rank": h.rank, "failure": None} for h in hosts]


HOST_LOG = []


def fake_resolver(settings):
    return [FakeHost(rank, HOST_LOG) for rank in range(len(settings["hosts"]))], InlineTransport()


def install_stub_vllm():
    class Executor:
        uses_ray = False

        def __init__(self, vllm_config):
            self.vllm_config = vllm_config
            self.parallel_config = vllm_config.parallel_config
            self._init_executor()

    names = ("vllm", "vllm.v1", "vllm.v1.executor", "vllm.v1.executor.abstract")
    saved = {n: sys.modules.get(n) for n in names}
    for name in names:
        sys.modules[name] = types.ModuleType(name)
    sys.modules["vllm.v1.executor.abstract"].Executor = Executor
    return saved


def restore(saved):
    for name, module in saved.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def fake_vllm_config(hosts=2, **extra):
    config = types.SimpleNamespace(
        parallel_config=types.SimpleNamespace(pipeline_parallel_size=1),
        compilation_config=types.SimpleNamespace(compilation_time=0, encoder_compilation_time=0),
        additional_config=vllm_executor.executor_additional_config(
            [{"name": f"h{i}", "namespace": "n"} for i in range(hosts)],
            resolver=f"{__name__}:fake_resolver", **extra))
    return config


class ViewExecutor(unittest.TestCase):
    def setUp(self):
        HOST_LOG.clear()
        self.saved = install_stub_vllm()
        self.addCleanup(restore, self.saved)
        self.cls = vllm_executor.make_executor_class()

    def test_collective_rpc_addresses_every_host_in_rank_order(self):
        executor = self.cls(fake_vllm_config(4))
        self.assertEqual(executor.collective_rpc("get_kv_cache_spec"),
                         [("get_kv_cache_spec", r) for r in range(4)])
        self.assertEqual([row[0] for row in HOST_LOG], [0, 1, 2, 3])

    def test_execute_model_returns_the_driver_worker_output(self):
        executor = self.cls(fake_vllm_config(2))
        self.assertEqual(executor.execute_model("scheduled"), "output-from-0")
        future = executor.execute_model("scheduled", non_block=True)
        self.assertIsInstance(future, Future)
        self.assertEqual(future.result(timeout=5), "output-from-0")
        self.assertEqual(executor.sample_tokens(None), ("sample_tokens", 0))
        self.assertEqual([row[1] for row in HOST_LOG].count("execute_model"), 4)

    def test_initialize_from_config_sends_one_config_per_host_then_warms_up(self):
        config = fake_vllm_config(2)
        executor = self.cls(config)
        executor.initialize_from_config(["cfg0", "cfg1"])
        self.assertEqual([(r, m, a) for r, m, a, _ in HOST_LOG],
                         [(0, "initialize_from_config", ("cfg0",)), (1, "initialize_from_config", ("cfg1",)),
                          (0, "compile_or_warm_up_model", ()), (1, "compile_or_warm_up_model", ())])
        self.assertEqual(config.compilation_config.compilation_time, 2.5)
        with self.assertRaises(ValueError):
            executor.initialize_from_config(["only-one"])

    def test_deferred_warmup_waits_for_the_caller(self):
        config = fake_vllm_config(2, defer_warmup=True)
        executor = self.cls(config)
        executor.initialize_from_config(["a", "b"])
        self.assertEqual({m for _, m, _, _ in HOST_LOG}, {"initialize_from_config"})
        executor.start_warmup()
        executor.wait_for_warmup()
        self.assertIn("compile_or_warm_up_model", {m for _, m, _, _ in HOST_LOG})
        self.assertEqual(config.compilation_config.compilation_time, 2.5)

    def test_shutdown_detaches_without_rpc_and_health_reports_failures(self):
        executor = self.cls(fake_vllm_config(2))
        executor.check_health()
        executor.rpc.transport.health = lambda hosts, timeout: [{"failure": "boom"}]
        with self.assertRaises(RuntimeError):
            executor.check_health()
        executor.shutdown()
        self.assertEqual(HOST_LOG, [])

    def test_pipeline_parallelism_is_refused(self):
        config = fake_vllm_config(2)
        config.parallel_config.pipeline_parallel_size = 2
        with self.assertRaises(ValueError):
            self.cls(config)

    def test_module_attribute_resolves_lazily_and_is_cached(self):
        vllm_executor._EXECUTOR_CLASS = None
        try:
            cls = vllm_executor.VllmViewExecutor
            self.assertIs(cls, vllm_executor.VllmViewExecutor)
            self.assertEqual((cls.__module__, cls.__qualname__), (vllm_executor.__name__, "VllmViewExecutor"))
            from vllm.utils.import_utils import resolve_obj_by_qualname  # noqa: F401  (absent in stub)
        except ImportError:
            pass
        finally:
            vllm_executor._EXECUTOR_CLASS = None

    def test_view_rpc_per_host_args_must_match(self):
        rpc = vllm_executor.ViewRpc([FakeHost(0, HOST_LOG)], InlineTransport())
        with self.assertRaises(ValueError):
            rpc.call("m", per_host_args=[(1,), (2,)])
        with self.assertRaises(ValueError):
            vllm_executor.ViewRpc([], InlineTransport())


class StepEnvelope(unittest.TestCase):
    def test_step_forwards_to_the_worker_and_extracts_tokens(self):
        sys.modules.setdefault("vllm", types.ModuleType("vllm"))
        saved = {n: sys.modules.get(n) for n in ("vllm", "vllm.config")}
        config_module = types.ModuleType("vllm.config")
        import contextlib
        config_module.set_current_vllm_config = lambda config: contextlib.nullcontext()
        sys.modules["vllm.config"] = config_module
        self.addCleanup(restore, saved)

        class Worker:
            vllm_config = object()

            def execute_model(self, so):
                return types.SimpleNamespace(sampled_token_ids=[[3, 4], [5]], so=so)

            def get_kv_cache_spec(self):
                return {"layer": 1}

        engine = engine_vllm.VllmEngine(EngineSpec("e", "vllm", (0, 1), 8))
        with self.assertRaises(RuntimeError):
            engine.step({"rpc": "get_kv_cache_spec"})
        engine.worker = Worker()
        out = engine.step({"rpc": "execute_model", "args": ("so",)})
        self.assertEqual((out["tokens"], out["result"].so), ([[3, 4], [5]], "so"))
        self.assertEqual(engine.step({"rpc": "get_kv_cache_spec"})["tokens"], [])
        self.assertEqual(engine.step({"rpc": lambda worker, k: (worker, k), "args": (7,)})["result"][1], 7)


if __name__ == "__main__":
    unittest.main()
