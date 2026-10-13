"""Host staging of native TPU attention pages, without a second allocator.

The caller owns request fencing, native block reservation and model identity.
Every worker RPC must run with the engine's execution queue drained and step
submission excluded: JAX updates donate the live cache buffers, and the TPU
Ray executor's compiled graph cannot tolerate other worker calls while a step
is in flight. Each paused RPC does only device-side work (gather or scatter)
and hands the device-to-host copy, object-store transfer and host-to-device
upload to a worker background thread, which reports completion straight to
the engine actor. Both engines keep stepping during that host work.
Addressing is per native cache group; null/evicted pages must be omitted, not
renumbered.

Supported arrays are the installed RPA five-dimensional, unquantized attention
layout, with only the packed-head axis partitioned. Native Gemma4 contiguous KV
head replicas are proven byte-identical on the source devices, and only one
canonical copy of every original head crosses the host. Each destination
worker fetches only the canonical heads its own devices hold and repeats them
into its local shards. Other TP-dependent packing changes, MLA, Mamba and
context/data sharding are rejected.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Optional

FORMAT = "tpu-rpa-host-pieces-v2"
DEFAULT_STAGE_BYTES_PER_DEVICE = 64 << 20


def _indices(index: Any, shape: Any) -> list[list[int]]:
    result = []
    for part, size in zip(index, shape):
        if not isinstance(part, slice):
            raise ValueError("KV shards must use slice indices")
        start, stop, step = part.indices(int(size))
        if step != 1:
            raise ValueError("strided KV shards are unsupported")
        result.append([start, stop])
    if len(result) != len(shape):
        raise ValueError("incomplete KV shard index")
    return result


def _native_head_semantics(runner: Any, name: str, spec: Any) -> dict | None:
    """Read original heads from the live Gemma4 attention module, not HF guesses.

    Installed gemma4.py preserves num_kv_heads through projection/norm/RoPE.
    attention_interface.sharded_ragged_paged_attention repeats consecutive K/V
    heads before shard_map(... P(ATTN_DATA, ATTN_HEAD, None)); that resharding
    assembles projection head-dimension shards before the cache is written.
    """
    model = getattr(runner, "model", None)
    backbone = getattr(model, "language_model", None)
    if backbone is None:
        backbone = getattr(getattr(model, "model", None), "language_model", None)
    layers = getattr(backbone, "layers", None)
    prefix, separator, suffix = name.partition(".")
    if layers is None or prefix != "layer" or not separator or not suffix.isdigit():
        return None
    attention = getattr(layers[int(suffix)], "self_attn", None)
    if (type(attention).__module__, type(attention).__name__) != (
        "tpu_inference.models.jax.gemma4", "Gemma4Attention"
    ):
        return None
    from tpu_inference.layers.common.sharding import ShardingAxisName
    from tpu_inference.utils import get_mesh_shape_product

    if any(attention.mesh.shape.get(axis, 1) != 1 for axis in ("dcp", "pcp")):
        return None
    original = int(attention.num_kv_heads)
    head_dim = int(attention.head_dim_original)
    effective = int(spec.num_kv_heads)
    tp_size = int(get_mesh_shape_product(attention.mesh, ShardingAxisName.ATTN_HEAD))
    if original <= 0 or effective != max(original, tp_size) or effective % original:
        raise ValueError(f"native Gemma4 KV replication disagrees with allocated spec: {name}")
    if int(attention.head_dim) != int(spec.head_size):
        raise ValueError(f"native Gemma4 head dimension disagrees with allocated spec: {name}")
    # The hd64 kernel packs K/V in the last axis instead; leave that unchanged.
    if head_dim == 64:
        return None
    return {"order": "gemma4-rpa-consecutive-repeat", "original_num_kv_heads": original,
            "effective_num_kv_heads": effective, "original_head_dim": head_dim,
            "attention_tp_size": tp_size}


def _logical_groups(layout: dict) -> list[dict]:
    """Compare semantic groups while retaining unknown layouts verbatim."""
    groups = []
    for group in layout["groups"]:
        layers = []
        for row in group["layers"]:
            row = dict(row)
            semantics = row.get("head_semantics")
            if semantics is not None:
                row["num_kv_heads"] = semantics["original_num_kv_heads"]
                row["head_semantics"] = {key: value for key, value in semantics.items()
                                         if key not in ("effective_num_kv_heads", "attention_tp_size")}
            layers.append(row)
        groups.append({**group, "layers": layers})
    return groups


def cache_layout(runner: Any) -> dict[str, Any]:
    """Describe physical arrays, aliases, cache groups and actual device slices."""
    import jax

    caches = runner.kv_caches
    if not caches:
        raise ValueError("native KV cache is not allocated")
    mapping = runner.layer_name_to_kvcache_index
    groups = []
    memberships: list[list[int]] = [[] for _ in caches]
    for group_id, group in enumerate(runner.kv_cache_config.kv_cache_groups):
        specs = getattr(group.kv_cache_spec, "kv_cache_specs", None)
        rows = []
        for name in group.layer_names:
            spec = specs[name] if specs is not None else group.kv_cache_spec
            kind = type(spec).__name__
            if kind not in ("FullAttentionSpec", "SlidingWindowSpec"):
                raise ValueError(f"unsupported native KV spec {kind}: {name}")
            index = int(mapping[name])
            if group_id not in memberships[index]:
                memberships[index].append(group_id)
            rows.append({"layer_name": name, "array_index": index,
                         "spec_type": kind, "block_size": int(spec.block_size),
                         "storage_block_size": int(spec.storage_block_size),
                         "head_size": int(spec.head_size),
                         "num_kv_heads": int(spec.num_kv_heads),
                         "head_semantics": _native_head_semantics(runner, name, spec),
                         "sliding_window": getattr(spec, "sliding_window", None)})
        groups.append({"group_id": group_id, "layers": rows})
    arrays = []
    for index, cache in enumerate(caches):
        if not isinstance(cache, jax.Array) or cache.ndim != 5:
            raise ValueError(f"cache {index} is not a five-dimensional attention array")
        shape = list(map(int, cache.shape))
        if str(cache.dtype) not in ("bfloat16", "float16", "float32"):
            raise ValueError(f"unsupported KV dtype {cache.dtype}; quantization scales are not transported")
        if not memberships[index]:
            raise ValueError(f"physical cache {index} has no native cache-group mapping")
        for group_id in memberships[index]:
            for row in groups[group_id]["layers"]:
                if row["array_index"] == index and not (
                    row["block_size"] == row["storage_block_size"] == shape[1]
                ):
                    raise ValueError("logical/storage/physical page sizes differ; context packing unsupported")
        members = [row for gid in memberships[index] for row in groups[gid]["layers"]
                   if row["array_index"] == index]
        semantics = members[0]["head_semantics"]
        if any(row["head_semantics"] != semantics for row in members):
            semantics = None
        if semantics is not None and shape[2] * shape[3] != 2 * semantics["effective_num_kv_heads"]:
            raise ValueError(f"native repeated heads disagree with physical packing: array_index={index}")
        shards = []
        for device, raw_index in cache.sharding.devices_indices_map(cache.shape).items():
            bounds = _indices(raw_index, shape)
            if any(bounds[axis] != [0, shape[axis]] for axis in (0, 1, 3, 4)):
                raise ValueError(
                    "only packed-head partitioning is supported; page/context/packing sharding found: "
                    f"array_index={index}, shape={shape}, dtype={cache.dtype}, "
                    f"worker_process_index={jax.process_index()}, "
                    f"device_id={device.id}, device_process_index={device.process_index}, "
                    f"shard_bounds={bounds}, sharding={cache.sharding}"
                )
            shards.append({"device_id": int(device.id),
                           "process_index": int(device.process_index), "index": bounds})
        arrays.append({"array_index": index, "group_ids": memberships[index],
                       "layer_names": sorted(name for name, value in mapping.items() if value == index),
                       "shape": shape, "dtype": str(cache.dtype), "head_semantics": semantics,
                       "bytes_per_block": int(cache.nbytes // shape[0]),
                       "device_layout": str(getattr(cache, "format", getattr(cache, "layout", None))),
                       "sharding": str(cache.sharding), "shards": shards})
    return {"format": FORMAT, "ready": True, "process_index": int(jax.process_index()),
            "process_count": int(jax.process_count()), "groups": groups, "arrays": arrays}


def _validate_addressing(layout: dict, addressing: dict) -> dict[int, dict]:
    computed = addressing["num_computed_tokens"]
    if not isinstance(computed, int) or computed <= 0:
        raise ValueError("a nonempty computed KV prefix is required")
    groups = addressing["groups"]
    if [g["group_id"] for g in groups] != list(range(len(layout["groups"]))):
        raise ValueError("addressing must contain every native group in order")
    for group, descriptor in zip(groups, layout["groups"]):
        ids, logical, ranges = (group[key] for key in
                                ("block_ids", "logical_block_indices", "valid_token_ranges"))
        if not (len(ids) == len(logical) == len(ranges)):
            raise ValueError("page IDs, logical positions and valid ranges must align")
        if len(set(ids)) != len(ids) or logical != sorted(set(logical)):
            raise ValueError("duplicate physical blocks or unordered logical blocks")
        sizes = {row["block_size"] for row in descriptor["layers"]}
        if len(sizes) != 1:
            raise ValueError("mixed block sizes within a group are unsupported")
        block_size = sizes.pop()
        capacities = [a["shape"][0] for a in layout["arrays"] if group["group_id"] in a["group_ids"]]
        for block, position, valid in zip(ids, logical, ranges):
            if not isinstance(block, int) or not 0 < block < min(capacities):
                raise ValueError("null, negative or out-of-bounds physical KV block")
            if not isinstance(position, int) or position < 0:
                raise ValueError("invalid logical KV block position")
            start, stop = valid
            if not (position * block_size <= start < stop <= min((position + 1) * block_size, computed)):
                raise ValueError("valid KV token range lies outside its page/computed prefix")
    # Native hybrid groups may share an array, but may not own the same page.
    for array in layout["arrays"]:
        blocks = [block for gid in array["group_ids"] for block in groups[gid]["block_ids"]]
        if len(blocks) != len(set(blocks)):
            raise ValueError("cache groups overlap in a shared physical array")
    return {group["group_id"]: group for group in groups}


@lru_cache(maxsize=32)
def _gather(sharding: Any) -> Any:
    import jax
    import jax.numpy as jnp

    def gather(cache: Any, indices: Any, starts: Any, stops: Any) -> Any:
        pages = cache[indices]
        positions = jnp.arange(cache.shape[1])[None, :]
        valid = (positions >= starts[:, None]) & (positions < stops[:, None])
        return jnp.where(valid[:, :, None, None, None], pages, 0)

    return jax.jit(gather, out_shardings=sharding)


@lru_cache(maxsize=32)
def _scatter(sharding: Any) -> Any:
    import jax

    return jax.jit(lambda cache, indices, pages: cache.at[indices].set(pages),
                   donate_argnums=(0,), out_shardings=sharding)


def _put(value: Any) -> Any:
    import ray

    return ray.put(value)


def _fetch(notifier: Any, transaction_id: str, keys: list) -> list:
    """Resolve pieces through a Ray actor call, never refs pickled into a worker RPC.

    Worker RPCs travel over vLLM's shared-memory queue with plain pickle, which
    drops ObjectRef ownership; refs returned by a Ray call are properly borrowed.
    """
    import ray

    return ray.get(ray.get(notifier.omp_kv_refs.remote(transaction_id, keys)))


def split_refs(payload: dict) -> tuple[dict, dict]:
    """Replace every piece ref with an integer key; the caller keeps the refs."""
    refs, arrays = {}, []
    for row in payload["arrays"]:
        pieces = []
        for piece in row["pieces"]:
            key = len(refs)
            refs[key] = piece["ref"]
            pieces.append({**{k: v for k, v in piece.items() if k != "ref"}, "key": key})
        arrays.append({**row, "pieces": pieces})
    return {**payload, "arrays": arrays}, refs


def _geometry(descriptor: dict) -> dict:
    """Canonical units of one physical array: original heads, or raw axis-2 rows."""
    shape = descriptor["shape"]
    semantics = descriptor.get("head_semantics")
    if semantics is None:
        return {"mode": "raw", "units": shape[2], "repeat": 1}
    effective = semantics["effective_num_kv_heads"]
    original = semantics["original_num_kv_heads"]
    return {"mode": "heads", "units": original, "repeat": effective // original}


def _heads(bounds: list[int], width: int) -> tuple[int, int]:
    """Effective heads held by an axis-2 interval of packed (head, K/V) rows."""
    start, stop = bounds[0] * width, bounds[1] * width
    if start % 2 or stop % 2:
        raise ValueError("KV shard boundary splits one head's packed K/V pair")
    return start // 2, stop // 2


def _per_device_bytes(sharding: Any, shape: tuple, itemsize: int) -> int:
    size = itemsize
    for extent in sharding.shard_shape(shape):
        size *= int(extent)
    return size


@lru_cache(maxsize=32)
def _replica_check(effective: int, repeat: int, dtype: str) -> Any:
    """Bitwise proof, on device, that every native head replica is identical."""
    import jax
    import jax.numpy as jnp

    bits = {2: jnp.uint16, 4: jnp.uint32}[jnp.dtype(dtype).itemsize]

    def check(pages: Any) -> Any:
        raw = jax.lax.bitcast_convert_type(pages, bits)
        heads = raw.reshape(*raw.shape[:2], effective, 2, raw.shape[-1])
        base = heads[:, :, 0::repeat]
        same = jnp.bool_(True)
        for offset in range(1, repeat):
            same = same & jnp.all(heads[:, :, offset::repeat] == base)
        return same

    return jax.jit(check)


def _selected_rows(layout: dict, groups: dict) -> list[tuple[dict, int]]:
    return [(descriptor, gid) for descriptor in layout["arrays"]
            for gid in descriptor["group_ids"] if groups[gid]["block_ids"]]


def _executor(staging: dict) -> Any:
    from concurrent.futures import ThreadPoolExecutor

    executor = staging.get("_executor")
    if executor is None:
        executor = staging["_executor"] = ThreadPoolExecutor(1, thread_name_prefix="omp-kv")
    return executor


def _notify(notifier: Any, transaction_id: str, message: dict) -> None:
    """Tell the engine actor a background chunk finished; never a worker RPC."""
    notifier.omp_kv_notify.remote(transaction_id, message)


def _background(staging: dict, notifier: Any, transaction_id: str, process_index: int,
                kind: str, chunk: int, job: Any) -> None:
    """Every worker numbers chunks identically, since all receive the same RPC sequence."""
    def run() -> None:
        base = {"kind": kind, "process_index": process_index, "chunk": chunk}
        try:
            message = {**base, **job()}
        except BaseException as exc:  # reported to the actor, which fails closed
            message = {**base, "error": f"{type(exc).__name__}: {exc}"}
        try:
            _notify(notifier, transaction_id, message)
        except BaseException as exc:  # the actor times out; keep the cause visible
            import sys
            print(f"omp-kv notify failed for {transaction_id}: {exc!r}", file=sys.stderr, flush=True)

    _executor(staging).submit(run)


def export_gather(runner: Any, staging: dict, transaction_id: str, addressing: dict,
                  start: int, budget_bytes: int, notifier: Any) -> dict:
    """Device-only step under the engine pause, then background host staging.

    Gathers live pages of rows [start, next) into new device buffers, zeroes
    tokens outside the valid range and proves head replicas equal on device.
    The device-to-host copy and object-store publication run on a worker
    background thread while the engine keeps stepping; completion is reported
    to ``notifier`` (the engine actor), never polled through a worker RPC.
    """
    import time
    import numpy as np

    started = time.perf_counter()
    layout = cache_layout(runner)
    groups = _validate_addressing(layout, addressing)
    rows = _selected_rows(layout, groups)
    key = ("export", transaction_id)
    if start == 0:
        if key in staging:
            raise RuntimeError("KV export already started for this transaction")
        staging[key] = {"addressing": addressing, "cursor": 0, "chunks": 0}
    record = staging.get(key)
    if record is None or record["addressing"] != addressing or record["cursor"] != start:
        raise ValueError("export chunk disagrees with this transaction's addressing/cursor")
    cursor, used = start, 0
    gathered = []
    while cursor < len(rows):
        descriptor, gid = rows[cursor]
        cache = runner.kv_caches[descriptor["array_index"]]
        group = groups[gid]
        shape = (len(group["block_ids"]), *cache.shape[1:])
        size = _per_device_bytes(cache.sharding, shape, cache.dtype.itemsize)
        if gathered and used + size > budget_bytes:
            break
        base = np.asarray(group["logical_block_indices"], dtype=np.int32) * cache.shape[1]
        valid = np.asarray(group["valid_token_ranges"], dtype=np.int32)
        selected = _gather(cache.sharding)(cache, np.asarray(group["block_ids"], dtype=np.int32),
                                           valid[:, 0] - base, valid[:, 1] - base)
        geometry = _geometry(descriptor)
        same = None
        if geometry["mode"] == "heads" and geometry["repeat"] > 1:
            semantics = descriptor["head_semantics"]
            same = _replica_check(semantics["effective_num_kv_heads"], geometry["repeat"],
                                  str(cache.dtype))(selected)
        gathered.append({"descriptor": descriptor, "group_id": gid, "selected": selected,
                         "replicas_equal": same})
        used += size
        cursor += 1
    for row in gathered:
        row["selected"].block_until_ready()
        if row["replicas_equal"] is not None:
            row["replicas_equal"].block_until_ready()
    record["cursor"] = cursor
    chunk = record["chunks"]
    record["chunks"] += 1
    if cursor >= len(rows):
        staging.pop(key, None)
    _background(staging, notifier, transaction_id, layout["process_index"], "export", chunk,
                lambda: _stage_export(gathered))
    return {"process_index": layout["process_index"], "next": cursor, "total": len(rows),
            "rows_gathered": len(gathered), "device_bytes": used,
            "gather_s": time.perf_counter() - started,
            "layout": layout if start == 0 else None}


def _stage_export(rows: list) -> dict:
    """Background: copy one canonical replica of the gathered chunk to the host."""
    import time
    import jax
    import numpy as np

    started = time.perf_counter()
    wanted = []  # (row, units, pick, shard); units None marks a pure-replica twin
    twins = []   # (position of twin in wanted, position of first copy in wanted)
    for row in rows:
        descriptor = row["descriptor"]
        if row["replicas_equal"] is not None and not bool(row["replicas_equal"]):
            raise ValueError(
                f"native repeated KV heads disagree: array_index={descriptor['array_index']}")
        selected = row["selected"]
        geometry = _geometry(descriptor)
        seen: dict[tuple[int, int], int] = {}
        for shard in selected.addressable_shards:
            bounds = _indices(shard.index, selected.shape)
            if any(bounds[a] != [0, selected.shape[a]] for a in (0, 1, 3, 4)):
                raise ValueError("export contains unsupported shard partitioning")
            interval = tuple(bounds[2])
            if interval in seen:
                # Pure replication across devices: copy both and compare bytes.
                twins.append((len(wanted), seen[interval]))
                wanted.append((row, None, None, shard))
                continue
            if geometry["mode"] == "heads":
                first, stop = _heads(bounds[2], selected.shape[3])
                repeat = geometry["repeat"]
                offset = (-first) % repeat
                if first + offset >= stop:
                    continue
                units = [(first + offset) // repeat, (stop - 1) // repeat + 1]
                pick = (offset, repeat, stop - first)
            else:
                units, pick = list(bounds[2]), None
            seen[interval] = len(wanted)
            wanted.append((row, units, pick, shard))
    host = [np.asarray(data) for data in jax.device_get([item[3].data for item in wanted])]
    copied_s = time.perf_counter() - started
    rows.clear()  # release the gathered device buffers
    for twin, original in twins:
        if not np.array_equal(host[twin].view(np.uint8), host[original].view(np.uint8)):
            raise ValueError("replicated KV shards disagree")
    pieces, total = [], 0
    for (row, units, pick, _), data in zip(wanted, host):
        if units is None:
            continue
        if pick is not None:
            offset, repeat, count = pick
            data = data.reshape(*data.shape[:2], count, 2, data.shape[-1])[:, :, offset::repeat]
        piece = np.ascontiguousarray(data)
        if piece.shape[2] != units[1] - units[0]:
            raise ValueError("canonical KV piece disagrees with its unit range")
        total += piece.nbytes
        pieces.append({"array_index": row["descriptor"]["array_index"],
                       "group_id": row["group_id"], "units": units,
                       "shape": list(piece.shape), "dtype": str(piece.dtype),
                       "nbytes": int(piece.nbytes), "ref": _put(piece)})
    return {"pieces": pieces, "bytes": int(total), "d2h_s": copied_s,
            "stage_s": time.perf_counter() - started}


def discard(staging: dict, transaction_id: str) -> None:
    staging.pop(("export", transaction_id), None)
    staging.pop(("import", transaction_id), None)


def assemble_manifest(layouts: list[dict], addressing: dict, stages: list[dict]) -> dict:
    """Assemble piece coverage by canonical unit, never worker/process ordering."""
    if not layouts:
        raise ValueError("missing worker exports")
    layout = layouts[0]
    processes = [item["process_index"] for item in layouts]
    if sorted(processes) != list(range(layout["process_count"])):
        raise ValueError("expected exactly one export per native worker process")
    for item in layouts:
        if item["groups"] != layout["groups"] or item["arrays"] != layout["arrays"]:
            raise ValueError("workers disagree on native cache groups/physical structure")
    groups = _validate_addressing(layout, addressing)
    rows = {(d["array_index"], gid): {"array_index": d["array_index"], "group_id": gid,
                                      "blocks": len(groups[gid]["block_ids"]), "pieces": []}
            for d, gid in _selected_rows(layout, groups)}
    for stage in stages:
        for piece in stage["pieces"]:
            key = (piece["array_index"], piece["group_id"])
            if key not in rows:
                raise ValueError("export contains an unexpected KV array")
            if piece["shape"][0] != rows[key]["blocks"]:
                raise ValueError("export piece page count disagrees with addressing")
            rows[key]["pieces"].append(piece)
    total = 0
    for (index, _), row in rows.items():
        units = _geometry(layout["arrays"][index])["units"]
        cursor = 0
        row["pieces"].sort(key=lambda piece: piece["units"])
        for piece in row["pieces"]:
            start, stop = piece["units"]
            if start != cursor or stop <= start:
                raise ValueError("KV piece coverage has a gap or overlap")
            cursor = stop
            total += piece["nbytes"]
        if cursor != units:
            raise ValueError("KV pieces do not cover every canonical unit")
    return {"format": FORMAT, "layout": layout, "addressing": addressing,
            "arrays": list(rows.values()), "bytes": int(total)}


def _validate_import(layout: dict, groups: dict, addressing: dict, payload: dict) -> None:
    if payload["format"] != FORMAT or _logical_groups(payload["layout"]) != _logical_groups(layout):
        raise ValueError("source/destination native cache groups differ")
    source_groups = _validate_addressing(payload["layout"], payload["addressing"])
    if payload["addressing"]["num_computed_tokens"] != addressing["num_computed_tokens"]:
        raise ValueError("source/destination computed prefix differs")
    for gid, group in groups.items():
        if any(group[key] != source_groups[gid][key] for key in
               ("logical_block_indices", "valid_token_ranges")):
            raise ValueError("source/destination logical KV coverage differs")
    source_arrays = payload["layout"]["arrays"]
    if len(source_arrays) != len(layout["arrays"]):
        raise ValueError("source/destination physical array count differs")
    for source, destination in zip(source_arrays, layout["arrays"]):
        if any(source[key] != destination[key] for key in
               ("array_index", "group_ids", "layer_names", "dtype")):
            raise ValueError("source/destination physical array identity differs")
        source_semantics = source.get("head_semantics")
        destination_semantics = destination.get("head_semantics")
        if source_semantics is not None and destination_semantics is not None:
            if any(source_semantics[key] != destination_semantics[key] for key in
                   ("order", "original_num_kv_heads", "original_head_dim")):
                raise ValueError("source/destination native head semantics differ")
            if any(source["shape"][axis] != destination["shape"][axis] for axis in (1, 3, 4)):
                raise ValueError("native head repetition cannot change page/packing/head dimensions")
        elif (source_semantics is None) != (destination_semantics is None) or \
                source["shape"][1:] != destination["shape"][1:]:
            raise ValueError("source/destination physical layout differs; no proven native head mapping")
    expected = {(d["array_index"], gid) for d, gid in _selected_rows(layout, groups)}
    received = [(row["array_index"], row["group_id"]) for row in payload["arrays"]]
    if len(received) != len(expected) or set(received) != expected:
        raise ValueError("missing/duplicate/unexpected KV payload array")


def import_advance(runner: Any, staging: dict, transaction_id: str, *, commit: bool,
                   start: int, budget_bytes: int, notifier: Any,
                   addressing: Optional[dict] = None, payload: Optional[dict] = None) -> dict:
    """One paused step of a pipelined import; the caller holds the engine drained.

    If ``commit``, scatter the oldest fully staged chunk into the reserved pages.
    Then, while fewer than two chunks are staging, plan the next chunk and hand
    its fetch, head repeat and host-to-device upload to the worker background
    thread, which reports completion to ``notifier``. The live cache is only
    read or written here, never in the background.

    If scatter execution itself fails, the caller must fail/tear down the
    destination; it must not schedule from a partial import.
    """
    import time
    from collections import deque

    started = time.perf_counter()
    layout = cache_layout(runner)
    key = ("import", transaction_id)
    if payload is not None:
        if start != 0 or key in staging:
            raise RuntimeError("KV import already started for this transaction")
        groups = _validate_addressing(layout, addressing)
        _validate_import(layout, groups, addressing, payload)
        if any("ref" in piece for row in payload["arrays"] for piece in row["pieces"]):
            raise ValueError("strip ObjectRefs with split_refs before a worker RPC")
        staging[key] = {"payload": payload, "groups": groups, "cursor": 0,
                        "pending": 0, "ready": deque(), "chunks": 0, "notifier": notifier,
                        "transaction_id": transaction_id}
    record = staging.get(key)
    if record is None or record["cursor"] != start:
        raise ValueError("import chunk disagrees with this transaction's cursor")
    committed = None
    if commit:
        if record.get("error"):
            raise RuntimeError(f"background KV staging failed: {record['error']}")
        if not record["ready"]:
            raise RuntimeError("no staged KV import chunk is ready to commit")
        uploads = record["ready"].popleft()
        record["pending"] -= 1
        scatter = time.perf_counter()
        written = []
        for row in uploads:
            index = row["array_index"]
            cache = runner.kv_caches[index]
            updated = _scatter(cache.sharding)(cache, row["block_ids"], row["updates"])
            runner.kv_caches[index] = updated
            written.append(updated)
        for updated in written:
            updated.block_until_ready()
        committed = {"arrays_written": len(uploads), "scatter_s": time.perf_counter() - scatter}
    arrays = record["payload"]["arrays"]
    cursor = start
    submitted = False
    if cursor < len(arrays) and record["pending"] < 2:
        used, plans = 0, []
        while cursor < len(arrays):
            row = arrays[cursor]
            cache = runner.kv_caches[row["array_index"]]
            shape = (len(record["groups"][row["group_id"]]["block_ids"]),
                     *map(int, cache.shape[1:]))
            size = _per_device_bytes(cache.sharding, shape, cache.dtype.itemsize)
            if plans and used + size > budget_bytes:
                break
            plans.append({"row": row, "shape": shape, "sharding": cache.sharding,
                          "dtype": str(cache.dtype),
                          "destination": layout["arrays"][row["array_index"]]})
            used += size
            cursor += 1
        record["cursor"] = cursor
        record["pending"] += 1
        submitted = True
        chunk = record["chunks"]
        record["chunks"] += 1
        _background(staging, notifier, transaction_id, layout["process_index"], "import", chunk,
                    lambda: _stage_import(record, plans))
    if cursor >= len(arrays) and record["pending"] == 0 and not record["ready"]:
        staging.pop(key, None)
    return {"process_index": layout["process_index"], "next": cursor, "total": len(arrays),
            "submitted": submitted, "committed": committed,
            "advance_s": time.perf_counter() - started}


def _stage_import(record: dict, plans: list) -> dict:
    """Background: fetch only locally held canonical pieces and upload local shards."""
    import time
    import jax
    import numpy as np

    started = time.perf_counter()
    try:
        payload, groups = record["payload"], record["groups"]
        record["progress"] = "planning"
        needs, fetch = [], []
        for plan in plans:
            row, shape = plan["row"], plan["shape"]
            geometry = _geometry(plan["destination"])
            local = {}
            for raw in plan["sharding"].addressable_devices_indices_map(shape).values():
                bounds = tuple(_indices(raw, shape)[2])
                if geometry["mode"] == "heads":
                    first, stop = _heads(list(bounds), shape[3])
                    local[bounds] = [head // geometry["repeat"] for head in range(first, stop)]
                else:
                    local[bounds] = list(range(*bounds))
            needed = {unit for units in local.values() for unit in units}
            chosen = [piece for piece in row["pieces"]
                      if any(piece["units"][0] <= unit < piece["units"][1] for unit in needed)]
            for piece in chosen:
                if piece["dtype"] != plan["dtype"] or piece["shape"][0] != shape[0]:
                    raise ValueError("KV payload shape/dtype differs from declared native source")
            needs.append((plan, local, chosen))
            fetch.extend(piece["key"] for piece in chosen)
        record["progress"] = f"fetching {len(fetch)} pieces"
        fetched = iter(_fetch(record["notifier"], record["transaction_id"], fetch) if fetch else [])
        fetch_s = time.perf_counter() - started
        record["progress"] = "uploading"
        fetched_bytes, conversions, uploads = 0, [], []
        for plan, local, chosen in needs:
            row, shape = plan["row"], plan["shape"]
            index, gid = row["array_index"], row["group_id"]
            source = payload["layout"]["arrays"][index]
            destination = plan["destination"]
            canonical = {}
            for piece in chosen:
                data = np.asarray(next(fetched))
                fetched_bytes += data.nbytes
                for offset, unit in enumerate(range(*piece["units"])):
                    canonical[unit] = data[:, :, offset]
            slices = {}
            for bounds, units in local.items():
                parts = np.stack([canonical[unit] for unit in units], axis=2)
                slices[bounds] = parts.reshape(shape[0], shape[1], bounds[1] - bounds[0], *shape[3:])
            if source["shape"][1:] != destination["shape"][1:]:
                semantics = destination["head_semantics"]
                conversions.append({"array_index": index, "group_id": gid,
                                    "original_heads": semantics["original_num_kv_heads"],
                                    "source_effective_heads": source["head_semantics"]["effective_num_kv_heads"],
                                    "destination_effective_heads": semantics["effective_num_kv_heads"],
                                    "source_replicas_byte_equal": True})
            updates = jax.make_array_from_callback(
                shape, plan["sharding"],
                lambda raw, slices=slices, shape=shape: slices[tuple(_indices(raw, shape)[2])])
            uploads.append({"array_index": index, "group_id": gid, "updates": updates,
                            "block_ids": np.asarray(groups[gid]["block_ids"], dtype=np.int32)})
        record["progress"] = "awaiting device upload"
        total = 0
        for upload in uploads:
            upload["updates"].block_until_ready()
            total += sum(int(shard.data.nbytes) for shard in upload["updates"].addressable_shards)
    except BaseException as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    record["ready"].append(uploads)
    return {"arrays_staged": len(uploads), "fetched_bytes": int(fetched_bytes),
            "bytes": int(total), "head_conversions": conversions, "fetch_s": fetch_s,
            "stage_s": time.perf_counter() - started}
