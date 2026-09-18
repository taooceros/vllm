"""Host staging of native TPU attention pages, without a second allocator.

The caller owns request fencing, native block reservation and model identity.
Every worker RPC must run with the engine's execution queue drained and step
submission excluded: JAX updates donate the live cache buffers. Addressing is
per native cache group; null/evicted pages must be omitted, not renumbered.

Supported arrays are the installed RPA five-dimensional, unquantized attention
layout, with only the packed-head axis partitioned. Native Gemma4 contiguous KV
head replication can be converted after validating every source replica. Other
TP-dependent packing changes, MLA, Mamba and context/data sharding are rejected.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

FORMAT = "tpu-rpa-host-pages-v1"


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


def _canonical_pages(data: Any, descriptor: dict) -> Any:
    """Unpack ordinary RPA heads and prove every native repeat is byte-identical."""
    import numpy as np

    semantics = descriptor.get("head_semantics")
    if semantics is None:
        return data
    original = semantics["original_num_kv_heads"]
    effective = semantics["effective_num_kv_heads"]
    repeats = effective // original
    heads = data.reshape(*data.shape[:2], effective, 2, data.shape[-1])
    canonical = heads[:, :, ::repeats, :, :]
    for offset in range(1, repeats):
        replica = heads[:, :, offset::repeats, :, :]
        if not np.array_equal(canonical.view(np.uint8), replica.view(np.uint8)):
            raise ValueError(
                f"native repeated KV heads disagree: array_index={descriptor["array_index"]}, "
                f"original_heads={original}, effective_heads={effective}, replica_offset={offset}"
            )
    return canonical


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


def export_blocks(runner: Any, addressing: dict) -> dict:
    """Gather only live pages; invalid tokens are zeroed before device-to-host."""
    import time
    import jax
    import numpy as np

    started = time.perf_counter()
    layout = cache_layout(runner)
    groups = _validate_addressing(layout, addressing)
    rows = []
    total = 0
    for descriptor, cache in zip(layout["arrays"], runner.kv_caches):
        for gid in descriptor["group_ids"]:
            group = groups[gid]
            if not group["block_ids"]:
                continue
            base = np.asarray(group["logical_block_indices"], dtype=np.int32) * cache.shape[1]
            valid = np.asarray(group["valid_token_ranges"], dtype=np.int32)
            selected = _gather(cache.sharding)(cache, np.asarray(group["block_ids"], dtype=np.int32),
                                                valid[:, 0] - base, valid[:, 1] - base)
            chunks = []
            for shard in selected.addressable_shards:
                # Replicated shards are validated/deduplicated during assembly.
                data = np.asarray(jax.device_get(shard.data))
                total += data.nbytes
                chunks.append({"index": _indices(shard.index, selected.shape), "data": data,
                               "device_id": int(shard.device.id)})
            rows.append({"array_index": descriptor["array_index"], "group_id": gid,
                         "shape": list(selected.shape), "dtype": str(selected.dtype), "shards": chunks})
    return {"format": FORMAT, "layout": layout, "addressing": addressing, "arrays": rows,
            "bytes": int(total), "gather_s": time.perf_counter() - started}


def assemble_exports(exports: list[dict]) -> dict:
    """Assemble by physical shard coordinates, never worker/process ordering."""
    import numpy as np

    if not exports or any(item["format"] != FORMAT for item in exports):
        raise ValueError("missing or incompatible worker exports")
    first = exports[0]
    layout = first["layout"]
    processes = [item["layout"]["process_index"] for item in exports]
    if sorted(processes) != list(range(layout["process_count"])):
        raise ValueError("expected exactly one export per native worker process")
    for item in exports:
        if item["addressing"] != first["addressing"] or item["layout"]["groups"] != layout["groups"]:
            raise ValueError("workers disagree on native addressing/cache groups")
        if item["layout"]["arrays"] != layout["arrays"]:
            raise ValueError("workers disagree on physical cache structure")
    keys = [(row["array_index"], row["group_id"]) for row in first["arrays"]]
    rows = []
    for item in exports:
        if [(r["array_index"], r["group_id"]) for r in item["arrays"]] != keys:
            raise ValueError("workers disagree on exported physical arrays")
    for offset, key in enumerate(keys):
        descriptor = first["arrays"][offset]
        shape = descriptor["shape"]
        chunks: dict[tuple[int, int], Any] = {}
        for item in exports:
            row = item["arrays"][offset]
            if row["shape"] != shape or row["dtype"] != descriptor["dtype"]:
                raise ValueError("workers disagree on exported shape/dtype")
            for shard in row["shards"]:
                bounds, data = shard["index"], np.asarray(shard["data"])
                if any(bounds[a] != [0, shape[a]] for a in (0, 1, 3, 4)):
                    raise ValueError("export contains unsupported shard partitioning")
                if list(data.shape) != [b - a for a, b in bounds] or str(data.dtype) != row["dtype"]:
                    raise ValueError("shard data disagrees with its physical descriptor")
                interval = tuple(bounds[2])
                if interval in chunks and not np.array_equal(
                    chunks[interval].view(np.uint8), data.view(np.uint8)
                ):
                    raise ValueError("replicated KV shards disagree")
                chunks[interval] = data
        cursor = 0
        for start, stop in sorted(chunks):
            if start != cursor or stop <= start:
                raise ValueError("KV shard coverage has a gap or overlap")
            cursor = stop
        if cursor != shape[2]:
            raise ValueError("KV shards do not cover the physical packed-head axis")
        data = next(iter(chunks.values()))
        if len(chunks) > 1:
            data = np.empty(shape, dtype=data.dtype)
            for (start, stop), chunk in chunks.items():
                data[:, :, start:stop, :, :] = chunk
        _canonical_pages(data, layout["arrays"][key[0]])
        rows.append({"array_index": key[0], "group_id": key[1], "data": data})
    return {"format": FORMAT, "layout": layout, "addressing": first["addressing"],
            "arrays": rows, "bytes": sum(row["data"].nbytes for row in rows)}


def import_blocks(runner: Any, addressing: dict, payload: dict) -> dict:
    """Populate native-reserved pages, then wait for every device write to finish.

    Validation precedes donation. If execution itself fails, the caller must
    fail/tear down the destination; it must not schedule from a partial import.
    """
    import time
    import jax
    import numpy as np

    started = time.perf_counter()
    layout = cache_layout(runner)
    groups = _validate_addressing(layout, addressing)
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
        elif source["shape"][1:] != destination["shape"][1:]:
            raise ValueError("source/destination physical layout differs; no proven native head mapping")
    expected = {(a["array_index"], gid) for a in layout["arrays"] for gid in a["group_ids"]
                if groups[gid]["block_ids"]}
    received = [(row["array_index"], row["group_id"]) for row in payload["arrays"]]
    if len(received) != len(expected) or set(received) != expected:
        raise ValueError("missing/duplicate/unexpected KV payload array")
    prepared = []
    conversions = []
    for row in payload["arrays"]:
        index, gid = row["array_index"], row["group_id"]
        cache = runner.kv_caches[index]
        source = source_arrays[index]
        destination = layout["arrays"][index]
        data = np.asarray(row["data"])
        if data.shape != (len(groups[gid]["block_ids"]), *source["shape"][1:]) or data.dtype != cache.dtype:
            raise ValueError("KV payload shape/dtype differs from declared native source")
        canonical = _canonical_pages(data, source)
        if source["shape"][1:] != destination["shape"][1:]:
            semantics = destination["head_semantics"]
            factor = semantics["effective_num_kv_heads"] // semantics["original_num_kv_heads"]
            data = np.repeat(canonical, factor, axis=2).reshape(
                len(groups[gid]["block_ids"]), *cache.shape[1:])
            conversions.append({"array_index": index, "group_id": gid,
                                "original_heads": semantics["original_num_kv_heads"],
                                "source_effective_heads": source["head_semantics"]["effective_num_kv_heads"],
                                "destination_effective_heads": semantics["effective_num_kv_heads"],
                                "source_replicas_byte_equal": True})
        prepared.append((index, gid, data))
    total = 0
    for index, gid, data in prepared:
        cache = runner.kv_caches[index]
        updates = jax.make_array_from_callback(data.shape, cache.sharding, lambda bounds: data[bounds])
        indices = np.asarray(groups[gid]["block_ids"], dtype=np.int32)
        updated = _scatter(cache.sharding)(cache, indices, updates)
        runner.kv_caches[index] = updated
        updated.block_until_ready()
        total += sum(int(shard.data.nbytes) for shard in updates.addressable_shards)
    return {"process_index": layout["process_index"], "arrays_written": len(prepared),
            "bytes": total, "head_conversions": conversions,
            "scatter_s": time.perf_counter() - started}
