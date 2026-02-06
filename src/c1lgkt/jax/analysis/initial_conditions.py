"""
Utilities for building particle initial conditions from TOML specs.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence
import zlib

import jax
import jax.numpy as jnp

from .parse_toml import load_toml


class InitialConditionError(ValueError):
    """Raised when an initial condition spec is invalid."""


DEFAULT_COORDS: tuple[str, ...] = ("R", "varphi", "Z", "upar", "mu")

TagArray = dict[str, jnp.ndarray]
Sampler = Callable[[Mapping[str, Any], jax.Array | None, int | None], jnp.ndarray]
Op = Callable[[Sequence[TagArray]], TagArray]


def register_sampler(name: str, fn: Sampler, registry: dict[str, Sampler] | None = None) -> dict[str, Sampler]:
    """
    Register a sampler function under a name. Returns the registry used.
    """
    reg = SAMPLERS if registry is None else registry
    reg[name] = fn
    return reg


def register_op(name: str, fn: Op, registry: dict[str, Op] | None = None) -> dict[str, Op]:
    """
    Register an operation under a name. Returns the registry used.
    """
    reg = OPS if registry is None else registry
    reg[name] = fn
    return reg


def build_initial_conditions_from_toml(path: str, *, coords: Sequence[str] | None = None) -> TagArray:
    """
    Convenience wrapper that loads a TOML file and evaluates its initial condition tree.
    """
    spec = load_toml(path)
    return build_initial_conditions(spec, coords=coords)


def build_initial_conditions(
    spec: Mapping[str, Any],
    *,
    coords: Sequence[str] | None = None,
    samplers: Mapping[str, Sampler] | None = None,
    ops: Mapping[str, Op] | None = None,
) -> TagArray:
    """
    Evaluate an initial condition spec and return a dict of tagged arrays.
    """
    ic_spec = spec.get("initial_conditions")
    if ic_spec is None:
        raise InitialConditionError("Missing [initial_conditions] table")

    nodes: Mapping[str, Any] | None = ic_spec.get("nodes")
    if not nodes:
        raise InitialConditionError("[initial_conditions] must define a non-empty nodes table")

    root = ic_spec.get("root")
    if not root:
        raise InitialConditionError("[initial_conditions] must define a root node")

    coord_order = tuple(ic_spec.get("coords", coords or DEFAULT_COORDS))
    seed = ic_spec.get("seed")

    sampler_registry = dict(SAMPLERS)
    if samplers:
        sampler_registry.update(samplers)

    op_registry = dict(OPS)
    if ops:
        op_registry.update(ops)

    cache: dict[str, TagArray] = {}

    def _eval_node(name: str) -> TagArray:
        if name in cache:
            return cache[name]
        if name not in nodes:
            raise InitialConditionError(f"Unknown node '{name}'")

        node = nodes[name]
        kind = node.get("kind")
        if kind is None:
            if "sampler" in node:
                kind = "leaf"
            elif "op" in node:
                kind = "op"

        if kind == "leaf":
            tag = node.get("tag")
            if not tag:
                raise InitialConditionError(f"Leaf node '{name}' missing tag")
            sampler_name = node.get("sampler")
            if not sampler_name:
                raise InitialConditionError(f"Leaf node '{name}' missing sampler")
            if sampler_name not in sampler_registry:
                raise InitialConditionError(f"Unknown sampler '{sampler_name}' in node '{name}'")
            sampler = sampler_registry[sampler_name]
            rng_key = _rng_for_node(seed, name, node)
            arr = sampler(node, rng_key, seed)
            arr = _as_1d_array(arr, name=f"{name}:{tag}")
            out = {tag: arr}
        elif kind == "op":
            op_name = node.get("op")
            if not op_name:
                raise InitialConditionError(f"Op node '{name}' missing op")
            if op_name not in op_registry:
                raise InitialConditionError(f"Unknown op '{op_name}' in node '{name}'")
            inputs = node.get("inputs")
            if not inputs:
                raise InitialConditionError(f"Op node '{name}' missing inputs")
            blocks = [_eval_node(n) for n in inputs]
            out = op_registry[op_name](blocks)
        else:
            raise InitialConditionError(f"Node '{name}' has invalid kind '{kind}'")

        _assert_valid_block(out, node=name)
        cache[name] = out
        return out

    result = _eval_node(root)
    _ensure_coord_coverage(result, coord_order)
    return result


def assemble_state(samples: TagArray, *, coords: Sequence[str] | None = None, axis: int = 0) -> jnp.ndarray:
    """
    Stack samples into a (5, N) array (axis=0) or (N, 5) array (axis=1).
    """
    coord_order = tuple(coords or DEFAULT_COORDS)
    _ensure_coord_coverage(samples, coord_order)
    arrays = [samples[c] for c in coord_order]
    return jnp.stack(arrays, axis=axis)


# ---- Samplers ----

def _sample_linspace(node: Mapping[str, Any], _key: jax.Array | None, _seed: int | None) -> jnp.ndarray:
    start = node.get("start")
    stop = node.get("stop")
    num = node.get("num")
    if start is None or stop is None or num is None:
        raise InitialConditionError("linspace sampler requires start, stop, num")
    endpoint = node.get("endpoint", True)
    return jnp.linspace(start, stop, int(num), endpoint=bool(endpoint))


def _sample_arange(node: Mapping[str, Any], _key: jax.Array | None, _seed: int | None) -> jnp.ndarray:
    start = node.get("start", 0.0)
    stop = node.get("stop")
    step = node.get("step", 1.0)
    if stop is None:
        raise InitialConditionError("arange sampler requires stop")
    return jnp.arange(start, stop, step)


def _sample_values(node: Mapping[str, Any], _key: jax.Array | None, _seed: int | None) -> jnp.ndarray:
    values = node.get("values")
    if values is None:
        raise InitialConditionError("values sampler requires values")
    return jnp.asarray(values)


def _sample_constant(node: Mapping[str, Any], _key: jax.Array | None, _seed: int | None) -> jnp.ndarray:
    value = node.get("value")
    num = node.get("num", 1)
    if value is None:
        raise InitialConditionError("constant sampler requires value")
    return jnp.full((int(num),), value)


def _sample_uniform(node: Mapping[str, Any], key: jax.Array | None, _seed: int | None) -> jnp.ndarray:
    if key is None:
        raise InitialConditionError("uniform sampler requires a seed")
    num = node.get("num")
    low = node.get("low", 0.0)
    high = node.get("high", 1.0)
    if num is None:
        raise InitialConditionError("uniform sampler requires num")
    return jax.random.uniform(key, shape=(int(num),), minval=low, maxval=high)


def _sample_normal(node: Mapping[str, Any], key: jax.Array | None, _seed: int | None) -> jnp.ndarray:
    if key is None:
        raise InitialConditionError("normal sampler requires a seed")
    num = node.get("num")
    mean = node.get("mean", 0.0)
    std = node.get("std", 1.0)
    if num is None:
        raise InitialConditionError("normal sampler requires num")
    return mean + std * jax.random.normal(key, shape=(int(num),))


SAMPLERS: dict[str, Sampler] = {
    "linspace": _sample_linspace,
    "arange": _sample_arange,
    "values": _sample_values,
    "constant": _sample_constant,
    "uniform": _sample_uniform,
    "normal": _sample_normal,
}


# ---- Operations ----

def _op_stack(blocks: Sequence[TagArray]) -> TagArray:
    merged: TagArray = {}
    expected_len: int | None = None
    normalized_blocks: list[tuple[int, TagArray]] = []

    for block in blocks:
        _assert_valid_block(block, node="stack")
        if not block:
            continue
        lengths = []
        norm_block: TagArray = {}
        for tag, arr in block.items():
            arr1 = _as_1d_array(arr, name=f"stack:{tag}")
            lengths.append(int(arr1.shape[0]))
            norm_block[tag] = arr1

        non_scalar = [n for n in lengths if n != 1]
        if non_scalar:
            if len(set(non_scalar)) != 1:
                raise InitialConditionError("stack op requires consistent lengths within each input")
            block_len = non_scalar[0]
        else:
            block_len = 1

        if block_len > 1:
            if expected_len is None:
                expected_len = block_len
            elif expected_len != block_len:
                raise InitialConditionError(
                    "stack op requires all inputs to have the same length (or be scalar)"
                )

        normalized_blocks.append((block_len, norm_block))

    if expected_len is None:
        expected_len = 1

    for block_len, block in normalized_blocks:
        if block_len not in (1, expected_len):
            raise InitialConditionError("stack op requires all inputs to have the same length (or be scalar)")
        for tag, arr in block.items():
            if tag in merged:
                raise InitialConditionError(f"stack op got duplicate tag '{tag}'")
            if arr.shape[0] == expected_len:
                merged[tag] = arr
            elif arr.shape[0] == 1 and expected_len > 1:
                merged[tag] = jnp.broadcast_to(arr, (expected_len,))
            else:
                raise InitialConditionError(
                    "stack op requires all inputs to have the same length (or be scalar)"
                )

    return merged


def _op_concat(blocks: Sequence[TagArray]) -> TagArray:
    if not blocks:
        return {}
    tag_set = set(blocks[0].keys())
    for block in blocks:
        _assert_valid_block(block, node="concat")
        if set(block.keys()) != tag_set:
            raise InitialConditionError("concat op requires all inputs to have identical tag sets")
    out: TagArray = {}
    for tag in tag_set:
        out[tag] = jnp.concatenate([b[tag] for b in blocks], axis=0)
    return out


def _op_meshgrid(blocks: Sequence[TagArray]) -> TagArray:
    if not blocks:
        return {}

    sizes = []
    normalized_blocks: list[TagArray] = []
    for block in blocks:
        _assert_valid_block(block, node="meshgrid")
        if not block:
            raise InitialConditionError("meshgrid op does not accept empty inputs")
        norm_block: TagArray = {}
        block_len = None
        for tag, arr in block.items():
            arr1 = _as_1d_array(arr, name=f"meshgrid:{tag}")
            if block_len is None:
                block_len = arr1.shape[0]
            elif arr1.shape[0] != block_len:
                raise InitialConditionError("meshgrid inputs must have consistent lengths per block")
            norm_block[tag] = arr1
        if block_len is None:
            block_len = 0
        sizes.append(block_len)
        normalized_blocks.append(norm_block)

    total = 1
    for s in sizes:
        total *= max(int(s), 1)

    out: TagArray = {}
    for i, block in enumerate(normalized_blocks):
        before = 1
        for s in sizes[:i]:
            before *= max(int(s), 1)
        after = 1
        for s in sizes[i + 1 :]:
            after *= max(int(s), 1)
        for tag, arr in block.items():
            expanded = jnp.tile(jnp.repeat(arr, after), before)
            if expanded.shape[0] != total:
                raise InitialConditionError("meshgrid failed to build consistent output length")
            out[tag] = expanded

    return out


OPS: dict[str, Op] = {
    "stack": _op_stack,
    "concat": _op_concat,
    "meshgrid": _op_meshgrid,
}


# ---- Helpers ----

def _as_1d_array(arr: Any, *, name: str) -> jnp.ndarray:
    a = jnp.asarray(arr)
    if a.ndim == 0:
        return a.reshape((1,))
    if a.ndim != 1:
        return a.reshape((-1,))
    return a


def _block_len(block: TagArray) -> int | None:
    if not block:
        return None
    first = next(iter(block.values()))
    first1 = _as_1d_array(first, name="block_len")
    return int(first1.shape[0])

def _assert_valid_block(block: TagArray, *, node: str) -> None:
    for tag, arr in block.items():
        if not isinstance(tag, str):
            raise InitialConditionError(f"Invalid tag type in {node}")
        a = jnp.asarray(arr)
        if a.ndim > 1:
            raise InitialConditionError(f"Non-1D arrays not supported for tag '{tag}' in {node}")
        _as_1d_array(arr, name=f"{node}:{tag}")


def _ensure_coord_coverage(block: TagArray, coords: Sequence[str]) -> None:
    missing = [c for c in coords if c not in block]
    if missing:
        raise InitialConditionError(f"Missing coordinate tags: {missing}")


def _rng_for_node(seed: int | None, name: str, node: Mapping[str, Any]) -> jax.Array | None:
    node_seed = node.get("seed", seed)
    if node_seed is None:
        return None
    name_hash = zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF
    key = jax.random.PRNGKey(int(node_seed))
    return jax.random.fold_in(key, int(name_hash))







