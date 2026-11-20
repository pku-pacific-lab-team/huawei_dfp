"""Shared latent/positional cache storage, without MLA attention execution."""
from copy import deepcopy

from ..core.kv_cache import make_path, simulate_payloads
from ..operators.adapters import adapt_native_result
from ..operators.mla import MLACacheAccess
from ..operators.primitives import Gemm
from .attention import prepare, boundary
from .blocks import gemm_storage


def cache_allocations(cache, arch, spec, stripe):
    """One backing per batch/component, sized for future native GEMM views.

    Latent storage serves score and weighted-sum roles. Their maximum allocation
    is reserved once; physical layout compatibility still needs validation.
    """
    result = {}
    for label, width in zip(("latent", "rope"), cache.widths):
        views = {"score_weight": gemm_storage(Gemm(1, width, cache.capacity), arch, spec)["B"]}
        if label == "latent":
            views["weighted_value"] = gemm_storage(Gemm(1, cache.capacity, width), arch, spec)["B"]
        size = stripe(max(views.values()))
        result[label] = {"bytes_per_batch": size, "batch_count": cache.batch,
                         "allocated_bytes": cache.batch * size, "native_view_bytes": views,
                         "logical_capacity_bytes": cache.batch * cache.capacity * width}
    return result


def simulate_mla_cache(operator, architecture, spec, *, mode="count", execution=None, attention_execution=None,
                       tile_elements=None, mem_slices=None, streams=None, reserved_bytes=0,
                       hemisphere=0, include_details=False):
    if not isinstance(operator, MLACacheAccess):
        raise ValueError("Expected MLACacheAccess")
    execution, profile, allocation, capacity, stripe = prepare(architecture, spec, execution, attention_execution, mode,
        mem_slices, streams, reserved_bytes, hemisphere, softmax=False)
    c = operator.cache
    allocated = cache_allocations(c, architecture, spec, stripe)
    token_bytes = c.batch * sum(stripe(w) for w in c.widths) * operator.append_tokens
    live = sum(v["allocated_bytes"] for v in allocated.values()) + token_bytes + stripe(reserved_bytes)
    size = max(c.widths) * (operator.cache_after.length if operator.read_cache else 1)
    path = make_path(architecture, spec, execution, mode, live, capacity, allocation, tile_elements, size)
    phases = []
    for label, width in zip(("latent", "rope"), c.widths):
        if operator.append_tokens:
            phases.append((label + "_append", width, True, c.batch))
    for label, width in zip(("latent", "rope"), c.widths):
        if operator.read_cache:
            phases.append((label + "_read", operator.cache_after.length * width, False, c.batch))
    per_token = c.batch * sum(c.widths)
    native = simulate_payloads(operator, path, execution, mode, phases=phases,
        appended_bytes=per_token * operator.append_tokens,
        cache_read_bytes=per_token * operator.cache_after.length if operator.read_cache else 0,
        model="mla_compressed_cache_access",
        policy="Append shared latent KV and positional K once per batch; stream each filled component once. No expanded K/V cache or per-query-head repetition.")
    state = {"before": c.to_dict(), "after": operator.cache_after.to_dict(), "allocations": allocated,
             "filled_bytes_before": per_token * c.length, "filled_bytes_after": per_token * operator.cache_after.length,
             "appended_bytes": per_token * operator.append_tokens,
             "write_token_interval": [c.length, operator.cache_after.length],
             "output_aliases": {out.name: inp.name for out, inp in zip(operator.spec.outputs, c.buffers)},
             "semantics": "One shared latent and positional backing per batch; output descriptors alias existing storage. No numerical values."}
    report = native.report
    report["cache_state"] = state
    report["boundary"] = boundary(architecture, execution, profile, allocation, state)
    report["execution_contract"].update(profile_reference_id=profile.reference_id, cache_policy=profile.config["cache"],
        vector_profile_reference_id=execution.reference_id, shared_across_query_heads=True,
        read_policy="one_stream_per_batch_component", attention_core_included=False)
    report["attention_execution_profile"] = deepcopy(profile.config)
    report["vector_execution_profile"] = deepcopy(execution.config)
    report["memory"]["lifetime_validation"] = {
        "status": "validated_symbolic_schedule", "peak_bytes": live, "new_token_allocated_bytes": token_bytes,
        "reserved_bytes": stripe(reserved_bytes),
        "policy": "Both preallocated backings and new tokens persist; outputs alias cache inputs; one reused transfer tile."}
    report["limitations"].extend([
        "INT8 performance what-if, not official BF16/FP8 numerical simulation.",
        "Capacity reserves the maximum native score/value view once; packing/transposition remains symbolic.",
        "Standalone read streams once per component; future attention must account for its own per-head reads."])
    return adapt_native_result(operator.spec, architecture, native, include_details=include_details)
