"""Shared latent/positional storage and serial absorbed MLA using native kernels."""
from copy import deepcopy

from ..core.kv_cache import make_path, simulate_payloads
from ..core.block_schedule import storage_schedule
from ..core.block_elementwise import make_path as elementwise_path, simulate as run_elementwise
from ..hardware.blocks import BlockExecution, load_block_execution
from ..operators.adapters import adapt_native_result
from ..operators.mla import MLACacheAccess, MLADecodeAttention, mla_attention_steps
from ..operators.blocks import block_steps
from ..operators.composite import Softmax
from ..operators.projections import AttentionOutputProjection
from ..operators.primitives import Gemm
from ..operators.types import OpResult
from .attention import prepare, boundary
from .blocks import gemm_storage
from .composition import collect_children
from .simulation import simulate_gemm, simulate_composite


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
    return cache_result(operator, architecture, spec, execution, profile, mode, live, capacity, allocation,
                        tile_elements, allocated, token_bytes, stripe(reserved_bytes), include_details=include_details)


def cache_state(operator, allocated):
    c, per_token = operator.cache, operator.cache.batch * sum(operator.cache.widths)
    return {"before": c.to_dict(), "after": operator.cache_after.to_dict(), "allocations": allocated,
            "filled_bytes_before": per_token * c.length, "filled_bytes_after": per_token * operator.cache_after.length,
            "appended_bytes": per_token * operator.append_tokens,
            "write_token_interval": [c.length, operator.cache_after.length],
            "output_aliases": {out.name: inp.name for out, inp in zip(operator.spec.outputs, c.buffers)},
            "semantics": "One shared latent and positional backing per batch; output descriptors alias existing storage. No numerical values."}


def cache_result(operator, architecture, spec, execution, profile, mode, live, capacity, allocation,
                 tile_elements, allocated, token_bytes, reserved, *, include_details):
    c = operator.cache
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
    state = cache_state(operator, allocated)
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
        "reserved_bytes": reserved,
        "policy": "Both preallocated backings and new tokens persist; outputs alias cache inputs; one reused transfer tile."}
    report["limitations"].extend([
        "INT8 performance what-if, not official BF16/FP8 numerical simulation.",
        "Capacity reserves the maximum native score/value view once; packing/transposition remains symbolic.",
        "Standalone read streams once per component; future attention must account for its own per-head reads."])
    return adapt_native_result(operator.spec, architecture, native, include_details=include_details)


def _plan(inputs, output, steps, arch, spec, stripe):
    tensors, sizes, native, plan = {t.name: t for t in inputs}, {}, {}, []
    for name, op in steps:
        tensors.update({t.name: t for t in op.spec.outputs})
        plan.append({"id": name, "inputs": [t.name for t in op.inputs], "output": op.spec.outputs[0].name})
        if isinstance(op, Gemm):
            native[name] = gemm_storage(op, arch, spec)
            for key, size in native[name].items():
                sizes[key] = max(sizes.get(key, 0), size)
    sizes = {name: stripe(max(t.nbytes, sizes.get(name, 0))) for name, t in tensors.items()}
    life = storage_schedule([t.name for t in inputs], [output.name], plan, sizes)
    return sizes, native, plan, life


def _run_stage(op, arch, spec, mode, execution, block, allocation, stripe, live, capacity, native, tile):
    """Run existing kernels with the complete caller-owned live storage reserved."""
    if live > capacity:
        raise ValueError("unsupported_capacity: MLA live tensors exceed storage")
    resident = allocation is not None
    policy = dict(mem_slices=list(allocation.slices), streams=allocation.streams) if resident else {}
    if isinstance(op, Gemm):
        if resident:
            policy["reserved_bytes"] = live - sum(stripe(v) for v in native.values())
            if arch == "tsp_cim":
                policy["hemisphere"] = allocation.hemisphere
        result = simulate_gemm(op, arch, spec, mode=mode, include_details=True, **policy)
        report = result.native_report
        actual = report["memory"]["resident_bytes"] if resident else report["memory"]["hbm"][
            "padded_footprint_bytes" if arch == "ascend" else "logical_footprint_bytes"]
        if actual != (live if resident else sum(native.values())):
            raise ValueError("Native GEMM allocation differs from MLA view contract")
    elif isinstance(op, Softmax):
        if resident:
            policy.update(reserved_bytes=live - sum(stripe(t.nbytes) for t in (*op.inputs, *op.spec.outputs)),
                          hemisphere=allocation.hemisphere)
        result = simulate_composite(op, arch, spec, mode=mode, execution=execution, tile_elements=tile,
                                    include_details=True, **policy)
    else:
        path = elementwise_path(op, arch, spec, execution, mode, live, capacity, allocation, tile)
        result = adapt_native_result(op.spec, arch, run_elementwise(op, path, execution, block, mode), include_details=True)
    peak = result.native_report["memory"]["resident_bytes"] if resident else live
    if peak < live or peak > capacity:
        raise ValueError("unsupported_capacity: MLA live tensors and scratch exceed storage")
    return result, peak


def simulate_mla_attention(operator, architecture, spec, *, mode="count", execution=None, attention_execution=None,
                           block_execution=None, tile_elements=None, mem_slices=None, streams=None,
                           reserved_bytes=0, hemisphere=0, include_details=False):
    if not isinstance(operator, MLADecodeAttention):
        raise ValueError("Expected MLADecodeAttention")
    execution, profile, allocation, capacity, stripe = prepare(architecture, spec, execution, attention_execution, mode,
        mem_slices, streams, reserved_bytes, hemisphere, softmax=True)
    block = load_block_execution() if block_execution is None else block_execution
    if not isinstance(block, BlockExecution):
        raise ValueError("Expected BlockExecution")
    c, d = operator.cache, operator.dimensions
    repeats = c.batch * d.query_heads
    inputs, steps = mla_attention_steps(operator)
    head_output = steps[-1][1].spec.outputs[0]
    sizes, native_sizes, plan, life = _plan(inputs, head_output, steps, architecture, spec, stripe)
    q, qp, wk, wv, ks, kp, vs = inputs
    allocated = cache_allocations(c, architecture, spec, stripe)
    for t, component in ((ks, "latent"), (vs, "latent"), (kp, "rope")):
        if sizes[t.name] > allocated[component]["bytes_per_batch"]:
            raise ValueError("MLA active cache view exceeds preallocated backing")
    globals_ = {"query_nope": sizes[q.name] * repeats, "query_rope": sizes[qp.name] * repeats,
                "kv_up_weight": (sizes[wk.name] + sizes[wv.name]) * d.query_heads,
                "latent_cache": allocated["latent"]["allocated_bytes"], "rope_cache": allocated["rope"]["allocated_bytes"],
                "context": sizes[head_output.name] * repeats, "reserved": stripe(reserved_bytes)}
    bindings = {q.name: "query_nope", qp.name: "query_rope", wk.name: "kv_up_weight", wv.name: "kv_up_weight",
                ks.name: "latent_cache", vs.name: "latent_cache", kp.name: "rope_cache", head_output.name: "context"}
    output_plan, output_life, output_steps, output_native, output_sizes = [], None, [], {}, {}
    if operator.output_weight is not None:
        projection = AttentionOutputProjection(operator.context, operator.output_weight, operator.output_name)
        output_steps = block_steps(projection)
        output_sizes, output_native, output_plan, output_life = _plan(
            projection.inputs, projection.spec.outputs[0], output_steps, architecture, spec, stripe)
        globals_["context"] = max(globals_["context"], output_sizes[operator.context.name])
        globals_["output_weight"] = output_sizes[operator.output_weight.name]
    base = sum(globals_.values())
    children, peaks = [], []
    state = cache_state(operator.cache_access, allocated)
    append_peak = None
    if operator.append_tokens:
        tokens = c.batch * sum(stripe(w) for w in c.widths)
        append_peak = base + tokens
        child = cache_result(operator.cache_access, architecture, spec, execution, profile, mode, append_peak, capacity,
            allocation, tile_elements, allocated, tokens, base - sum(v["allocated_bytes"] for v in allocated.values()), include_details=True)
        children.append(("cache_append", child, 1))
        peaks.append(append_peak)
    for (name, op), row in zip(steps, life["steps"]):
        live = base + sum(sizes[n] for n in row["live_tensors"] if n not in bindings)
        result, peak = _run_stage(op, architecture, spec, mode, execution, block, allocation, stripe, live,
                                  capacity, native_sizes.get(name), tile_elements)
        row.update(head_view_storage_bytes=row.pop("storage_bytes"), storage_bytes=live, scratch_bytes=peak - live, peak_bytes=peak)
        children.append((name, result, repeats))
        peaks.append(peak)
    if output_steps:
        output_bindings = {operator.context.name: "context", operator.output_weight.name: "output_weight"}
        for (name, op), row in zip(output_steps, output_life["steps"]):
            live = base + sum(output_sizes[n] for n in row["live_tensors"] if n not in output_bindings)
            result, peak = _run_stage(op, architecture, spec, mode, execution, block, allocation, stripe, live,
                                      capacity, output_native.get(name), tile_elements)
            row.update(projection_storage_bytes=row.pop("storage_bytes"), storage_bytes=live, scratch_bytes=peak - live, peak_bytes=peak)
            children.append(("output." + name, result, 1))
            peaks.append(peak)
        output_life.update(peak_bytes=max(r["peak_bytes"] for r in output_life["steps"]), view_bindings=output_bindings)
    aggregate = collect_children(children, mode)
    memory = aggregate["memory"]
    memory["mem" if allocation else "hbm"] = {"scope": "single_hem" if allocation else "chip", "capacity_bytes": capacity, "peak_bytes": max(peaks)}
    life.update(peak_bytes=max(peaks), global_allocations=globals_, view_bindings=bindings, head_repetitions=repeats,
        cache_append_peak_bytes=append_peak, output_projection=output_life,
        kv_weight_partitions={"key_bytes_per_head": sizes[wk.name], "value_bytes_per_head": sizes[wv.name], "head_count": d.query_heads},
        policy="Queries, weights, cache and context persist. One serial head workspace is reused. New tokens release after append; projection temporaries allocate after all heads. Bound views are not allocated twice.")
    memory["lifetime_validation"] = life
    length = operator.cache_access.cache_after.length
    macs = repeats * (d.qk_nope_head_dim * d.kv_lora_rank + 2 * length * d.kv_lora_rank +
                      length * d.qk_rope_head_dim + d.kv_lora_rank * d.v_head_dim)
    if output_steps:
        macs += c.batch * d.query_heads * d.v_head_dim * d.hidden_size
    report = {"schema_version": 1, "model": architecture + "_mla_decode_attention", "mode": mode,
        "workload": operator.spec.to_dict(), "boundary": boundary(architecture, execution, profile, allocation, state),
        "execution_contract": {"scope": execution.policy["scope"], "schedule": profile.config["schedule"], "reuse": profile.config["reuse"],
            "layout": profile.config["layout"], "profile_reference_id": profile.reference_id,
            "block_profile_reference_id": block.reference_id, "vector_profile_reference_id": execution.reference_id,
            "attention_impl": "absorb", "scale": operator.scale, "scale_multiplier": operator.scale_multiplier, "scale_source": operator.scale_source},
        "cache_state": state, "memory": memory, "timing": aggregate["timing"], "action_counts": aggregate["action_counts"],
        "work_counts": {"logical": {"operations": aggregate["logical_operations"], "useful_macs": macs,
            "operation_counts": aggregate["operation_counts"], "output_bytes": operator.spec.outputs[0].nbytes,
            "cache_appended_bytes": state["appended_bytes"], "cache_read_payload_bytes": repeats * length * (2 * d.kv_lora_rank + d.qk_rope_head_dim)},
            "issued": {"operations": None, "status": "per_node_only"}, "internal": {"boundary_elements": aggregate["boundary_elements"], "status": "per_node_only"}},
        "schedule": {"steps_per_head": plan, "output_projection_steps": output_plan, "nodes": aggregate["nodes"],
            "head_repetitions": repeats, "head_binding": "Each (batch, query head) shares that batch's latent/positional cache; head-specific K/V weight sections are shared across batches.",
            "order": "Append once, execute the complete template per batch/query head serially, then optionally project all contexts. Repetition counts summarize serial work."},
        "attention_execution_profile": deepcopy(profile.config), "block_execution_profile": deepcopy(block.config), "vector_execution_profile": deepcopy(execution.config),
        "limitations": ["INT8 performance what-if with explicit conversions; no numerical quantization accuracy claim.",
            "Preprocessing and table generation are separate. Uniform-length one-token Decode; position equals the newest cached token.",
            "K/V weight partitions, cache transposes and concatenated contexts use symbolic padded views; no physical repacking/ISA model.",
            "Cache capacity is shared, but native cold-cache reads repeat per query head. No extra standalone cache scan.",
            "Scale multiplier is explicit caller input (for example a squared YaRN mscale); no inferred context-extension policy.",
            "Single core/TensorCore/hemisphere; no paging, eviction, chip partition, inter-head overlap, MoE or full-model execution."]}
    return OpResult.from_native(operator.spec, architecture, aggregate["ledger"], report, include_details=include_details)
