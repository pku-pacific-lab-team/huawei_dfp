"""Decode head templates bound to shared, preallocated KV backing storage."""
from copy import deepcopy

from ..core.block_elementwise import make_path as make_elementwise_path, simulate as run_elementwise
from ..core.block_schedule import storage_schedule
from ..core.kv_cache import make_path as make_cache_path, simulate as run_cache
from ..core.vector import positive
from ..core.vector_resources import resident_allocation
from ..hardware.attention import AttentionExecution, load_attention_execution
from ..hardware.blocks import BlockExecution, load_block_execution
from ..operators.adapters import adapt_native_result
from ..operators.attention import KVCacheAccess, DecodeAttention, attention_steps
from ..operators.composite import Softmax
from ..operators.primitives import Gemm
from ..operators.types import OpResult
from .backends import get_backend
from .blocks import align, gemm_storage
from .composition import collect_children
from .simulation import execution_profile, simulate_gemm, simulate_composite


def prepare(arch, spec, execution, attention_execution, mode, mem_slices, streams, reserved_bytes, hemisphere, *, softmax):
    get_backend(arch, spec)
    execution = execution_profile(arch, execution, mode, composite=softmax)
    profile = load_attention_execution() if attention_execution is None else attention_execution
    if not isinstance(profile, AttentionExecution):
        raise ValueError("Expected AttentionExecution")
    allocation = None
    if arch in {"tsp", "tsp_cim"}:
        native = spec.native_reference_spec if arch == "tsp_cim" else spec
        allocation = resident_allocation(native, mem_slices, streams, reserved_bytes, hemisphere, label="Attention")
        capacity = native.mem_slice_capacity_bytes * len(allocation.slices)
        stripe = lambda size: align(size, len(allocation.slices))
    else:
        if mem_slices is not None or streams is not None or reserved_bytes != 0 or hemisphere != 0:
            raise ValueError("MEM allocation options require TSP or TSP-CIM")
        capacity = int(positive(spec.chip["hbm_usable_bytes"] if arch == "ascend" else spec.values["hbm_usable_bytes"], "HBM capacity"))
        stripe = lambda size: size
    return execution, profile, allocation, capacity, stripe


def cache_allocations(cache, arch, spec, stripe):
    # Reserve each KV head independently for the largest native view up to capacity.
    key = gemm_storage(Gemm(1, cache.head_dim, cache.capacity), arch, spec)["B"]
    value = gemm_storage(Gemm(1, cache.capacity, cache.head_dim), arch, spec)["B"]
    heads = cache.batch * cache.kv_heads
    return {name: {"bytes_per_head": stripe(size), "head_count": heads, "allocated_bytes": heads * stripe(size),
                   "logical_capacity_bytes": heads * cache.capacity * cache.head_dim}
            for name, size in (("key", key), ("value", value))}


def state_description(access, allocations):
    c = access.cache
    return {"before": c.to_dict(), "after": access.cache_after.to_dict(), "allocations": deepcopy(allocations),
            "filled_bytes_before": 2 * c.batch * c.kv_heads * c.length * c.head_dim,
            "filled_bytes_after": 2 * c.batch * c.kv_heads * access.cache_after.length * c.head_dim,
            "appended_bytes": 2 * c.batch * c.kv_heads * c.head_dim * access.append_tokens,
            "write_token_interval": [c.length, access.cache_after.length],
            "output_aliases": {out.name: inp.name for out, inp in zip(access.spec.outputs, c.buffers)},
            "semantics": "Immutable length/version descriptors over the same preallocated backing buffers; no numerical cache values are stored."}


def boundary(arch, execution, profile, allocation, state):
    resident = allocation is not None
    result = {"scope": execution.policy["scope"], "chip_execution_included": False, "numeric_values_simulated": False,
              "start": "tensors_in_local_MEM" if resident else "tensors_in_HBM",
              "end": "tensors_in_local_MEM" if resident else "tensors_in_HBM",
              "offchip_supply_included": not resident, "layout": profile.config["layout"],
              "state_aliases": state["output_aliases"], "cache_update": "append_only_within_capacity"}
    if resident:
        result.update(selected_hemisphere=allocation.hemisphere, mem_slices=list(allocation.slices), streams_per_direction=allocation.streams)
    return result


def cache_result(op, arch, spec, execution, profile, mode, live, capacity, allocation, tile_elements, allocations, *, include_details):
    size = (op.cache_after.length if op.read_cache else 1) * op.cache.head_dim
    path = make_cache_path(arch, spec, execution, mode, live, capacity, allocation, tile_elements, size)
    native = run_cache(op, path, execution, mode)
    state = state_description(op, allocations)
    native.report["cache_state"] = state
    native.report["boundary"] = boundary(arch, execution, profile, allocation, state)
    native.report["execution_contract"].update(profile_reference_id=profile.reference_id, cache_policy=profile.config["cache"],
                                               vector_profile_reference_id=execution.reference_id)
    native.report["attention_execution_profile"] = deepcopy(profile.config)
    native.report["vector_execution_profile"] = deepcopy(execution.config)
    native.report["memory"]["lifetime_validation"] = {
        "status": "validated_symbolic_schedule", "peak_bytes": live,
        "policy": "Preallocated K/V and new-token inputs remain resident; state outputs alias the cache; transfer buffer is reused."}
    return adapt_native_result(op.spec, arch, native, include_details=include_details)


def simulate_kv_cache(operator, architecture, spec, *, mode="count", execution=None, attention_execution=None,
                      tile_elements=None, mem_slices=None, streams=None, reserved_bytes=0, hemisphere=0, include_details=False):
    if not isinstance(operator, KVCacheAccess):
        raise ValueError("Expected KVCacheAccess")
    execution, profile, allocation, capacity, stripe = prepare(architecture, spec, execution, attention_execution, mode,
        mem_slices, streams, reserved_bytes, hemisphere, softmax=False)
    allocated = cache_allocations(operator.cache, architecture, spec, stripe)
    tokens = 2 * operator.cache.batch * operator.cache.kv_heads * stripe(operator.cache.head_dim) * operator.append_tokens
    live = sum(v["allocated_bytes"] for v in allocated.values()) + tokens + stripe(reserved_bytes)
    return cache_result(operator, architecture, spec, execution, profile, mode, live, capacity, allocation, tile_elements,
                        allocated, include_details=include_details)


def simulate_attention(operator, architecture, spec, *, mode="count", execution=None, attention_execution=None,
                       block_execution=None, tile_elements=None, mem_slices=None, streams=None,
                       reserved_bytes=0, hemisphere=0, include_details=False):
    if not isinstance(operator, DecodeAttention):
        raise ValueError("Expected DecodeAttention")
    execution, profile, allocation, capacity, stripe = prepare(architecture, spec, execution, attention_execution, mode,
        mem_slices, streams, reserved_bytes, hemisphere, softmax=True)
    block = load_block_execution() if block_execution is None else block_execution
    if not isinstance(block, BlockExecution):
        raise ValueError("Expected BlockExecution")
    resident, c = allocation is not None, operator.cache
    repetitions = c.batch * operator.query_heads
    inputs, steps = attention_steps(operator)
    sizes, native_sizes, tensors, plan = {}, {}, {t.name: t for t in inputs}, []
    for node_id, op in steps:
        tensors.update({t.name: t for t in op.spec.outputs})
        plan.append({"id": node_id, "inputs": [t.name for t in op.inputs], "output": op.spec.outputs[0].name})
        if isinstance(op, Gemm):
            native_sizes[node_id] = gemm_storage(op, architecture, spec)
            for name, size in native_sizes[node_id].items():
                sizes[name] = max(sizes.get(name, 0), size)
    sizes = {name: stripe(max(t.nbytes, sizes.get(name, 0))) for name, t in tensors.items()}
    output = steps[-1][1].spec.outputs[0]
    lifetime = storage_schedule([t.name for t in inputs], [output.name], plan, sizes)
    allocated = cache_allocations(c, architecture, spec, stripe)
    global_allocations = {"query": sizes[inputs[0].name] * repetitions,
                          "key_cache": allocated["key"]["allocated_bytes"], "value_cache": allocated["value"]["allocated_bytes"],
                          "output": sizes[output.name] * repetitions, "reserved": stripe(reserved_bytes)}
    # These four names refer to views of global backing, never additional allocations.
    bindings = {inputs[0].name: "query", inputs[1].name: "key_cache", inputs[2].name: "value_cache", output.name: "output"}
    if sizes[inputs[1].name] > allocated["key"]["bytes_per_head"] or sizes[inputs[2].name] > allocated["value"]["bytes_per_head"]:
        raise ValueError("Active KV view exceeds its preallocated head backing")
    base = sum(global_allocations.values())
    state = state_description(operator.cache_access, allocated)
    children, peaks = [], []
    if operator.append_tokens:
        new_tokens = 2 * c.batch * c.kv_heads * stripe(c.head_dim)
        cache = cache_result(operator.cache_access, architecture, spec, execution, profile, mode,
                             base + new_tokens, capacity, allocation, tile_elements, allocated, include_details=True)
        children.append(("cache_append", cache, 1))
        peaks.append(base + new_tokens)
    for (node_id, op), row in zip(steps, lifetime["steps"]):
        live = base + sum(sizes[name] for name in row["live_tensors"] if name not in bindings)
        row["head_view_storage_bytes"] = row.pop("storage_bytes")
        row["storage_bytes"] = live
        if live > capacity:
            raise ValueError("unsupported_capacity: shared KV Cache, query/output and attention intermediates exceed storage")
        policy = {}
        if resident:
            policy.update(mem_slices=list(allocation.slices), streams=allocation.streams)
        if isinstance(op, Gemm):
            if resident:
                policy["reserved_bytes"] = live - sum(stripe(size) for size in native_sizes[node_id].values())
                if architecture == "tsp_cim":
                    policy["hemisphere"] = hemisphere
            result = simulate_gemm(op, architecture, spec, mode=mode, include_details=True, **policy)
            report = result.native_report
            actual = report["memory"]["resident_bytes"] if resident else report["memory"]["hbm"][
                "padded_footprint_bytes" if architecture == "ascend" else "logical_footprint_bytes"]
            expected = live if resident else sum(native_sizes[node_id].values())
            if actual != expected:
                raise ValueError("Native GEMM allocation differs from the attention view contract")
        elif isinstance(op, Softmax):
            if resident:
                policy.update(reserved_bytes=live - sum(stripe(t.nbytes) for t in (*op.inputs, *op.spec.outputs)), hemisphere=hemisphere)
            result = simulate_composite(op, architecture, spec, mode=mode, execution=execution,
                                        tile_elements=tile_elements, include_details=True, **policy)
        else:
            path = make_elementwise_path(op, architecture, spec, execution, mode, live, capacity, allocation, tile_elements)
            result = adapt_native_result(op.spec, architecture, run_elementwise(op, path, execution, block, mode), include_details=True)
        report = result.native_report
        peak = report["memory"]["resident_bytes"] if resident else live
        if peak > capacity or peak < live:
            raise ValueError("Invalid attention live/scratch allocation")
        row.update(scratch_bytes=peak - live, peak_bytes=peak)
        peaks.append(peak)
        children.append((node_id, result, repetitions))
    aggregate = collect_children(children, mode)
    memory = aggregate["memory"]
    memory["mem" if resident else "hbm"] = {"scope": "single_hem" if resident else "chip", "capacity_bytes": capacity, "peak_bytes": max(peaks)}
    lifetime.update(peak_bytes=max(peaks), global_allocations=global_allocations, view_bindings=bindings,
                    head_repetitions=repetitions, cache_append_peak_bytes=peaks[0] if operator.append_tokens else None,
                    policy="Global query/cache/output allocations persist; one serial query-head workspace is reused. New K/V inputs release after append. Bound views are not allocated twice.")
    memory["lifetime_validation"] = lifetime
    length = c.length + operator.append_tokens
    useful_macs = 2 * repetitions * length * c.head_dim
    report = {"schema_version": 1, "model": architecture + "_decode_attention", "mode": mode,
              "workload": operator.spec.to_dict(), "boundary": boundary(architecture, execution, profile, allocation, state),
              "execution_contract": {"scope": execution.policy["scope"], "schedule": profile.config["schedule"],
                                     "reuse": profile.config["reuse"], "layout": profile.config["layout"],
                                     "profile_reference_id": profile.reference_id, "block_profile_reference_id": block.reference_id,
                                     "vector_profile_reference_id": execution.reference_id},
              "cache_state": state, "memory": memory, "timing": aggregate["timing"], "action_counts": aggregate["action_counts"],
              "work_counts": {"logical": {"operations": aggregate["logical_operations"], "useful_macs": useful_macs,
                                          "operation_counts": aggregate["operation_counts"], "output_bytes": operator.query.nbytes,
                                          "cache_appended_bytes": state["appended_bytes"],
                                          "cache_read_payload_bytes": 2 * repetitions * length * c.head_dim},
                              "issued": {"operations": None, "status": "per_node_only"},
                              "internal": {"boundary_elements": aggregate["boundary_elements"], "status": "per_node_only"}},
              "schedule": {"steps_per_head": plan, "nodes": aggregate["nodes"], "head_repetitions": repetitions,
                           "group_size": operator.query_heads // c.kv_heads,
                           "head_binding": "(batch b, query head h) uses cache (b, floor(h / group_size)); no cross-head cache retention",
                           "order": "Append once; for each batch/query head execute the ENTIRE head template before the next head. Node repetition counts are accounting summaries."},
              "attention_execution_profile": deepcopy(profile.config), "block_execution_profile": deepcopy(block.config),
              "vector_execution_profile": deepcopy(execution.config),
              "limitations": ["One-token uniform-length Decode only; Q/new K/V are already projected and position-processed. No projection, RoPE, output projection, padding mask or sliding window.",
                              "GQA/MQA share KV backing capacity; every query head retains native cold GEMM reads. No materialized repeat_kv tensor.",
                              "INT8 Q/K/V/probability/output storage is a what-if assumption, with FP32 scale/Softmax and coarse boundary costs; no numerical accuracy claim.",
                              "K transpose, head slices and append offsets are symbolic padded views, not a physical layout or transpose-instruction model.",
                              "No cache growth, eviction, offchip paging, MLA, MoE, chip partition or inter-head overlap."]}
    return OpResult.from_native(operator.spec, architecture, aggregate["ledger"], report, include_details=include_details)
