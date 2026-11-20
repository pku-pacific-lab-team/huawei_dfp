"""Serial fixed-routing experts with globally resident weights and bounded scratch."""
from copy import deepcopy

from ..core import moe as payload
from ..core.vector import positive
from ..core.vector_resources import resident_allocation
from ..hardware.blocks import BlockExecution, load_block_execution
from ..operators.adapters import adapt_native_result
from ..operators.blocks import block_steps
from ..operators.moe import MoEExperts, _MoEPayload
from ..operators.primitives import Gemm
from ..operators.types import OpResult
from .backends import get_backend
from .blocks import align, gemm_storage, simulate_block
from .composition import collect_children
from .simulation import execution_profile


def binding_sizes(operator, architecture, spec, stripe):
    """Native FFN input views; execution and action counts remain in simulate_block."""
    sizes = {t.name: t.nbytes for t in (*operator.inputs, *operator.spec.outputs)}
    for _, stage in block_steps(operator):
        if isinstance(stage, Gemm):
            for name, size in gemm_storage(stage, architecture, spec).items():
                if name in sizes:
                    sizes[name] = max(sizes[name], size)
    return {name: stripe(size) for name, size in sizes.items()}


def expert_allocations(operator, architecture, spec, stripe, reserved_bytes=0):
    """Global expert storage, shared by standalone and whole-MoE planners."""
    r, h, i = operator.routing, operator.hidden, operator.intermediate
    per_weight = (stripe(gemm_storage(Gemm(1, h, i), architecture, spec)["B"]),) * 2 + (
                  stripe(gemm_storage(Gemm(1, i, h), architecture, spec)["B"]),)
    globals_ = {"input": stripe(operator.x.nbytes), "route_indices": stripe(r.indices.nbytes),
                "routed_output": stripe(operator.spec.outputs[0].nbytes), "reserved": stripe(reserved_bytes)}
    for key, size in zip(("routed_gate_weight", "routed_up_weight", "routed_down_weight"), per_weight):
        globals_[key] = r.expert_count * size
    shared, shared_sizes = operator.shared(), {}
    if shared:
        shared_sizes = binding_sizes(shared, architecture, spec, stripe)
        globals_["input"] = max(globals_["input"], shared_sizes[shared.x.name])
        for key, weight in zip(("shared_gate_weight", "shared_up_weight", "shared_down_weight"), shared.inputs[1:]):
            globals_[key] = shared_sizes[weight.name]
    return per_weight, globals_, shared, shared_sizes


def simulate_moe_experts(operator, architecture, spec, **options):
    # Parent-owned HBM reservations are internal; public allocation options retain
    # the same validation as other operators.
    if "_parent_bytes" in options:
        raise ValueError("Parent reservations are internal to whole MoE")
    return _simulate_moe_experts(operator, architecture, spec, **options)


def _simulate_moe_experts(operator, architecture, spec, *, mode="count", execution=None, block_execution=None,
                         tile_elements=None, mem_slices=None, streams=None, reserved_bytes=0,
                         hemisphere=0, include_details=False, _parent_bytes=0):
    if not isinstance(operator, MoEExperts):
        raise ValueError("Expected MoEExperts")
    get_backend(architecture, spec)
    execution = execution_profile(architecture, execution, mode)
    block = load_block_execution() if block_execution is None else block_execution
    if not isinstance(block, BlockExecution):
        raise ValueError("Expected BlockExecution")
    resident, allocation = architecture in {"tsp", "tsp_cim"}, None
    if resident:
        native = spec.native_reference_spec if architecture == "tsp_cim" else spec
        allocation = resident_allocation(native, mem_slices, streams, reserved_bytes, hemisphere, label="MoE")
        capacity = native.mem_slice_capacity_bytes * len(allocation.slices)
        stripe = lambda size: align(size, len(allocation.slices))
    else:
        if mem_slices is not None or streams is not None or reserved_bytes != 0 or hemisphere != 0:
            raise ValueError("MEM allocation options require TSP or TSP-CIM")
        capacity = int(positive(spec.chip["hbm_usable_bytes"] if architecture == "ascend" else spec.values["hbm_usable_bytes"], "HBM capacity"))
        stripe = lambda size: size
    r, h, i = operator.routing, operator.hidden, operator.intermediate
    per_weight, globals_, shared, shared_sizes = expert_allocations(operator, architecture, spec, stripe, reserved_bytes)
    if _parent_bytes:
        globals_["parent_live_tensors"] = stripe(_parent_bytes)
    # Routed output slots persist. The shared final output is allocated by its
    # own last conversion, after all routed outputs have been published.
    base = sum(globals_.values())
    children, timeline, experts = [], [], []

    def check(size):
        if size > capacity:
            raise ValueError("unsupported_capacity: MoE live tensors and scratch exceed storage")

    def transfer(node, source, target, row_bytes, rows, live):
        check(live)
        op = _MoEPayload(source, target, row_bytes, rows, "moe_" + node.split(".")[-1])
        path = payload.make_path(architecture, spec, execution, mode, live, capacity, allocation, tile_elements, row_bytes)
        result = adapt_native_result(op.spec, architecture, payload.simulate_payload(op, path, execution, mode), include_details=True)
        children.append((node, result, 1))
        timeline.append({"node": node, "storage_bytes": live, "scratch_bytes": 0, "peak_bytes": live})

    def run_expert(node, expert, sizes, extra):
        policy = dict(mem_slices=list(allocation.slices), streams=allocation.streams,
                      reserved_bytes=extra, hemisphere=hemisphere) if resident else {}
        result = simulate_block(expert, architecture, spec, mode=mode, execution=execution, block_execution=block,
                                tile_elements=tile_elements, include_details=True, **policy)
        life = result.native_report["memory"]["lifetime_validation"]
        actual = {t["tensor"]: t["bytes"] for t in life["lifetimes"]}
        if any(actual[name] != size for name, size in sizes.items()):
            raise ValueError("Native FFN allocation differs from MoE view contract")
        for row in life["steps"]:
            row = {**row, "node": node + "." + row["node"]}
            if not resident:
                row["storage_bytes"] += extra
                row["peak_bytes"] += extra
            check(row["peak_bytes"])
            timeline.append(row)
        children.append((node, result, 1))
        return result

    check(base)
    transfer("route_read", r.indices, None, 4 * r.top_k, r.batch, base)
    for e, slots in enumerate(r.slots_by_expert):
        if not slots:
            continue
        expert = operator.expert(e)
        sizes = binding_sizes(expert, architecture, spec, stripe)
        weights = tuple(sizes[t.name] for t in expert.inputs[1:])
        if weights != per_weight:
            raise ValueError("Native expert weight storage unexpectedly depends on token count")
        tag, n = f"expert_{e}", len(slots)
        transfer(tag + ".dispatch", operator.x, expert.x, h, n, base + sizes[expert.x.name])
        run_expert(tag + ".ffn", expert, sizes, base - sum(weights))
        # The packed input and FFN intermediates are now dead; only this
        # expert's FP32 result overlaps its final destination slots.
        transfer(tag + ".publish", expert.spec.outputs[0], operator.spec.outputs[0], 4 * h, n,
                 base + sizes[expert.output_name])
        experts.append({"expert": e, "token_count": n, "source_tokens": [token for token, _ in slots],
            "destination_slots": [list(slot) for slot in slots], "input_allocated_bytes": sizes[expert.x.name],
            "output_allocated_bytes": sizes[expert.output_name], "weight_view_bytes": list(weights),
            "view_bindings": {t.name: {"backing": key, "expert": e} for t, key in zip(expert.inputs[1:],
                                ("routed_gate_weight", "routed_up_weight", "routed_down_weight"))}})
    if shared:
        run_expert("shared.ffn", shared, shared_sizes,
                   base - sum(shared_sizes[t.name] for t in shared.inputs))
        globals_["shared_output"] = shared_sizes[shared.output_name]
    aggregate = collect_children(children, mode)
    peak = max(row["peak_bytes"] for row in timeline)
    memory = aggregate["memory"]
    memory["mem" if resident else "hbm"] = {"scope": "single_hem" if resident else "chip", "capacity_bytes": capacity, "peak_bytes": peak}
    memory["lifetime_validation"] = {"status": "validated_symbolic_schedule", "global_allocations": globals_,
        "persistent_bytes_before_shared_output": base, "routed_weight_bytes_per_expert": list(per_weight),
        "steps": timeline, "peak_bytes": peak,
        "policy": "All declared weights, input, indices and routed output slots persist. One expert's packed rows and FFN workspace live at a time. Publish overlaps only its temporary FP32 result. Shared output allocates in the final FFN conversion."}
    boundary = {"scope": execution.policy["scope"], "chip_execution_included": False, "numeric_values_simulated": False,
        "start": "tensors_in_local_MEM" if resident else "tensors_in_HBM", "end": "outputs_in_local_MEM" if resident else "outputs_in_HBM",
        "intermediate_storage": "local_MEM" if resident else "HBM", "offchip_supply_included": not resident,
        "input_output_alias_included": False, "layout": block.config["layout"]}
    if resident:
        boundary.update(selected_hemisphere=hemisphere, mem_slices=list(allocation.slices), streams_per_direction=allocation.streams)
    macs = 3 * h * i * r.batch * (r.top_k + operator.shared_experts)
    report = {"schema_version": 1, "model": architecture + "_moe_experts", "mode": mode,
        "workload": operator.spec.to_dict(), "boundary": boundary,
        "execution_contract": {"scope": execution.policy["scope"], "schedule": "serial_active_experts_then_shared",
            "layout": block.config["layout"], "weight_storage": "all_declared_experts_resident",
            "routing_compute_included": False, "weighted_combine_included": False,
            "block_profile_reference_id": block.reference_id, "vector_profile_reference_id": execution.reference_id},
        "routing": r.to_dict(), "memory": memory, "timing": aggregate["timing"], "action_counts": aggregate["action_counts"],
        "work_counts": {"logical": {"operations": aggregate["logical_operations"], "useful_macs": macs,
            "operation_counts": aggregate["operation_counts"], "output_bytes": sum(t.nbytes for t in operator.spec.outputs),
            "route_index_read_bytes": r.indices.nbytes, "dispatch_payload_bytes": r.batch * r.top_k * h,
            "publish_payload_bytes": 4 * r.batch * r.top_k * h,
            "declared_weight_logical_bytes": 3 * h * i * (r.expert_count + operator.shared_experts),
            "active_weight_logical_bytes": 3 * h * i * (len(experts) + operator.shared_experts)},
            "issued": {"operations": None, "status": "per_node_only"},
            "internal": {"boundary_elements": aggregate["boundary_elements"], "status": "per_node_only"}},
        "schedule": {"experts": experts, "skipped_experts": [e for e, n in enumerate(r.counts) if not n],
            "nodes": aggregate["nodes"], "shared_ffn_included": bool(shared),
            "output_semantics": "Unweighted FP32 [batch, selection_slot, hidden] plus optional FP32 shared branch; no sum across slots."},
        "block_execution_profile": deepcopy(block.config), "vector_execution_profile": deepcopy(execution.config),
        "limitations": ["Fixed caller-provided routing, one Decode token per batch item; no gate scores, Top-K/group selection, route weights or combine.",
            "INT8 matrices and SwiGLU storage with FP32 expert outputs are performance assumptions, not official BF16/FP8 numerical inference.",
            "All declared expert weights must fit in HBM/local MEM; inactive weights occupy capacity but execute no kernels or weight loads.",
            "Dispatch/publish charge row payload copies; selection, padding views and offsets are symbolic, not gather/scatter ISA or physical transactions.",
            "No expert parallelism, offloaded weight streaming, inter-resource communication or full-model execution."]}
    return OpResult.from_native(operator.spec, architecture, aggregate["ledger"], report, include_details=include_details)
