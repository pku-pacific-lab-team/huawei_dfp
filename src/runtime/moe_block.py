"""Serial router -> experts -> combine with one global storage schedule."""
from copy import deepcopy

from ..core import block_moe
from ..core.vector import positive
from ..core.vector_resources import resident_allocation
from ..hardware.blocks import BlockExecution, load_block_execution
from ..operators.adapters import adapt_native_result
from ..operators.moe_block import MoE, MoERouter, MoECombine, _RouterSelection
from ..operators.primitives import Gemm
from ..operators.types import OpResult
from .backends import get_backend
from .blocks import align, gemm_storage
from .composition import collect_children
from .moe import expert_allocations, _simulate_moe_experts
from .simulation import execution_profile, simulate_gemm


class Execution:
    def __init__(self, arch, spec, mode, execution, block_execution, tile_elements,
                 mem_slices, streams, reserved_bytes, hemisphere, *, selection):
        get_backend(arch, spec)
        self.arch, self.spec, self.mode, self.tile = arch, spec, mode, tile_elements
        self.execution = execution_profile(arch, execution, mode, composite=selection)
        self.block = load_block_execution() if block_execution is None else block_execution
        if not isinstance(self.block, BlockExecution):
            raise ValueError("Expected BlockExecution")
        self.allocation = None
        if arch in {"tsp", "tsp_cim"}:
            native = spec.native_reference_spec if arch == "tsp_cim" else spec
            self.allocation = resident_allocation(native, mem_slices, streams, reserved_bytes, hemisphere, label="MoE")
            self.capacity = native.mem_slice_capacity_bytes * len(self.allocation.slices)
            self.stripe = lambda size: align(size, len(self.allocation.slices))
        else:
            if mem_slices is not None or streams is not None or reserved_bytes != 0 or hemisphere != 0:
                raise ValueError("MEM allocation options require TSP or TSP-CIM")
            self.capacity = int(positive(spec.chip["hbm_usable_bytes"] if arch == "ascend" else spec.values["hbm_usable_bytes"], "HBM capacity"))
            self.stripe = lambda size: size
        self.reserved = self.stripe(reserved_bytes)
        self.children, self.timeline = [], []

    def check(self, size):
        if size > self.capacity:
            raise ValueError("unsupported_capacity: whole MoE live tensors and scratch exceed storage")

    def record(self, node, result, live, *, allocations=None):
        peak = result.native_report["memory"]["resident_bytes"] if self.allocation else live
        self.check(peak)
        if peak < live:
            raise ValueError("Native MoE allocation is smaller than caller live storage")
        self.children.append((node, result, 1))
        self.timeline.append({"node": node, "storage_bytes": live, "scratch_bytes": peak-live, "peak_bytes": peak,
                              "phase_allocations": allocations or {}})

    def arithmetic(self, node, op, live, kernel, *, allocations=None):
        self.check(live)
        path = getattr(block_moe, kernel + "_path")(op, self.arch, self.spec, self.execution, self.mode, live,
            self.capacity, self.allocation, self.tile)
        native = getattr(block_moe, kernel)(op, path, self.execution, self.block, self.mode)
        self.record(node, adapt_native_result(op.spec, self.arch, native, include_details=True), live, allocations=allocations)

    def router(self, op, persistent, occupied=()):
        gate = gate_op(op, occupied)
        sizes = gemm_storage(gate, self.arch, self.spec)
        logits = self.stripe(sizes[gate.output_name])
        live = persistent + logits
        self.check(live)
        policy = {}
        if self.allocation:
            policy.update(mem_slices=list(self.allocation.slices), streams=self.allocation.streams,
                          reserved_bytes=live - sum(self.stripe(v) for v in sizes.values()))
            if self.arch == "tsp_cim":
                policy["hemisphere"] = self.allocation.hemisphere
        result = simulate_gemm(gate, self.arch, self.spec, mode=self.mode, include_details=True, **policy)
        actual = (result.native_report["memory"]["resident_bytes"] if self.allocation else
                  result.native_report["memory"]["hbm"]["padded_footprint_bytes" if self.arch == "ascend" else "logical_footprint_bytes"])
        if actual != (live if self.allocation else sum(sizes.values())):
            raise ValueError("Native router GEMM storage differs from planner")
        self.record("router.gate", result, live, allocations={"logits": logits})
        workspace = block_moe.router_workspace(op, self.stripe)
        workspace.update(logits=logits, route_indices=self.stripe(op.spec.outputs[0].nbytes), route_weights=self.stripe(op.spec.outputs[1].nbytes))
        self.arithmetic("router.select", _RouterSelection(op, gate.spec.outputs[0]), persistent + sum(workspace.values()), "router", allocations=workspace)

    def finish(self, op, persistent, *, macs=0, routing=None, include_details=False):
        combined = collect_children(self.children, self.mode)
        peak = max(s["peak_bytes"] for s in self.timeline)
        combined["memory"]["mem" if self.allocation else "hbm"] = {
            "scope": "single_hem" if self.allocation else "chip", "capacity_bytes": self.capacity, "peak_bytes": peak}
        combined["memory"]["lifetime_validation"] = {"status": "validated_symbolic_schedule", "peak_bytes": peak,
            "persistent_allocations": persistent, "steps": self.timeline,
            "policy": "External inputs persist throughout. Router logits/workspace release after selection; indices release after experts; route weights and expert outputs survive through combine. Final output allocates only at combine. Each child reserves all other live storage."}
        boundary = {"scope": self.execution.policy["scope"], "chip_execution_included": False,
            "numeric_values_simulated": False, "input_output_alias_included": False, "layout": self.block.config["layout"],
            "start": "tensors_in_local_MEM" if self.allocation else "tensors_in_HBM",
            "end": "outputs_in_local_MEM" if self.allocation else "outputs_in_HBM",
            "intermediate_storage": "local_MEM" if self.allocation else "HBM", "offchip_supply_included": not bool(self.allocation)}
        if self.allocation:
            boundary.update(selected_hemisphere=self.allocation.hemisphere, mem_slices=list(self.allocation.slices), streams_per_direction=self.allocation.streams)
        report = {"schema_version": 1, "model": self.arch + "_" + op.operation, "mode": self.mode,
            "workload": op.spec.to_dict(), "boundary": boundary,
            "execution_contract": {"scope": self.execution.policy["scope"], "schedule": "serial_materialized_moe",
                "block_profile_reference_id": self.block.reference_id, "vector_profile_reference_id": self.execution.reference_id,
                "layout": self.block.config["layout"]},
            "memory": combined["memory"], "timing": combined["timing"], "action_counts": combined["action_counts"],
            "work_counts": {"logical": {"operations": combined["logical_operations"], "useful_macs": macs,
                "operation_counts": combined["operation_counts"], "output_bytes": sum(t.nbytes for t in op.spec.outputs)},
                "issued": {"operations": None, "status": "per_node_only"},
                "internal": {"boundary_elements": combined["boundary_elements"], "status": "per_node_only"}},
            "schedule": {"nodes": combined["nodes"]}, "block_execution_profile": deepcopy(self.block.config),
            "vector_execution_profile": deepcopy(self.execution.config),
            "limitations": ["One Decode token per batch item; serial single-resource execution, all declared weights resident.",
                "INT8 matrices/INT32 accumulation and FP32 nonlinear/combine work are performance assumptions, not BF16/FP8 numerical inference.",
                "Router sigmoid and repeated-max selection are coarse services; index generation, masks and layouts are symbolic, with no detailed Top-K/gather/scatter ISA.",
                "Router keeps whole score rows in local buffers; oversized rows fail capacity validation. Combine streams hidden tiles.",
                "No expert parallelism, weight offloading, full-model scheduling or chip-level partitioning."]}
        if routing is not None:
            report["routing"] = {**routing.to_dict(), "source": "caller_supplied_distribution_scenario", "router_work_included": True,
                                 "numeric_routing_verified": False}
        return OpResult.from_native(op.spec, self.arch, combined["ledger"], report, include_details=include_details)


def gate_op(op, occupied=()):
    names = {t.name for t in (*op.inputs, *op.spec.outputs, *occupied)}
    name = "router.logits"
    while name in names:
        name += "_"
    return Gemm.from_tensors(op.x, op.weight, output_name=name)


def prepare(arch, spec, options, *, selection):
    defaults = dict(mode="count", execution=None, block_execution=None, tile_elements=None, mem_slices=None,
                    streams=None, reserved_bytes=0, hemisphere=0)
    unknown = options.keys() - defaults.keys() - {"include_details"}
    if unknown:
        raise ValueError("Unknown MoE execution options: " + ", ".join(sorted(unknown)))
    return Execution(arch, spec, **{**defaults, **{k:v for k,v in options.items() if k != "include_details"}}, selection=selection)


def simulate_moe_combine(op, arch, spec, **options):
    if not isinstance(op, MoECombine):
        raise ValueError("Expected MoECombine")
    run = prepare(arch, spec, options, selection=False)
    persistent = {t.name: run.stripe(t.nbytes) for t in op.inputs}
    persistent = {"inputs": sum(persistent.values()), "reserved_bytes": run.reserved}
    output = run.stripe(op.spec.outputs[0].nbytes)
    run.arithmetic("combine", op, sum(persistent.values()) + output, "combine", allocations={"output": output})
    return run.finish(op, persistent, include_details=options.get("include_details", False))


def simulate_moe_router(op, arch, spec, **options):
    if not isinstance(op, MoERouter):
        raise ValueError("Expected MoERouter")
    run = prepare(arch, spec, options, selection=True)
    native = gemm_storage(gate_op(op), arch, spec)
    persistent = {t.name: run.stripe(max(t.nbytes, native.get(t.name, 0))) for t in op.inputs}
    persistent = {"inputs": sum(persistent.values()), "reserved_bytes": run.reserved}
    run.router(op, sum(persistent.values()))
    return run.finish(op, persistent, macs=gate_op(op).useful_macs, include_details=options.get("include_details", False))


def simulate_moe(op, arch, spec, **options):
    if not isinstance(op, MoE):
        raise ValueError("Expected MoE")
    run = prepare(arch, spec, options, selection=True)
    _, expert_globals, shared, shared_sizes = expert_allocations(op.experts, arch, spec, run.stripe, run.reserved)
    native = gemm_storage(gate_op(op.router), arch, spec)
    persistent = {k: v for k, v in expert_globals.items() if k not in {"route_indices", "routed_output"}}
    persistent["input"] = max(persistent["input"], run.stripe(native[op.router.x.name]))
    persistent["router_weight"] = run.stripe(native[op.router.weight.name])
    if op.router.bias is not None:
        persistent["router_bias"] = run.stripe(op.router.bias.nbytes)
    base = sum(persistent.values())
    run.router(op.router, base, (*op.inputs, *op.experts.spec.outputs, *op.spec.outputs))
    route_weight = run.stripe(op.router.spec.outputs[1].nbytes)
    extra = base - sum(v for k,v in expert_globals.items() if k not in {"route_indices", "routed_output"}) + route_weight
    child_options = {k:v for k,v in options.items() if k != "include_details"}
    child = _simulate_moe_experts(op.experts, arch, spec, **child_options, _parent_bytes=extra, include_details=True)
    run.children.append(("experts", child, 1))
    for row in child.native_report["memory"]["lifetime_validation"]["steps"]:
        run.check(row["peak_bytes"])
        run.timeline.append({**row, "node": "experts." + row["node"]})
    allocations = {"route_weights": route_weight, "routed_output": expert_globals["routed_output"],
                   "final_output": run.stripe(op.spec.outputs[0].nbytes)}
    if shared:
        allocations["shared_output"] = shared_sizes[shared.output_name]
    run.arithmetic("combine", op.combine(), base + sum(allocations.values()), "combine", allocations=allocations)
    macs = gate_op(op.router).useful_macs + child.work_counts["logical"]["useful_macs"]
    return run.finish(op, persistent, macs=macs, routing=op.experts.routing, include_details=options.get("include_details", False))
