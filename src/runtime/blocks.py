"""Materialized single-resource block composition; native kernels own all work."""
from collections import Counter
from copy import deepcopy
from dataclasses import replace

from ..core.block_schedule import storage_schedule
from ..core.block_elementwise import make_path, simulate as run_elementwise
from ..core import block_rope, block_embedding
from ..core.metrics import attach_timing_breakdowns
from ..core.vector import positive
from ..core.vector_resources import resident_allocation
from ..hardware.blocks import BlockExecution, load_block_execution
from ..operators.actions import ActionLedger
from ..operators.adapters import adapt_native_result
from ..operators.blocks import block_steps
from ..operators.composite import RMSNorm
from ..operators.rope import _RotaryTensor
from ..operators.embedding import Embedding
from ..operators.primitives import Gemm
from ..operators.types import OpResult
from .backends import get_backend
from .simulation import execution_profile, simulate_gemm, simulate_composite


def align(size, unit):
    return (size + unit - 1) // unit * unit


def gemm_storage(op, arch, spec):
    """Native resident allocation dimensions; no cache/traffic formula is duplicated."""
    m, k, n = op.M, op.K, op.N
    if arch == "ascend":
        m, k, n = align(m, 16), align(k, 32), align(n, 16)
    elif arch in {"tsp", "tsp_cim"}:
        native = spec.native_reference_spec if arch == "tsp_cim" else spec
        if arch == "tsp_cim":
            n = align(n, spec.macro_spec.block_n_dim)
        k, n = align(k, native.supercell_dim), align(n, native.supercell_dim)
    return dict(zip((op.inputs[0].name, op.inputs[1].name, op.output_name), (m * k, k * n, 4 * m * n)))


def simulate_block(operator, architecture, spec, *, mode="count", execution=None, block_execution=None,
                   tile_elements=None, mem_slices=None, streams=None, reserved_bytes=0, hemisphere=0,
                   include_details=False):
    get_backend(architecture, spec)
    steps = block_steps(operator)
    execution = execution_profile(architecture, execution, mode, composite=any(isinstance(op, (RMSNorm, _RotaryTensor)) for _, op in steps))
    profile = load_block_execution() if block_execution is None else block_execution
    if not isinstance(profile, BlockExecution):
        raise ValueError("Expected a BlockExecution profile")
    resident = architecture in {"tsp", "tsp_cim"}
    allocation = None
    if resident:
        native = spec.native_reference_spec if architecture == "tsp_cim" else spec
        allocation = resident_allocation(native, mem_slices, streams, reserved_bytes, hemisphere, label="Block")
        capacity = native.mem_slice_capacity_bytes * len(allocation.slices)
        stripe = lambda size: align(size, len(allocation.slices))
    else:
        if mem_slices is not None or streams is not None or reserved_bytes != 0 or hemisphere != 0:
            raise ValueError("MEM allocation options require TSP or TSP-CIM")
        capacity = int(positive(spec.chip["hbm_usable_bytes"] if architecture == "ascend" else spec.values["hbm_usable_bytes"], "HBM capacity"))
        stripe = lambda size: size
    tensors = {t.name: t for t in operator.inputs}
    plan, sizes, native_sizes = [], {}, {}
    for node_id, op in steps:
        tensors.update({t.name: t for t in op.spec.outputs})
        plan.append({"id": node_id, "inputs": [t.name for t in op.inputs], "output": op.spec.outputs[0].name})
        if isinstance(op, Gemm):
            native_sizes[node_id] = gemm_storage(op, architecture, spec)
            for name, size in native_sizes[node_id].items():
                sizes[name] = max(sizes.get(name, 0), size)
    sizes = {name: stripe(max(t.nbytes, sizes.get(name, 0))) for name, t in tensors.items()}
    lifetime = storage_schedule([t.name for t in operator.inputs], [t.name for t in operator.spec.outputs],
                                plan, sizes, reserved_bytes=stripe(reserved_bytes))
    if lifetime["peak_bytes"] > capacity:
        raise ValueError("unsupported_capacity: live block tensors exceed " + ("local MEM" if resident else "HBM"))
    nodes, entries, native_actions, memory = [], [], {}, {}
    logical_ops, boundary_elements = Counter(), Counter()
    total_ops, useful_macs = 0, 0
    elapsed, services = {}, {}
    complete_services = True
    for (node_id, op), live in zip(steps, lifetime["steps"]):
        if isinstance(op, Gemm):
            policy = {}
            if resident:
                own = sum(stripe(size) for size in native_sizes[node_id].values())
                policy = dict(mem_slices=list(allocation.slices), streams=allocation.streams,
                              reserved_bytes=live["storage_bytes"] - own)
                if architecture == "tsp_cim":
                    policy["hemisphere"] = hemisphere
            result = simulate_gemm(op, architecture, spec, mode=mode, include_details=True, **policy)
            useful_macs += op.useful_macs
        elif isinstance(op, RMSNorm):
            policy = {}
            if resident:
                own = sum(stripe(t.nbytes) for t in (*op.inputs, *op.spec.outputs))
                policy = dict(mem_slices=list(allocation.slices), streams=allocation.streams,
                              reserved_bytes=live["storage_bytes"] - own, hemisphere=hemisphere)
            result = simulate_composite(op, architecture, spec, mode=mode, execution=execution,
                                        tile_elements=tile_elements, include_details=True, **policy)
        elif isinstance(op, (_RotaryTensor, Embedding)):
            kernel = block_rope if isinstance(op, _RotaryTensor) else block_embedding
            path = kernel.make_path(op, architecture, spec, execution, mode, live["storage_bytes"], capacity, allocation, tile_elements)
            native_result = (kernel.simulate(op, path, execution, profile, mode) if isinstance(op, _RotaryTensor)
                             else kernel.simulate(op, path, execution, mode))
            result = adapt_native_result(op.spec, architecture, native_result, include_details=True)
        else:
            path = make_path(op, architecture, spec, execution, mode, live["storage_bytes"], capacity, allocation, tile_elements)
            native_result = run_elementwise(op, path, execution, profile, mode)
            result = adapt_native_result(op.spec, architecture, native_result, include_details=True)
        report = result.native_report
        if isinstance(op, Gemm):
            # Fail explicitly if a native allocation contract changes underneath this planner.
            actual = (report["memory"]["resident_bytes"] if resident else report["memory"]["hbm"][
                "padded_footprint_bytes" if architecture == "ascend" else "logical_footprint_bytes"])
            expected = live["storage_bytes"] if resident else sum(native_sizes[node_id].values())
            if actual != expected:
                raise ValueError("Native GEMM allocation differs from the block storage contract")
        for action in result.action_ledger.entries:
            key = node_id + "." + action.id
            entries.append(replace(action, id=key, source=f"#/native_report/action_counts/{key}",
                                   attributes=action.attributes + (("block_node", node_id),),
                                   summarizes=tuple(node_id + "." + name for name in action.summarizes)))
            native_actions[key] = deepcopy(report["action_counts"][action.id])
        total_ops += report["work_counts"]["logical"]["operations"]
        logical_ops.update(report["work_counts"]["logical"].get("operation_counts", {}))
        boundary_elements.update(report["work_counts"]["internal"].get("boundary_elements", {}))
        live["scratch_bytes"] = report["memory"]["resident_bytes"] - live["storage_bytes"] if resident else 0
        if live["scratch_bytes"] < 0:
            raise ValueError("Native storage is smaller than the live block allocation")
        live["peak_bytes"] = live["storage_bytes"] + live["scratch_bytes"]
        if live["peak_bytes"] > capacity:
            raise ValueError("unsupported_capacity: live block tensors and scratch exceed local MEM")
        for name, req in result.resource_requirements.items():
            if name.lower() in {"mem", "hbm"}:
                continue
            key = name.lower()
            if key not in memory or req["required_bytes"] > memory[key]["peak_bytes"]:
                memory[key] = {"scope": req["scope"], "capacity_bytes": req["capacity_bytes"], "peak_bytes": req["required_bytes"]}
        if mode == "timing":
            elapsed[node_id] = result.cycle_count
            by_stage = report["timing"]["service_cycles"]["by_stage"]
            if by_stage is None:
                complete_services = False
            else:
                services.update({node_id + "." + key: value for key, value in by_stage.items()})
        nodes.append({"id": node_id, "operator": op.spec.to_dict(), "cycle_count": result.cycle_count,
                      "latency_s": result.latency_s, "native_report": report})
    peak = max(step["peak_bytes"] for step in lifetime["steps"])
    lifetime["peak_bytes"] = peak
    memory["mem" if resident else "hbm"] = {"scope": "single_hem" if resident else "chip",
                                            "capacity_bytes": capacity, "peak_bytes": peak}
    memory["lifetime_validation"] = lifetime
    total = sum(elapsed.values()) if mode == "timing" else None
    latency = None if mode == "count" or any(n["latency_s"] is None for n in nodes) else sum(n["latency_s"] for n in nodes)
    timing = {"status": "not_evaluated" if total is None else "analytical_estimate", "total_cycles": total,
              "time_ns": None if latency is None else latency * 1e9, "model": "serial_materialized_nodes"}
    attach_timing_breakdowns(timing, service=services if complete_services and mode == "timing" else None,
                            elapsed=elapsed if mode == "timing" else None)
    scope = execution.policy["scope"]
    boundary = {"scope": scope, "chip_execution_included": False, "numeric_values_simulated": False,
                "start": "tensors_in_local_MEM" if resident else "tensors_in_HBM",
                "end": "output_in_local_MEM" if resident else "output_in_HBM",
                "intermediate_storage": "local_MEM" if resident else "HBM",
                "offchip_supply_included": not resident, "input_output_alias_included": False,
                "layout": profile.config["layout"]}
    if resident:
        boundary.update(selected_hemisphere=hemisphere, mem_slices=list(allocation.slices), streams_per_direction=allocation.streams)
    report = {"schema_version": 1, "model": architecture + "_materialized_block", "mode": mode,
              "workload": operator.spec.to_dict(), "boundary": boundary,
              "execution_contract": {"scope": scope, "schedule": profile.config["schedule"],
                                     "storage": profile.config["storage"], "layout": profile.config["layout"],
                                     "profile_reference_id": profile.reference_id, "vector_profile_reference_id": execution.reference_id},
              "action_counts": native_actions, "memory": memory, "timing": timing,
              "work_counts": {"logical": {"operations": total_ops, "useful_macs": useful_macs,
                                          "operation_counts": dict(logical_ops), "output_bytes": sum(t.nbytes for t in operator.spec.outputs)},
                              "issued": {"operations": None, "status": "per_node_only", "basis": "See native matrix counts; no SIMD issue model"},
                              "internal": {"boundary_elements": dict(boundary_elements), "status": "per_node_only"}},
              "schedule": {"steps": plan, "nodes": nodes}, "block_execution_profile": deepcopy(profile.config),
              "vector_execution_profile": deepcopy(execution.config),
              "limitations": ["One core/TensorCore/hemisphere; no chip partition, communication or inter-node overlap.",
                              "All nodes materialize outputs; native GEMM cold-cache policies retained; no extra edge-transfer charge.",
                              "Symbolic padded views share one allocation per tensor; layout conversion instructions are not modeled.",
                              "INT8 boundaries use explicit uncalibrated work proxies, without scales, saturation values or numerical accuracy claims.",
                              "Inputs persist; intermediate storage releases after its last consumer. Local buffers are reused between nodes."]}
    if hasattr(operator, "storage_views"):
        report["schedule"]["storage_views"] = operator.storage_views
        report["limitations"].append("Head/matrix shapes are symbolic views of the same backing; physical packing requires layout validation, with no standalone reshape/transpose model.")
    return OpResult.from_native(operator.spec, architecture, ActionLedger(tuple(entries)), report, include_details=include_details)
