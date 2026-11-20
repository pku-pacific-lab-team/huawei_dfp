"""Serial FP32 block arithmetic with explicit stored-type boundary work."""
from collections import Counter
from copy import deepcopy
from fractions import Fraction

from .composite_memory import LocalMemoryPath, ResidentMemoryPath
from .metrics import attach_timing_breakdowns
from .operator_result import ActionSpec, NativeResult
from .vector import cycles, positive, tile_groups
from .vector_resources import ascend_services, tpu_services, resident_services


def make_path(op, arch, spec, execution, mode, live_bytes, capacity, allocation, tile_elements):
    """Reserve a disjoint tile workspace; external tensor storage is owned by the graph."""
    output = op.spec.outputs[0]
    scratch_per_element = 4 * sum(t.dtype != "FP32" for t in op.inputs)
    if op.operation == "storage_boundary" and output.dtype == "FP32":
        # The conversion writes directly into the materialized FP32 output.
        scratch_per_element = 0
    scratch_per_element += 4 * (op.operation == "swiglu")
    scratch_per_element += 4 * (output.dtype == "INT8" and op.operation not in {"ffn_output", "storage_boundary"})
    stored_per_element = sum(t.nbytes // t.numel for t in op.inputs) + output.nbytes // output.numel
    return make_workspace_path(arch, spec, execution, mode, live_bytes, capacity, allocation, tile_elements,
                               elements=output.numel, scratch_per_element=scratch_per_element,
                               stored_per_element=stored_per_element)


def make_workspace_path(arch, spec, execution, mode, live_bytes, capacity, allocation, tile_elements, *,
                        elements, scratch_per_element, stored_per_element, local_fixed_bytes=0):
    """Shared architectural services with caller-declared, disjoint tile slots."""
    timed = mode == "timing"
    resident = allocation is not None
    native = spec.native_reference_spec if arch == "tsp_cim" else spec
    if resident:
        local = None
        # Scratch is one byte-striped pool, with disjoint logical subranges.
        available = capacity - live_bytes
        per_element = scratch_per_element
        max_tile = available // per_element if per_element else elements
        resources = resident_services(native, allocation, timed)
        clock = None if native.clock_freq_ghz is None else Fraction(str(native.clock_freq_ghz)) * 10**9
        compute, budget = native.vxm_service_cycles, native.values["vxm_base_ops_per_cycle"]
        provenance = deepcopy(native.parameters)
        services = {"mem_read_bpc": resources.read_bpc, "mem_write_bpc": resources.write_bpc,
                    "stream_bpc_per_direction": resources.stream_bpc}
        required = ["mem_read_bpc", "mem_write_bpc", "stream_width_bytes", "streams_per_direction", "vxm_base_ops_per_cycle"]
    else:
        local = "ub" if arch == "ascend" else "vmem"
        available = spec.capacities["UB"] if arch == "ascend" else int(positive(spec.values["vmem_usable_bytes"], "vmem_usable_bytes"))
        per_element = scratch_per_element + stored_per_element
        max_tile = (available - local_fixed_bytes) // per_element
        resources = ascend_services(spec, execution, timed) if arch == "ascend" else tpu_services(spec, timed)
        clock = (None if spec.clock_ghz is None else Fraction(str(spec.clock_ghz)) * 10**9) if arch == "ascend" else spec.values["clock_hz"]
        compute = spec.vector_service_cycles
        budget = spec.vector_parameters["base_ops_per_cycle"] if arch == "ascend" else spec.values["vector_base_ops_per_cycle"]
        provenance = spec.provenance()
        services = {"hbm_bpc": resources.hbm_bpc, "local_read_bpc": resources.read_bpc,
                    "local_write_bpc": resources.write_bpc}
        required = (["clock_freq_ghz", "chip.hbm_bandwidth_bytes_per_second", "chip.architectural_cores",
                     "ascend_ub_read_bpc", "ascend_ub_write_bpc", "vector.base_ops_per_cycle"] if arch == "ascend" else
                    ["clock_hz", "hbm_bandwidth_bytes_per_second", "tensorcores_per_chip",
                     "vmem_bandwidth_bytes_per_second", "vector_base_ops_per_cycle"])
    tile, _ = tile_groups(elements, tile_elements, execution, max_tile)
    if resident:
        slices = len(allocation.slices)
        scratch = ((tile * per_element + slices - 1) // slices) * slices
        memory = {"scope": "single_hem", "capacity_bytes": capacity, "resident_bytes": live_bytes + scratch,
                  "scratch_tile_bytes": scratch, "row_state_bytes": 0,
                  "mem_slices": list(allocation.slices), "streams_per_direction": allocation.streams}
        boundary = {"start": "tensors_in_local_MEM", "end": "output_in_local_MEM",
                    "selected_hemisphere": allocation.hemisphere, "offchip_supply_included": False,
                    "compute_resource": "one_shared_vxm_service"}
    else:
        memory = {"hbm": {"scope": "chip", "capacity_bytes": capacity, "peak_bytes": live_bytes},
                  local: {"scope": "core" if arch == "ascend" else "tensorcore", "capacity_bytes": available,
                          "allocated_bytes": tile * per_element + local_fixed_bytes, "scratch_tile_bytes": tile * scratch_per_element}}
        boundary = {"start": "tensors_in_HBM", "end": "output_in_HBM", "cache_policy": "bypass_for_elementwise"}
    kwargs = dict(spec=native, timed=timed, tile=tile,
                  requested_tile=execution.policy["tile_elements"] if tile_elements is None else tile_elements,
                  clock=clock, startup=execution.policy["transfer_startup_cycles"], compute=compute, budget=budget,
                  compute_prefix="vxm_" if resident else "vector." if arch == "ascend" else "vector_", memory=memory, boundary=boundary,
                  provenance=provenance, references={"hardware": getattr(spec, "reference_id", None)}, services=services, required=required,
                  read_bpc=resources.read_bpc, write_bpc=resources.write_bpc)
    if resident:
        return ResidentMemoryPath(**kwargs, width=native.stream_width_bytes, stream_bpc=resources.stream_bpc)
    return LocalMemoryPath(**kwargs, local=local, hbm_bpc=resources.hbm_bpc)


def simulate(op, path, execution, block_execution, mode):
    """Reuse architectural compute and memory services, with no extra edge traffic."""
    counts, operation_counts, boundary_counts, times = Counter(), Counter(), Counter(), Counter()
    groups, used = [], set()
    output = op.spec.outputs[0]
    _, tiles = tile_groups(output.numel, path.tile, execution)
    for n, repeats in tiles:
        stages = []

        def transfer(label, size, direction):
            actions, time = getattr(path, direction)(size)
            counts.update({k: v * repeats for k, v in actions.items()})
            times[direction] += time * repeats
            stages.append({"stage": label, "bytes": size, "cycles": time if path.timed else None})

        def step(label, reads, writes, operation, *, proxy=None):
            rc, rt = path.access(reads, "read")
            wc, wt = path.access(writes, "write")
            compute = 0
            cost_name = "fp32_basic_op_cost" if operation in {"fp32_add", "fp32_mul"} else operation + "_cost"
            used.add("block." + proxy if proxy else path.compute_prefix + cost_name)
            if path.timed:
                if proxy:
                    cost = positive(block_execution.values[proxy], proxy)
                    compute = cycles(n * cost, positive(path.budget, "vector_compute_budget"))
                else:
                    compute = path.compute({operation: n})
                compute += execution.policy["compute_startup_cycles"]
            counts.update({k: v * repeats for k, v in (rc + wc).items()})
            counts[operation] += n * repeats
            (boundary_counts if operation.startswith("boundary_") else operation_counts)[operation] += n * repeats
            for key, time in (("read", rt), ("compute", compute), ("write", wt)):
                times[key] += time * repeats
            stages.append({"stage": label, "read_bytes": reads, "write_bytes": writes,
                           "operation": operation, "elements": n,
                           "cycles": rt + compute + wt if path.timed else None})

        for i, tensor in enumerate(op.inputs):
            itemsize = tensor.nbytes // tensor.numel
            transfer(f"load_input_{i}", n * itemsize, "load")
            if tensor.dtype != "FP32":
                step(f"input_{i}_to_fp32", [n * itemsize], [4 * n],
                     "boundary_to_fp32", proxy="to_fp32_cost")
        if op.operation == "residual":
            step("residual_add", [4 * n, 4 * n], [4 * n], "fp32_add")
        elif op.operation == "attention_scale":
            step("scale_by_inverse_sqrt_head_dim", [4 * n], [4 * n], "fp32_mul")
        elif op.operation == "swiglu":
            step("negate_gate", [4 * n], [4 * n], "fp32_mul")
            step("exp", [4 * n], [4 * n], "fp32_exp")
            step("add_one", [4 * n], [4 * n], "fp32_add")
            step("reciprocal", [4 * n], [4 * n], "fp32_rcp", proxy="fp32_rcp_cost")
            step("silu", [4 * n, 4 * n], [4 * n], "fp32_mul")
            step("multiply_up", [4 * n, 4 * n], [4 * n], "fp32_mul")
        elif op.operation not in {"ffn_output", "storage_boundary"}:
            raise ValueError("Unsupported block elementwise kernel")
        if output.dtype == "INT8":
            step("output_to_int8", [4 * n], [n], "boundary_to_int8", proxy="to_int8_cost")
        transfer("export_output", n * output.nbytes // output.numel, "export")
        groups.append({"elements": n, "repetitions": repeats, "stages": stages})
    scope = execution.policy["scope"]
    metadata, actions = {}, {}
    component = "vxm" if isinstance(path, ResidentMemoryPath) else "vector"
    for key, count in counts.items():
        arithmetic = key in operation_counts or key in boundary_counts
        if arithmetic:
            category = "nonlinear" if key in {"fp32_exp", "fp32_rcp"} else "vector"
            meta = (component, key, category, "modeled_internal" if key in boundary_counts else "logical")
        else:
            meta = path.action_specs[key]
        c, name, category, basis = meta
        actions[key] = {"count": count, "unit": "element" if key in boundary_counts else "op" if arithmetic else "byte",
                        "scope": scope, "basis": basis}
        dtype = "FP32" if key in operation_counts else "mixed_storage_boundary" if key in boundary_counts else "mixed_storage"
        metadata[key] = ActionSpec(c, name, category, dtype)
    total = sum(times.values()) if path.timed else None
    timing = {"status": "analytical_estimate" if path.timed else "not_evaluated", "total_cycles": total,
              "time_ns": None if total is None or path.clock is None else float(Fraction(total) * 10**9 / Fraction(str(path.clock))),
              "required_parameters": sorted(set(path.required) | used), "model": "serial_block_tile_stages",
              "derived_services": {k: None if v is None else float(v) for k, v in path.services.items()}}
    attach_timing_breakdowns(timing, service=dict(times) if path.timed else None, elapsed=dict(times) if path.timed else None)
    report = {"schema_version": 1, "model": "block_elementwise", "mode": mode,
              "workload": op.spec.to_dict(), "boundary": {"scope": scope, **path.boundary},
              "execution_contract": {"scope": scope, "selected_tile_elements": path.tile, "requested_tile_elements": path.requested_tile},
              "action_counts": actions, "counts": dict(counts), "memory": deepcopy(path.memory), "timing": timing,
              "work_counts": {"logical": {"operations": sum(operation_counts.values()), "operation_counts": dict(operation_counts),
                                          "output_bytes": output.nbytes},
                              "issued": {"operations": None, "status": "not_modeled"},
                              "internal": {"boundary_elements": dict(boundary_counts)}},
              "schedule": {"tile_groups": groups, "scratch_policy": "Disjoint cast inputs, activation workspace and result; tile buffers reused only after export."},
              "parameter_provenance": path.provenance, "reference_ids": path.references}
    return NativeResult(report, metadata, "Serial materialized block stages; coarse storage-boundary work, not ISA instructions")
