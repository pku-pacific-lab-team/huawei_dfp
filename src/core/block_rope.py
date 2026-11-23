"""Pair-wise FP32 rotation with explicit INT8 boundaries and reusable tile slots."""
from collections import Counter
from copy import deepcopy
from fractions import Fraction

from .block_elementwise import make_workspace_path
from .composite_memory import ResidentMemoryPath
from .metrics import attach_timing_breakdowns
from .operator_result import ActionSpec, NativeResult
from .vector import cycles, positive, tile_groups


def make_path(op, arch, spec, execution, mode, live_bytes, capacity, allocation, tile_elements):
    requested = execution.policy["tile_elements"] if tile_elements is None else tile_elements
    if type(requested) is not int or requested < 2:
        raise ValueError("RoPE tile_elements must allow at least one complete pair (2 elements)")
    # Per pair: 8 B converted input + four 4 B products. The converted input
    # slots are reused for the two rotation results only after all four MULs.
    path = make_workspace_path(arch, spec, execution, mode, live_bytes, capacity, allocation, requested // 2,
                               elements=op.x.shape[-1] // 2, scratch_per_element=24, stored_per_element=12)
    path.requested_tile = requested
    return path


def simulate(op, path, execution, block_execution, mode):
    counts, arithmetic, boundaries, times = Counter(), Counter(), Counter(), Counter()
    groups, used = [], set()
    heads = op.x.shape[0] * op.x.shape[1]
    _, tiles = tile_groups(op.x.shape[-1] // 2, path.tile, execution)
    for pairs, repeats in tiles:
        repeats *= heads
        stages = []

        def transfer(buffer, size, direction):
            actions, time = getattr(path, direction)(size)
            counts.update({k: v * repeats for k, v in actions.items()})
            times[direction] += time * repeats
            stages.append({"stage": direction, "buffer": buffer, "bytes": size, "cycles": time if path.timed else None})

        def step(label, reads, writes, operation, elements, proxy=None):
            rc, rt = path.access([n for _, n in reads], "read")
            wc, wt = path.access([n for _, n in writes], "write")
            compute = 0
            if proxy:
                used.add("block." + proxy)
            elif operation == "fp32_sub":
                used.add("composite.fp32_sub_cost")
            else:
                used.add(path.compute_prefix + "fp32_basic_op_cost")
            if path.timed:
                if proxy or operation == "fp32_sub":
                    cost = block_execution.values[proxy] if proxy else execution.composite_values["fp32_sub_cost"]
                    compute = cycles(elements * positive(cost, proxy or "fp32_sub_cost"), positive(path.budget, "vector_compute_budget"))
                else:
                    compute = path.compute({operation: elements})
                compute += execution.policy["compute_startup_cycles"]
            counts.update({k: v * repeats for k, v in (rc + wc).items()})
            counts[operation] += elements * repeats
            (boundaries if proxy else arithmetic)[operation] += elements * repeats
            for key, value in (("read", rt), ("compute", compute), ("write", wt)):
                times[key] += value * repeats
            stages.append({"stage": label, "reads": [{"buffer": b, "bytes": n} for b, n in reads],
                           "writes": [{"buffer": b, "bytes": n} for b, n in writes], "operation": operation,
                           "elements": elements, "cycles": rt + compute + wt if path.timed else None})

        n = pairs
        transfer("input", 2 * n, "load")
        transfer("cos_row", 4 * n, "load")
        transfer("sin_row", 4 * n, "load")
        step("input_to_fp32", [("input", 2*n)], [("x_fp32", 8*n)], "boundary_to_fp32", 2*n, "to_fp32_cost")
        for i, (x, table) in enumerate((("even", "cos"), ("odd", "sin"), ("even", "sin"), ("odd", "cos"))):
            step("multiply_" + str(i), [("x_fp32." + x, 4*n), (table + "_row", 4*n)],
                 [("product_" + str(i), 4*n)], "fp32_mul", n)
        step("rotate_even", [("product_0", 4*n), ("product_1", 4*n)], [("x_fp32.even", 4*n)], "fp32_sub", n)
        step("rotate_odd", [("product_2", 4*n), ("product_3", 4*n)], [("x_fp32.odd", 4*n)], "fp32_add", n)
        step("output_to_int8", [("x_fp32", 8*n)], [("output", 2*n)], "boundary_to_int8", 2*n, "to_int8_cost")
        transfer("output", 2*n, "export")
        groups.append({"pairs": n, "repetitions": repeats, "stages": stages})
    scope = execution.policy["scope"]
    component = "vxm" if isinstance(path, ResidentMemoryPath) else "vector"
    metadata, actions = {}, {}
    for key, count in counts.items():
        compute = key in arithmetic or key in boundaries
        c, name, category, basis = (component, key, "vector", "modeled_internal" if key in boundaries else "logical") if compute else path.action_specs[key]
        dtype = "FP32" if key in arithmetic else "mixed_storage_boundary" if key in boundaries else "mixed_storage"
        metadata[key] = ActionSpec(c, name, category, dtype)
        actions[key] = {"count": count, "unit": "element" if key in boundaries else "op" if compute else "byte", "scope": scope, "basis": basis}
    total = sum(times.values()) if path.timed else None
    timing = {"status": "analytical_estimate" if path.timed else "not_evaluated", "total_cycles": total,
              "time_ns": None if total is None or path.clock is None else float(Fraction(total) * 10**9 / Fraction(str(path.clock))),
              "required_parameters": sorted(set(path.required) | used), "model": "serial_adjacent_pair_rotation",
              "derived_services": {k: None if v is None else float(v) for k, v in path.services.items()}}
    attach_timing_breakdowns(timing, service=dict(times) if path.timed else None, elapsed=dict(times) if path.timed else None)
    report = {"schema_version": 1, "model": "rope_apply", "mode": mode, "workload": op.spec.to_dict(),
              "boundary": {"scope": scope, **path.boundary, "table_generation_included": False},
              "execution_contract": {"scope": scope, "selected_pair_tile": path.tile,
                                     "selected_tile_elements": 2 * path.tile,
                                     "requested_pair_tile": path.requested_tile // 2,
                                     "requested_tile_elements": path.requested_tile,
                                     "table_row": op.position, "head_repetitions": heads},
              "action_counts": actions, "memory": deepcopy(path.memory), "timing": timing,
              "work_counts": {"logical": {"operations": sum(arithmetic.values()), "operation_counts": dict(arithmetic),
                                          "output_bytes": op.x.nbytes, "table_read_payload_bytes": 4 * op.x.numel},
                              "issued": {"operations": None, "status": "not_modeled"},
                              "internal": {"boundary_elements": dict(boundaries)}},
              "schedule": {"tile_groups": groups, "scratch_bytes_per_pair": 24,
                           "policy": "Adjacent channels form a pair. All four products precede overwriting the FP32 input slots with rotated results. Full cos/sin tables persist; one selected row is reread for each head; head/tile templates are compressed."},
              "parameter_provenance": path.provenance, "reference_ids": path.references}
    return NativeResult(report, metadata, "FP32 pair rotation and coarse INT8 boundaries; no trigonometric generation or layout instruction model")
