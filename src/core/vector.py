"""Arithmetic/report helpers; architecture kernels own paths and timing."""
from collections import Counter
from copy import deepcopy
from fractions import Fraction
from math import ceil, isfinite

from .metrics import attach_timing_breakdowns
from .operator_result import ActionSpec, NativeResult


def positive(value, name):
    if type(value) not in (int, float) or not isfinite(value) or value <= 0:
        raise ValueError(f"Missing or invalid Vector service/capacity: {name}")
    return Fraction(str(value))


def tile_groups(total, requested, execution, capacity_elements=None):
    tile = execution.policy["tile_elements"] if requested is None else requested
    if type(tile) is not int or tile <= 0:
        raise ValueError("vector tile_elements must be a positive integer")
    if capacity_elements is not None:
        if capacity_elements < 1:
            raise ValueError("unsupported_capacity: local memory cannot hold one Vector element and its operands")
        tile = min(tile, capacity_elements)
    tile = min(total, tile)
    full, tail = divmod(total, tile)
    return tile, [(n, repeats) for n, repeats in ((tile, full), (tail, 1)) if n and repeats]


def cycles(size, rate, startup=0):
    return ceil(Fraction(size) / rate) + startup


def build_result(op, spec, execution, mode, memory, groups, counts, stages, action_specs,
                 clock_hz, *, boundary, services, required_parameters, reference_ids=None):
    """Collect native counts and action metadata; public adaptation is external."""
    arch, scope = execution.architecture, execution.policy["scope"]
    counts = dict(counts)
    counts.update(useful_ops=op.numel, logical_output_bytes=op.numel * 4,
                  vector_ops=op.numel if op.category == "vector" else 0,
                  nonlinear_ops=op.numel if op.category == "nonlinear" else 0)
    action_specs = {**action_specs, op.operation: ("vxm" if arch in {"tsp", "tsp_cim"} else "vector",
                                                op.operation, op.category, "op", "logical")}
    counts[op.operation] = op.numel
    actions, metadata = {}, {}
    for key, (component, name, category, unit, basis) in action_specs.items():
        actions[key] = {"count": counts[key], "unit": unit, "scope": scope, "basis": basis}
        metadata[key] = ActionSpec(component, name, category)
    total = sum(stages.values()) if mode == "timing" else None
    timing = {"status": "analytical_estimate" if total is not None else "not_evaluated",
              "total_cycles": total, "time_ns": None if total is None or clock_hz is None else float(Fraction(total) * 10**9 / Fraction(str(clock_hz))),
              "model": "vector_serial_tiles_and_stages_v1", "required_parameters": required_parameters,
              "derived_services": services}
    attach_timing_breakdowns(timing, service=dict(stages) if total is not None else None,
                            elapsed=dict(stages) if total is not None else None)
    provenance = deepcopy(spec.parameters) if scope == "single_hem" else spec.provenance()
    report = {
        "schema_version": 1, "model": f"{arch}_vector_serial", "mode": mode,
        "workload": {"operation": "vector", "op": op.operation, **op.spec.to_dict()},
        "boundary": {"scope": scope, "chip_execution_included": False, "numeric_values_simulated": False,
                     "in_place_included": False, **boundary},
        "execution_contract": {**deepcopy(execution.policy), "profile_reference_id": execution.reference_id,
                               "selected_tile_elements": max(n for n, _ in groups)},
        "counts": counts, "action_counts": actions,
        "work_counts": {"logical": {"operations": op.numel, "operation_counts": {op.operation: op.numel}, "output_bytes": op.numel * 4},
                        "issued": {"operations": None, "status": "not_modeled", "basis": "No SIMD instruction/padded arithmetic issue model"},
                        "internal": {"status": "not_modeled", "action_references": []}},
        "memory": memory, "timing": timing,
        "schedule": {"tile_count": sum(repeats for _, repeats in groups),
                     "tile_groups": [{"elements": n, "repetitions": r} for n, r in groups],
                     "policy": "All inputs read, then compute, then output write; no overlap across tiles"},
        "parameter_provenance": provenance, "vector_execution_profile": deepcopy(execution.config),
        "reference_ids": reference_ids or {},
        "energy": {"status": "not_evaluated", "total_pj": None},
        "limitations": ["Standalone analytical element-wise service; no numerical evaluation or domain checks on values.",
                        "EXP/RSQRT are algorithmic-work proxies. No broadcast, reduction, quantization, cache reuse or cross-op residency.",
                        "Only used services are required for timing. Counts do not require timing rates.",
                        "One shared compute budget; no replication by core, hemisphere or CIM domain inventory."]}
    return NativeResult(report, metadata,
                        "Logical tensor payload / element operations; stream groups are padded; no physical bank transactions")


def dma_actions(local):
    return {"hbm_read_bytes": ("hbm", "read", "offchip_memory", "byte", "logical"),
            "hbm_write_bytes": ("hbm", "write", "offchip_memory", "byte", "logical"),
            **{f"{local}_{phase}_{direction}_bytes": (local, direction, "onchip_memory", "byte", "logical")
               for phase, direction in (("input", "write"), ("operand", "read"), ("result", "write"), ("export", "read"))}}


def dma_counts(op, local):
    inputs, output = len(op.inputs) * op.numel * 4, op.numel * 4
    return Counter({"hbm_read_bytes": inputs, "hbm_write_bytes": output,
                    f"{local}_input_write_bytes": inputs, f"{local}_operand_read_bytes": inputs,
                    f"{local}_result_write_bytes": output, f"{local}_export_read_bytes": output})
