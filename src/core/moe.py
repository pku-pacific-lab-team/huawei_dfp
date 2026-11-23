"""Coarse MoE payload movement over existing memory services; no index ISA."""
from collections import Counter
from copy import deepcopy
from fractions import Fraction

from .composite_memory import ResidentMemoryPath
from .kv_cache import make_path
from .metrics import attach_timing_breakdowns
from .operator_result import ActionSpec, NativeResult
from .vector import tile_groups


def simulate_payload(op, path, execution, mode):
    counts, times, groups = Counter(), Counter(), []
    _, tiles = tile_groups(op.row_bytes, path.tile, execution)
    copying = op.target is not None
    for size, repeats in tiles:
        repeats *= op.rows
        if isinstance(path, ResidentMemoryPath):
            reads, rt = path.access([size], "read")
            writes, wt = path.access([size], "write") if copying else (Counter(), 0)
        else:
            reads, rt = path.load(size)
            writes, wt = path.export(size) if copying else (Counter(), 0)
        counts.update({key: count * repeats for key, count in (reads + writes).items()})
        times["read"] += rt * repeats
        if copying:
            times["write"] += wt * repeats
        groups.append({"bytes": size, "repetitions": repeats, "cycles_per_repetition": rt + wt if path.timed else None})
    actions, metadata = {}, {}
    scope = execution.policy["scope"]
    for key, count in counts.items():
        component, action, category, basis = path.action_specs[key]
        metadata[key] = ActionSpec(component, action, category, op.source.dtype)
        actions[key] = {"count": count, "unit": "byte", "scope": scope, "basis": basis}
    total = sum(times.values()) if path.timed else None
    timing = {"status": "not_evaluated" if total is None else "analytical_estimate", "total_cycles": total,
        "time_ns": None if total is None or path.clock is None else float(Fraction(total) * 10**9 / Fraction(str(path.clock))),
        "model": "serial_moe_payload_transfer", "derived_services": {k: None if v is None else float(v) for k, v in path.services.items()}}
    attach_timing_breakdowns(timing, service=dict(times) if path.timed else None, elapsed=dict(times) if path.timed else None)
    payload = op.row_bytes * op.rows
    report = {"schema_version": 1, "model": op.operation, "mode": mode, "workload": op.spec.to_dict(),
        "boundary": {"scope": scope}, "execution_contract": {"selected_tile_bytes": path.tile, "requested_tile_bytes": path.requested_tile},
        "memory": deepcopy(path.memory), "timing": timing, "action_counts": actions,
        "work_counts": {"logical": {"operations": 0, "output_bytes": payload if copying else 0,
            "read_payload_bytes": payload, "write_payload_bytes": payload if copying else 0},
            "issued": {"operations": None, "status": "not_applicable"}, "internal": {"status": "not_modeled"}},
        "schedule": {"tile_groups": groups, "policy": "Materialize selected rows without deduplication; byte-addressed payload services, no selection/index instructions."},
        "parameter_provenance": path.provenance,
        "limitations": ["Fixed assignments and offsets are caller metadata; no router/Top-K/index-generation cost is included.",
            "One indices read per supplied slot. Input rows and FP32 output rows move once per assignment.",
            "Resident transfers use streams; forwarding registers and physical gather/scatter transactions are not modeled."]}
    return NativeResult(report, metadata, "Coarse MoE row payload transfer over native memory services")
