"""Serial symbolic row lookup; no matrix arithmetic or gather instruction timing."""
from collections import Counter
from copy import deepcopy
from fractions import Fraction

from .block_elementwise import make_workspace_path
from .composite_memory import ResidentMemoryPath
from .metrics import attach_timing_breakdowns
from .operator_result import ActionSpec, NativeResult
from .vector import tile_groups


def make_path(op, arch, spec, execution, mode, live_bytes, capacity, allocation, tile_elements):
    # Local storage retains one 4 B index and a reusable INT8 row-copy tile.
    path = make_workspace_path(arch, spec, execution, mode, live_bytes, capacity, allocation, tile_elements,
                               elements=op.weight.shape[1], scratch_per_element=0, stored_per_element=1, local_fixed_bytes=4)
    if not isinstance(path, ResidentMemoryPath):
        path.memory[path.local]["index_state_bytes"] = 4
    return path


def simulate(op, path, execution, mode):
    counts, times, groups = Counter(), Counter(), []
    batch, hidden = op.indices.shape[0], op.weight.shape[1]
    resident = isinstance(path, ResidentMemoryPath)
    index_actions, index_time = path.access([4], "read")
    if not resident:
        loaded, elapsed = path.load(4)
        index_actions.update(loaded)
        index_time += elapsed
    counts.update({"index." + k: v * batch for k, v in index_actions.items()})
    times["index_read"] = index_time * batch
    groups.append({"phase": "index_read", "bytes": 4, "repetitions": batch, "cycles_per_repetition": index_time if path.timed else None})
    _, tiles = tile_groups(hidden, path.tile, execution)
    for size, repeats in tiles:
        repeats *= batch
        rc, rt = path.access([size], "read") if resident else path.load(size)
        wc, wt = path.access([size], "write") if resident else path.export(size)
        counts.update({"row." + k: v * repeats for k, v in (rc + wc).items()})
        times["row_read"] += rt * repeats
        times["output_write"] += wt * repeats
        groups.append({"phase": "row_copy", "bytes": size, "repetitions": repeats, "cycles_per_repetition": rt + wt if path.timed else None})
    scope = execution.policy["scope"]
    metadata, actions = {}, {}
    for key, count in counts.items():
        component, name, category, basis = path.action_specs[key.split(".", 1)[1]]
        metadata[key] = ActionSpec(component, name, category, "INT32" if key.startswith("index.") else "INT8")
        actions[key] = {"count": count, "unit": "byte", "scope": scope, "basis": basis}
    total = sum(times.values()) if path.timed else None
    timing = {"status": "analytical_estimate" if path.timed else "not_evaluated", "total_cycles": total,
              "time_ns": None if total is None or path.clock is None else float(Fraction(total) * 10**9 / Fraction(str(path.clock))),
              "model": "serial_index_read_and_row_copy",
              "required_parameters": [p for p in path.required if "base_ops_per_cycle" not in p],
              "derived_services": {k: None if v is None else float(v) for k, v in path.services.items()}}
    attach_timing_breakdowns(timing, service=dict(times) if path.timed else None, elapsed=dict(times) if path.timed else None)
    report = {"schema_version": 1, "model": "embedding_lookup", "mode": mode, "workload": op.spec.to_dict(),
              "boundary": {"scope": scope, **path.boundary, "index_values_validated": False},
              "execution_contract": {"scope": scope, "selected_tile_elements": path.tile,
                                     "requested_tile_elements": path.requested_tile, "duplicate_row_reuse": False},
              "memory": deepcopy(path.memory), "timing": timing, "action_counts": actions,
              "work_counts": {"logical": {"operations": 0, "operation_counts": {}, "output_bytes": batch * hidden,
                                          "index_read_bytes": 4 * batch, "table_read_bytes": batch * hidden,
                                          "table_capacity_bytes": op.weight.nbytes},
                              "issued": {"operations": None, "status": "not_applicable"},
                              "internal": {"boundary_elements": {}}},
              "schedule": {"tile_groups": groups,
                           "policy": "For each token, read its index once, then copy one row in tiles; repeated rows are reread. Groups compress repetitions, not cross-token execution overlap. Full weights and indices persist; output is separate."},
              "parameter_provenance": path.provenance, "reference_ids": path.references,
              "limitations": ["Index values are symbolic and assumed in range; no numeric token lookup is executed.",
                              "No address-generation, gather instruction or random-access penalty model. Transient address registers are outside capacity accounting.",
                              "HBM copies reuse one local buffer; MEM copies use directional stream service. No weight prefetch, duplicate-row cache or offchip paging."]}
    return NativeResult(report, metadata, "INT32 index reads and INT8 selected-row copies; full table residency is not full-table traffic")
