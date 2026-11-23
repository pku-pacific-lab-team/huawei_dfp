"""Compact row/pass schedules; transport remains architecture-specific.

Stages describe tensor work and tree dependencies, never machine instructions.
All rows, tiles, accesses and arithmetic stages are serialized in this policy.
"""
from collections import Counter
from copy import deepcopy
from fractions import Fraction
from math import ceil

from .metrics import attach_timing_breakdowns
from .composite_storage import validate_storage_schedule
from .composite_memory import MemoryPath
from .vector import positive
from .operator_result import ActionSpec, NativeResult


class RowSchedule:
    def __init__(self, op, path, execution, mode):
        self.op, self.path, self.execution = op, path, execution
        self.timed = mode == "timing"
        self.counts, self.times, self.operation_counts = Counter(), Counter(), Counter()
        self.groups, self.used = [], set()

    def group(self, phase, elements=1, repeats=1):
        self.current = {"phase": phase, "elements": elements, "repetitions_per_row": repeats, "stages": []}
        self.groups.append(self.current)
        self.multiplier = repeats * self.op.rows

    def record(self, name, counts, times, **details):
        self.counts.update({k: v * self.multiplier for k, v in counts.items()})
        if self.timed:
            self.times.update({k: v * self.multiplier for k, v in times.items()})
        self.current["stages"].append({"name": name, **details, "counts_per_execution": dict(counts),
                                       "cycles_per_execution": dict(times) if self.timed else None})

    def dma(self, name, elements, direction):
        counts, time = getattr(self.path, direction)(elements * 4)
        if counts:
            self.record(name, counts, {"dma": time}, direction=direction, logical_bytes=elements * 4)

    def step(self, name, reads, writes, operation=None, count=0, reduction=False):
        counts, read_time = self.path.access([n * 4 for _, n in reads], "read")
        written, write_time = self.path.access([n * 4 for _, n in writes], "write")
        counts.update(written)
        compute = dependency = 0
        if count:
            counts[operation] += count
            self.operation_counts[operation] += count * self.multiplier
            special = operation in {"fp32_sub", "fp32_max", "fp32_rcp"}
            cost = operation + "_cost" if special else {
                "fp32_add": "fp32_basic_op_cost", "fp32_mul": "fp32_basic_op_cost",
                "fp32_exp": "fp32_exp_cost", "fp32_rsqrt": "fp32_rsqrt_cost"}[operation]
            self.used.add("composite." + cost if special else self.path.compute_prefix + cost)
            self.used.add(self.path.compute_prefix + "base_ops_per_cycle")
            if reduction:
                self.used.add("composite.reduction_round_cycles")
            if self.timed:
                if special:
                    work = count * positive(self.execution.composite_values[cost], cost)
                    compute = ceil(work / positive(self.path.budget, "shared Vector budget"))
                else:
                    compute = self.path.compute({operation: count})
                if reduction:
                    lag = int(positive(self.execution.composite_values["reduction_round_cycles"], "reduction_round_cycles"))
                    dependency = max(0, lag - compute)
                compute += self.execution.policy["compute_startup_cycles"]
        self.record(name, counts, {"local_read": read_time, "compute": compute,
                    "reduction_dependency": dependency, "local_write": write_time},
                    reads=[{"buffer": b, "elements": n} for b, n in reads],
                    writes=[{"buffer": b, "elements": n} for b, n in writes],
                    operation_counts={operation: count} if count else {}, reduction_round=reduction)

    def reduce(self, n, operation, source):
        # Indexed surviving values avoid any implicit full-array reshuffle.
        remaining, level = n, 0
        while remaining > 1:
            pairs = remaining // 2
            self.step(f"reduce_{operation}_level_{level}", [(source if level == 0 else "tree_survivors", 2 * pairs)],
                      [("tile_scratch", pairs)], operation, pairs, reduction=True)
            remaining = (remaining + 1) // 2
            level += 1
        return source if n == 1 else "tree_partial"

    def merge(self, partial, state, operation, first):
        self.step("initialize_" + state if first else "merge_" + state,
                  [(partial, 1)] if first else [(partial, 1), (state, 1)], [(state, 1)],
                  None if first else operation, 0 if first else 1, reduction=not first)


def chunks(width, tile, separate_first=False):
    full, tail = divmod(width, tile)
    groups = [(tile, full)]
    if separate_first:
        groups = [(tile, 1), (tile, full - 1)]
    if tail:
        groups.append((tail, 1))
    return [(n, r) for n, r in groups if r]


def rmsnorm(s):
    op, tile = s.op, s.path.tile
    for i, (n, repeats) in enumerate(chunks(op.width, tile, True)):
        s.group("sum_squares", n, repeats)
        s.dma("load_x", n, "load")
        s.step("square", [("x", n)], [("tile_scratch", n)], "fp32_mul", n)
        partial = s.reduce(n, "fp32_add", "tile_scratch")
        s.merge(partial, "row_sum", "fp32_add", first=i == 0)
    s.group("normalizer")
    s.step("mean_scale_by_constant_1_over_width", [("row_sum", 1)], [("row_scale", 1)], "fp32_mul", 1)
    s.step("add_epsilon", [("row_scale", 1)], [("row_scale", 1)], "fp32_add", 1)
    s.step("rsqrt", [("row_scale", 1)], [("row_scale", 1)], "fp32_rsqrt", 1)
    for n, repeats in chunks(op.width, tile):
        s.group("apply_scale_and_weight", n, repeats)
        s.dma("reload_x", n, "load")
        s.dma("load_weight", n, "load")
        s.step("broadcast_scale", [("x", n), ("row_scale", 1)], [("tile_scratch", n)], "fp32_mul", n)
        s.step("multiply_weight", [("tile_scratch", n), ("weight", n)], [("output", n)], "fp32_mul", n)
        s.dma("export_output", n, "export")


def softmax(s):
    op, tile = s.op, s.path.tile
    for i, (n, repeats) in enumerate(chunks(op.width, tile, True)):
        s.group("row_max", n, repeats)
        s.dma("load_x", n, "load")
        partial = s.reduce(n, "fp32_max", "x")
        s.merge(partial, "row_max", "fp32_max", first=i == 0)
    for i, (n, repeats) in enumerate(chunks(op.width, tile, True)):
        s.group("exponentials_and_sum", n, repeats)
        s.dma("reload_x", n, "load")
        s.step("subtract_max", [("x", n), ("row_max", 1)], [("tile_scratch", n)], "fp32_sub", n)
        s.step("exp_to_output_workspace", [("tile_scratch", n)], [("output_exp", n)], "fp32_exp", n)
        s.dma("export_exp_workspace", n, "export")
        partial = s.reduce(n, "fp32_add", "output_exp")
        s.merge(partial, "row_sum", "fp32_add", first=i == 0)
    s.group("reciprocal")
    s.step("reciprocal_sum", [("row_sum", 1)], [("row_reciprocal", 1)], "fp32_rcp", 1)
    for n, repeats in chunks(op.width, tile):
        s.group("normalize", n, repeats)
        s.dma("load_exp_workspace", n, "load")
        s.step("broadcast_reciprocal", [("output_exp", n), ("row_reciprocal", 1)], [("output", n)], "fp32_mul", n)
        s.dma("export_output", n, "export")


def simulate(op, path: MemoryPath, execution, mode):
    if not isinstance(path, MemoryPath):
        raise ValueError("Expected an explicit composite MemoryPath")
    s = RowSchedule(op, path, execution, mode)
    (rmsnorm if op.operation == "rmsnorm" else softmax)(s)
    storage_validation = validate_storage_schedule(op, path, s.groups)
    arch, scope = execution.architecture, execution.policy["scope"]
    action_specs = dict(path.action_specs)
    component = "vxm" if arch in {"tsp", "tsp_cim"} else "vector"
    for name in s.operation_counts:
        action_specs[name] = (component, name, "nonlinear" if name in {"fp32_exp", "fp32_rsqrt", "fp32_rcp"} else "vector", "logical")
    metadata, native_actions = {}, {}
    for key, count in s.counts.items():
        component, name, category, basis = action_specs[key]
        unit = "op" if key in s.operation_counts else "byte"
        native_actions[key] = {"count": count, "unit": unit, "scope": scope, "basis": basis}
        metadata[key] = ActionSpec(component, name, category)
    total = sum(s.times.values()) if s.timed else None
    timing = {"status": "analytical_estimate" if s.timed else "not_evaluated", "total_cycles": total,
              "time_ns": None if total is None or path.clock is None else float(Fraction(total) * 10**9 / Fraction(str(path.clock))),
              "model": "normalization_serial_rows_passes_tiles_v1", "required_parameters": sorted(set(path.required) | s.used),
              "derived_services": {k: None if v is None else float(v) for k, v in path.services.items()}}
    attach_timing_breakdowns(timing, service=dict(s.times) if s.timed else None, elapsed=dict(s.times) if s.timed else None)
    workload = {"operation": op.operation, **op.spec.to_dict(), "axis": len(op.x.shape) - 1, "rows": op.rows, "width": op.width}
    if op.operation == "rmsnorm":
        workload["epsilon"] = op.epsilon
    memory = deepcopy(path.memory)
    memory["lifetime_validation"] = storage_validation
    memory["lifetimes"] = (
        ["One row state aliases sum, mean and scale; retained through output pass.",
         "Tile scratch is reused after each reduction; input is reread for scale/weight pass.",
         "Weight is stored once externally but read once per row; no cross-row local reuse."] if op.operation == "rmsnorm" else
        ["Row max and sum occupy two scalar slots during exp/sum pass; reciprocal reuses sum slot.",
         "Output stores exponentials before normalization; no separate full-size exp tensor.",
         "Tile scratch holds reduction partials and shifted inputs; indexed odd survivors remain live until consumed."])
    counts = dict(s.counts)
    counts.update(useful_ops=sum(s.operation_counts.values()), logical_output_bytes=op.x.nbytes)
    report = {"schema_version": 1, "model": f"{arch}_normalization_serial", "mode": mode,
              "workload": workload, "boundary": {"scope": scope, "chip_execution_included": False,
                  "numeric_values_simulated": False, "input_output_alias_included": False, **path.boundary},
              "execution_contract": {**{k: execution.policy[k] for k in ("scope", "path", "tile_elements", "transfer_startup_cycles", "compute_startup_cycles")},
                  "schedule": execution.composite["schedule"], "composite": deepcopy(execution.composite),
                  "profile_reference_id": execution.reference_id, "requested_tile_elements": path.requested_tile,
                  "selected_tile_elements": path.tile, "row_schedule": "serial_rows_passes_tiles"},
              "counts": counts, "action_counts": native_actions,
              "work_counts": {"logical": {"operations": sum(s.operation_counts.values()), "operation_counts": dict(s.operation_counts),
                                          "output_bytes": op.x.nbytes},
                              "issued": {"operations": None, "status": "not_modeled", "basis": "No instruction issue model"},
                              "internal": {"status": "not_modeled", "action_references": []}},
              "memory": memory, "timing": timing,
              "schedule": {"rows": op.rows, "tiles_per_row": ceil(Fraction(op.width, path.tile)),
                           "groups_per_row": s.groups,
                           "policy": "Repeat the entire group sequence per row; repeat each group's entire stage sequence per tile. Serial; each tree level and cross-tile merge waits for predecessors."},
              "parameter_provenance": path.provenance, "vector_execution_profile": deepcopy(execution.config),
              "reference_ids": path.references, "energy": {"status": "not_evaluated", "total_pj": None},
              "limitations": ["Last-axis FP32 analytical blocks; no numeric values, masking, quantization or machine instructions.",
                              "SUB/MAX/RCP and reduction round dependency are uncalibrated policy assumptions sharing the native compute budget.",
                              "Scalar constants are implicit; broadcasts read one stored row scalar per tile with no materialized broadcast tensor or extra shuffle cost.",
                              "No cross-row reuse, overlap, cache, bank-conflict or cross-operator residency model."]}
    return NativeResult(report, metadata,
                        "Logical tensor work and accesses; stream payload rounded per access; no physical transaction claim")
