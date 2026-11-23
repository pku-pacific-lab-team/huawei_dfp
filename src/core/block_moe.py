"""Explicit coarse router selection and streamed weighted accumulation services."""
from collections import Counter
from copy import deepcopy
from fractions import Fraction
from math import ceil, log2

from .block_elementwise import make_workspace_path
from .composite_memory import ResidentMemoryPath
from .metrics import attach_timing_breakdowns
from .operator_result import ActionSpec, NativeResult
from .vector import cycles, positive, tile_groups


class Schedule:
    def __init__(self, op, path, execution, block, mode):
        self.op, self.path, self.execution, self.block, self.mode = op, path, execution, block, mode
        self.counts, self.ops, self.boundaries, self.times = (Counter() for _ in range(4))
        self.stages, self.used = [], set()
        self.repeats = 1

    def dma(self, size, direction):
        counts, time = getattr(self.path, direction)(size)
        self.counts.update({k: v * self.repeats for k, v in counts.items()})
        self.times[direction] += time * self.repeats

    def step(self, label, reads, writes, operation=None, count=0, *, external=False, reduction_width=0):
        if external:
            for n in reads:
                self.dma(n, "load")
        rc, rt = self.path.access(reads, "read")
        wc, wt = self.path.access(writes, "write")
        self.counts.update({k: v * self.repeats for k, v in (rc + wc).items()})
        ct = 0
        if count:
            self.counts[operation] += count * self.repeats
            (self.boundaries if operation.startswith("boundary_") else self.ops)[operation] += count * self.repeats
            proxy = {"boundary_to_fp32": "to_fp32_cost", "boundary_to_int8": "to_int8_cost", "fp32_rcp": "fp32_rcp_cost"}.get(operation)
            key = "block." + proxy if proxy else "composite.fp32_max_cost" if operation == "fp32_max" else self.path.compute_prefix + ("fp32_exp_cost" if operation == "fp32_exp" else "fp32_basic_op_cost")
            self.used.add(key)
            if self.path.timed:
                if proxy or operation == "fp32_max":
                    cost = positive(self.block.values[proxy] if proxy else self.execution.composite_values["fp32_max_cost"], key)
                    ct = cycles(count * cost, positive(self.path.budget, "vector budget"))
                else:
                    ct = self.path.compute({operation: count})
                ct += self.execution.policy["compute_startup_cycles"]
        if reduction_width > 1:
            self.used.add("composite.reduction_round_cycles")
            if self.path.timed:
                # Coarse dependency floor; comparison/add work is charged once.
                floor = ceil(log2(reduction_width)) * positive(self.execution.composite_values["reduction_round_cycles"], "reduction_round_cycles")
                ct = max(ct, ceil(floor))
        for k, v in (("read", rt), ("compute", ct), ("write", wt)):
            self.times[k] += v * self.repeats
        if external:
            for n in writes:
                self.dma(n, "export")
        self.stages.append({"phase": label, "repetitions": self.repeats, "read_bytes": reads, "write_bytes": writes,
                            "operation": operation, "operations_per_repetition": count,
                            "reduction_width": reduction_width, "compute_cycles": ct if self.path.timed else None})

    def result(self, policy):
        path = self.path
        actions, metadata = {}, {}
        scope = self.execution.policy["scope"]
        component = "vxm" if isinstance(path, ResidentMemoryPath) else "vector"
        for key, count in self.counts.items():
            arithmetic = key in self.ops or key in self.boundaries
            if arithmetic:
                meta = (component, key, "nonlinear" if key in ("fp32_exp", "fp32_rcp") else "vector",
                        "modeled_internal" if key in self.boundaries else "logical")
            else:
                meta = path.action_specs[key]
            c, name, category, basis = meta
            actions[key] = {"count": count, "unit": "element" if key in self.boundaries else "op" if arithmetic else "byte", "scope": scope, "basis": basis}
            metadata[key] = ActionSpec(c, name, category, "FP32" if key in self.ops else "mixed_storage")
        total = sum(self.times.values()) if path.timed else None
        timing = {"status": "not_evaluated" if total is None else "analytical_estimate", "total_cycles": total,
                  "time_ns": None if total is None or path.clock is None else float(Fraction(total) * 10**9 / Fraction(str(path.clock))),
                  "model": "serial_moe_row_services", "required_parameters": sorted(set(path.required) | self.used),
                  "derived_services": {k: None if v is None else float(v) for k, v in path.services.items()}}
        attach_timing_breakdowns(timing, service=dict(self.times) if path.timed else None, elapsed=dict(self.times) if path.timed else None)
        report = {"schema_version": 1, "model": self.op.operation, "mode": self.mode, "workload": self.op.spec.to_dict(),
                  "boundary": {"scope": scope, **path.boundary}, "memory": deepcopy(path.memory), "timing": timing,
                  "execution_contract": {"selected_tile_elements": path.tile, "requested_tile_elements": path.requested_tile},
                  "action_counts": actions, "work_counts": {"logical": {"operations": sum(self.ops.values()),
                      "operation_counts": dict(self.ops), "output_bytes": sum(t.nbytes for t in self.op.spec.outputs)},
                      "issued": {"operations": None, "status": "not_modeled"}, "internal": {"boundary_elements": dict(self.boundaries)}},
                  "schedule": {"stages": self.stages, "policy": policy}, "parameter_provenance": path.provenance,
                  "limitations": ["Analytical FP32 services; no numerical values or ISA implementation.",
                      "Broadcasts, index addressing and winner exclusion are symbolic; no materialized broadcast or gather/scatter instructions."]}
        return NativeResult(report, metadata, policy)


def combine_path(op, arch, spec, execution, mode, live, capacity, allocation, tile):
    return make_workspace_path(arch, spec, execution, mode, live, capacity, allocation, tile,
        elements=op.experts.shape[-1], scratch_per_element=8,
        stored_per_element=4 + (1 if op.output_dtype == "INT8" else 4), local_fixed_bytes=4)


def combine(op, path, execution, block, mode):
    s = Schedule(op, path, execution, block, mode)
    b, k, h = op.experts.shape
    _, tiles = tile_groups(h, path.tile, execution)
    for n, repeats in tiles:
        s.repeats = b * repeats
        for slot in range(k):
            s.dma(4 * n, "load")
            s.dma(4, "load")
            s.step(f"slot_{slot}.weight_broadcast", [4*n, 4], [4*n], "fp32_mul", n)
            if slot:
                s.step(f"slot_{slot}.accumulate", [4*n, 4*n], [4*n], "fp32_add", n)
        if op.shared is not None:
            s.dma(4*n, "load")
            s.step("add_shared", [4*n, 4*n], [4*n], "fp32_add", n)
        if op.output_dtype == "INT8":
            s.step("output_to_int8", [4*n], [n], "boundary_to_int8", n)
        else:
            # Accumulator is scratch; publish into the distinct output backing.
            s.step("publish_fp32", [4*n], [4*n])
        s.dma(n * (1 if op.output_dtype == "INT8" else 4), "export")
    return s.result("Serial hidden tiles; two reusable FP32 buffers hold accumulator/product. First product initializes accumulator by alias. Weights read once per slot per hidden tile; final output is disjoint.")


def router_workspace(op, stripe):
    b, e = op.x.shape[0], op.weight.shape[1]
    # This deliberately materialized policy keeps all post-gate buffers until
    # selection finishes. No extra full [B,k,H] weight broadcast exists.
    sizes = {"original_scores": stripe(4*b*e), "normalizer": stripe(4*b)}
    if op.bias is not None:
        sizes["selection_scores"] = stripe(4*b*e)
    if op.groups > 1:
        sizes.update(group_scores=stripe(4*b*op.groups), group_indices=stripe(4*b*op.selected_groups),
                     group_winners=stripe(4*b*op.groups*(2 if op.bias is not None else 1)))
    return sizes


def router_path(op, arch, spec, execution, mode, live, capacity, allocation, tile):
    # Whole score rows fit a bounded local workspace; reject oversized rows
    # rather than silently omit their buffering. External buffers are in live.
    e, k = op.weight.shape[1], op.top_k
    return make_workspace_path(arch, spec, execution, mode, live, capacity, allocation, tile,
        elements=1, scratch_per_element=0, stored_per_element=12 * max(e, k, op.groups))


def router(op, path, execution, block, mode):
    s = Schedule(op, path, execution, block, mode)
    e, k, g, selected = op.weight.shape[1], op.top_k, op.groups, op.selected_groups
    s.repeats = op.x.shape[0]
    def step(label, reads, writes, operation=None, count=0, width=0):
        s.step(label, reads, writes, operation, count, external=True, reduction_width=width)
    step("logits_to_fp32", [4*e], [4*e], "boundary_to_fp32", e)
    for label, operation in (("negate", "fp32_mul"), ("exp", "fp32_exp"), ("add_one", "fp32_add"), ("reciprocal", "fp32_rcp")):
        step("sigmoid." + label, [4*e], [4*e], operation, e)
    if op.bias is not None:
        step("selection_only.add_bias", [4*e, 4*e], [4*e], "fp32_add", e)
    candidates = e
    if g > 1:
        width = e // g
        for group in range(g):
            winners = 2 if op.bias is not None else 1
            for j in range(winners):
                step(f"group_{group}.winner_{j}", [4*(width-j)], [4], "fp32_max", width-j-1, width-j)
            step(f"group_{group}.score", [4*winners], [4], "fp32_add" if winners == 2 else None, winners-1)
        for j in range(selected):
            step(f"select_group_{j}", [4*(g-j)], [4], "fp32_max", g-j-1, g-j)
        # Selected group IDs determine a symbolic filtered candidate view.
        step("read_selected_groups", [4*selected], [])
        candidates = selected * width
    for j in range(k):
        step(f"select_expert_{j}", [4*(candidates-j)], [4], "fp32_max", candidates-j-1, candidates-j)
    step("gather_original_sigmoid_scores", [4*k, 4*k], [4*k])
    step("sum_selected_original_scores", [4*k], [4], "fp32_add", k-1, k)
    step("reciprocal_selected_sum", [4], [4], "fp32_rcp", 1)
    step("normalize_selected_weights", [4*k, 4], [4*k], "fp32_mul", k)
    step("scale_selected_weights", [4*k], [4*k], "fp32_mul", k)
    result = s.result("Materialized score rows. Top-K uses repeated scan/reduction over remaining candidates: sum_j(N-j-1) MAX comparisons. Winner indices and exclusion masks are symbolic; dependency floor is ceil(log2(N))*reduction_round_cycles. Bias affects selection only; gather reads original sigmoid values.")
    result.report["selection_policy"] = {"kind": "repeated_max_without_replacement", "candidate_experts": candidates,
        "index_generation": "symbolic_no_integer_ISA_cost", "weights_from": "original_sigmoid_scores",
        "numeric_values_simulated": False, "group_score": "top2_sum" if op.bias is not None else "max"}
    return result
