"""Common semantic layers; these are views, never additional energy charges."""
from copy import deepcopy


def attach_timing_breakdowns(timing, *, service=None, elapsed=None):
    """Separate additive service work from a disjoint elapsed-time partition.

    None means the model has no such decomposition, never zero work. In
    particular, a compressed row window is not a sum of component services.
    """
    total = timing["total_cycles"]
    for name, stages in (("service_cycles", service), ("elapsed_cycles", elapsed)):
        if stages is not None:
            if total is None or not stages or any(type(v) is not int or v < 0 for v in stages.values()):
                raise ValueError(name + ": expected evaluated nonnegative integer stages")
            if name == "elapsed_cycles" and sum(stages.values()) != total:
                raise ValueError("elapsed_cycles must partition total_cycles")
        timing[name] = {
            "status": "not_evaluated" if total is None else "not_decomposed" if stages is None else "available",
            "by_stage": deepcopy(stages),
            "total_cycles": (sum(stages.values()) if stages is not None else None)
                            if name == "service_cycles" else total,
            "semantics": ("Sum of modeled stage service work; stages may overlap. Not elapsed time or a sum of resource busy cycles."
                          if name == "service_cycles" else
                          "Disjoint elapsed-time partition; available stages sum to timing.total_cycles. No overlap is charged twice."),
        }
    if total is not None and service is None:
        timing["service_cycles"]["reason"] = "Compressed row windows and load intervals do not provide separate component service work; load intervals may include waiting."
    if total is not None and elapsed is None:
        timing["elapsed_cycles"]["reason"] = "The pipeline computes total duration but does not attribute overlapping intervals to disjoint stages."
    return timing


def attach_contract(report, execution):
    if report["timing"]["total_cycles"] is None:
        attach_timing_breakdowns(report["timing"])
    counts = report["counts"]
    internal = [name for name, action in report["action_counts"].items()
                if action.get("scope") == "sum_of_macro_actions"]
    report["execution_contract"] = deepcopy(execution)
    report["work_counts"] = {
        "logical": {"operations": counts["useful_ops"], "output_bytes": counts["logical_output_bytes"]},
        "issued": {"operations": counts["executed_padded_ops"],
                   "basis": "padded arithmetic; 2 operations per MAC; excludes inactive unused arrays",
                   "status": "derived" if counts["executed_padded_ops"] is not None else "unavailable"},
        "internal": {"status": "partial_macro_actions" if internal else "not_modeled",
                     "action_references": internal},
        "semantics": "Views of existing counts, not additional additive actions. Logical payload is not a physical memory transaction. Unknown internal activity is not zero.",
    }
    return report
