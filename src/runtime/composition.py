"""Compose repeated child results without recounting native work or losing roles."""
from collections import Counter
from copy import deepcopy
from dataclasses import replace

from ..core.metrics import attach_timing_breakdowns
from ..operators.actions import ActionLedger


def collect_children(children, mode):
    """children: (ID, result, repetitions); repeat a whole serial template, not parallel lanes."""
    entries, actions, nodes, memory, elapsed, services = [], {}, [], {}, {}, {}
    logical, boundary = Counter(), Counter()
    total_ops = 0
    latency, complete_service = 0, True
    for node_id, result, repetitions in children:
        if type(repetitions) is not int or repetitions < 1:
            raise ValueError("Child repetitions must be positive integers")
        report = result.native_report
        for action in result.action_ledger.entries:
            key = node_id + "." + action.id
            entries.append(replace(action, id=key, count=None if action.count is None else action.count * repetitions,
                                   source=f"#/native_report/action_counts/{key}",
                                   attributes=tuple(("child." + k if k in {"parent_node", "repetitions"} or k.startswith("child.") else k, v)
                                                    for k, v in action.attributes) + (("parent_node", node_id), ("repetitions", str(repetitions))),
                                   summarizes=tuple(node_id + "." + name for name in action.summarizes)))
            actions[key] = deepcopy(report["action_counts"][action.id])
            actions[key]["count"] = entries[-1].count
        work = report["work_counts"]
        total_ops += work["logical"]["operations"] * repetitions
        logical.update({k: v * repetitions for k, v in work["logical"].get("operation_counts", {}).items()})
        boundary.update({k: v * repetitions for k, v in work["internal"].get("boundary_elements", {}).items()})
        for name, req in result.resource_requirements.items():
            key = name.lower()
            if key in {"mem", "hbm"}:
                continue
            if key not in memory or req["required_bytes"] > memory[key]["peak_bytes"]:
                memory[key] = {"scope": req["scope"], "capacity_bytes": req["capacity_bytes"], "peak_bytes": req["required_bytes"]}
        if mode == "timing":
            elapsed[node_id] = result.cycle_count * repetitions
            latency = None if latency is None or result.latency_s is None else latency + result.latency_s * repetitions
            by_stage = report["timing"]["service_cycles"]["by_stage"]
            if by_stage is None:
                complete_service = False
            else:
                services.update({node_id + "." + k: v * repetitions for k, v in by_stage.items()})
        nodes.append({"id": node_id, "repetitions": repetitions, "operator": result.operator.to_dict(),
                      "cycle_count_per_repetition": result.cycle_count, "native_report": report})
    total = sum(elapsed.values()) if mode == "timing" else None
    timing = {"status": "not_evaluated" if total is None else "cycles_only" if latency is None else "analytical_estimate",
              "total_cycles": total, "time_ns": None if total is None or latency is None else latency * 1e9,
              "model": "serial_repeated_child_templates"}
    attach_timing_breakdowns(timing, service=services if complete_service and mode == "timing" else None,
                            elapsed=elapsed if mode == "timing" else None)
    return {"ledger": ActionLedger(tuple(entries)), "action_counts": actions, "nodes": nodes, "memory": memory,
            "timing": timing, "logical_operations": total_ops, "operation_counts": dict(logical),
            "boundary_elements": dict(boundary)}
