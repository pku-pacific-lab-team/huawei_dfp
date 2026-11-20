"""Compact independent-operator output; successful reports only."""
import json
from pathlib import Path


def print_report(report):
    op = report["operator"]
    inputs = ", ".join(f"{t['name']}{t['shape']}:{t['dtype']}" for t in op["inputs"])
    print(f"[Operator] {report['architecture']} {op['kind']} ({inputs}) -> {op['outputs'][0]['dtype']}")
    print(f"[Time] {report['cycle_count']} cycles; {report['latency_s']} s ({report['timing_status']})")
    print(f"[Scope] {report['boundary']['scope']}; mode={report['mode']}")
    for name, memory in report["resource_requirements"].items():
        print(f"[Memory] {name}: {memory['required_bytes']} / {memory['capacity_bytes']} B ({memory['scope']})")
    for name, action in report["action_counts"].items():
        print(f"[Action] {name}: {action['count']} {action['unit']} ({action['basis']}, {action['role']})")


def write_report(report, path):
    # Serialize/validate before touching an existing report.
    content = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8")
