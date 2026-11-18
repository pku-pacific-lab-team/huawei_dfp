"""Compact analytical report and JSON serialization."""
import json
from pathlib import Path


def print_report(report):
    print("[Workload] " + json.dumps(report["workload"], ensure_ascii=False))
    print("[Scope] Single Ascend core; chip specification peak is not simulated chip throughput")
    print("[Counts] " + json.dumps(report["counts"]))
    print("[Timing] " + json.dumps(report["timing"]))
    print("[Actions] " + json.dumps(report["action_counts"]))
    print("[Energy/area] Not evaluated; action counts and hardware inventory are provided")


def write_report(report, filename):
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
