"""Compact analytical report and JSON serialization."""
import json
from pathlib import Path


def print_report(report):
    print("[Workload] " + json.dumps(report["workload"], ensure_ascii=False))
    print("[Scope] Single Ascend core; chip specification peak is not simulated chip throughput")
    print("[Counts] " + json.dumps(report["counts"]))
    print("[HBM] " + json.dumps(report["memory"]["hbm"]))
    print("[L2] " + json.dumps({**report["memory"]["l2"], **report["cache"]}))
    print("[Vector service; not invoked by GEMM] " + json.dumps(report["hardware_derived"]["vector_service"]))
    timing = dict(report["timing"])
    if "schedule" in timing:
        timing["schedule"] = {k: v for k, v in timing["schedule"].items()
                              if k not in {"events", "buffer_lifetimes"}}
    print("[Timing] " + json.dumps(timing))
    if "events" in report["timing"].get("schedule", {}):
        print("[Trace] Task events and buffer lifetimes are included in the JSON report")
    print("[Actions] " + json.dumps(report["action_counts"]))
    print("[Energy/area] Not evaluated; action counts and hardware inventory are provided")


def write_report(report, filename):
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
