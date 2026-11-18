"""Console summary and machine-readable count/timing report."""
import json
from pathlib import Path


def print_report(report):
    print("[Workload] " + json.dumps(report["workload"], ensure_ascii=False))
    print("[Scope] Hardware: chip; counting: single TensorCore; chip execution: not included")
    print("[Hierarchy] " + json.dumps(report["hardware_hierarchy"]))
    print("[Counts] " + json.dumps(report["counts"]))
    print("[CMEM] " + json.dumps(report["memory"]["cmem"]))
    print("[Inventory per TensorCore] " + json.dumps(report["hardware_inventory"]["per_tensorcore"]))
    print("[Inventory chip] " + json.dumps(report["hardware_inventory"]["chip"]))
    print("[Missing parameters] " + (", ".join(report["missing_parameters"]) or "none"))
    print("[Assumptions] " + ", ".join(report["hardware_derived"]["assumed_inputs"]))
    if report["mode"] == "timing":
        timing = report["timing"]
        print("[Timing] " + json.dumps({k: timing[k] for k in
              ("status", "total_cycles", "time_ns", "throughput_gops", "service_cycles", "elapsed_cycles")}))
        if "cmem_outer_overlap_saved_cycles" in timing:
            print("[CMEM overlap] " + json.dumps({k: timing[k] for k in
                  ("serialized_cmem_outer_cycles", "cmem_outer_overlap_saved_cycles")}))
        print("[Timing assumptions] " + "; ".join(timing["assumptions"]))
        print("[Energy/area] Not evaluated; chip peak is specification arithmetic")
    else:
        print("[Timing/energy/area] Not evaluated; chip peak is specification arithmetic")


def write_report(report, filename):
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
