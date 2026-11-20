"""Reference reporting: unknown values are never formatted as zero."""
import json
from pathlib import Path


def print_report(report):
    t = report["timing"]
    d = report["hardware_derived"]
    print("[TSP reference] " + report["reference_id"])
    print("[Scope] single hemisphere; chip inventory/peak reported separately")
    print(f"[Counts] useful_ops={report['counts']['useful_ops']} padded_ops={report['counts']['executed_padded_ops']} "
          f"weight_installs={report['counts']['weight_install_count']} PSUM_adds={report['counts']['vxm_add_ops']}")
    print(f"[Memory] {report['memory']['resident_bytes']} / {report['memory']['capacity_bytes']} local bytes")
    print(f"[Chip peak] geometric={d['geometric_int8_tops_chip']['value']} TOPS")
    print(f"[Timing] {t['status']}; single_hem_cycles={t['total_cycles']}; chip_cycles=not modeled")
    if t["missing_parameters"]:
        print("[Missing timing] " + ", ".join(t["missing_parameters"]))
    if report["mode"] == "timing":
        print("[Timing services] " + json.dumps(t["parameters"]))
        print("[Service cycles] " + json.dumps(t["service_cycles"]))
        print("[Elapsed cycles] " + json.dumps(t["elapsed_cycles"]) + f"; time_ns={t['time_ns']}")
        print("[Timing assumptions] " + "; ".join(t["assumptions"]))
    print("[Energy/area] not evaluated; action counts and inventory available")


def write_report(path, report):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
