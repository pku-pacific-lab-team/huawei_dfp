"""Synthetic arithmetic fixtures, never measured Ascend specifications."""
import json
from pathlib import Path

from src.hardware.ascend import AscendSpec
from src.utils.workload import parse_workloads

ROOT = Path(__file__).resolve().parents[2]


def hardware_config(timed=True):
    config = json.loads((ROOT / "configs/hardware/ascend910_reference.json").read_text(encoding="utf-8"))
    if timed:
        config["cube"]["macs_per_cycle"].update(value=8192, status="assumed", source="Synthetic arithmetic test")
        config["chip"]["hbm_bandwidth_bytes_per_second"].update(value=512 * 32 * 10**9, status="assumed", source="Synthetic arithmetic test")
        config["chip"]["l2_bytes_per_cycle"].update(value=512 * 32, status="assumed", source="Synthetic arithmetic test")
        for route in config["transfers"].values():
            route.update(value=512, status="assumed", source="Synthetic arithmetic test")
    return config


def spec(timed=True):
    return AscendSpec(hardware_config(timed))


def workload(m=16, k=32, n=16):
    return parse_workloads({"gemm": {"M": m, "K": k, "N": n}, "data_format": "INT8"})[0]
