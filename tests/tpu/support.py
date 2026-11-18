import json
from pathlib import Path

from src.hardware.tpu import TPUSpec
from src.utils.workload import parse_workloads

ROOT = Path(__file__).resolve().parents[2]


def hardware_config():
    return json.loads((ROOT / "configs/hardware/tpu_v4_reference.json").read_text(encoding="utf-8"))


def spec():
    return TPUSpec(hardware_config())


def workload(m=16, k=32, n=16):
    return parse_workloads({"gemm": {"M": m, "K": k, "N": n}, "data_format": "INT8"})[0]
