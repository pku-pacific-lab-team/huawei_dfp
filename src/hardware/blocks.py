"""Explicit coarse block assumptions, independent of native hardware calibration."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .parameters import read_parameter


class BlockExecution:
    def __init__(self, config):
        fields = {"schema_version", "profile_id", "schedule", "storage", "layout", "status", "source", "conditions", "parameters"}
        if not isinstance(config, dict) or set(config) != fields or type(config["schema_version"]) is not int or config["schema_version"] != 1:
            raise ValueError("Expected block execution profile schema 1")
        if (config["schedule"], config["storage"], config["layout"], config["status"]) != (
                "serial_materialized_nodes", "persistent_inputs_reused_intermediates", "symbolic_padded_views", "assumed"):
            raise ValueError("Unsupported block execution policy")
        for key in ("profile_id", "source", "conditions"):
            if not isinstance(config[key], str) or not config[key].strip():
                raise ValueError(f"Block profile requires {key}")
        costs = {"to_fp32_cost", "to_int8_cost", "fp32_rcp_cost"}
        if not isinstance(config["parameters"], dict) or set(config["parameters"]) != costs:
            raise ValueError("Block profile requires explicit FP32/INT8 boundary and reciprocal costs")
        self.values = {k: read_parameter(v, k, "equivalent_op/element", "task", required=False)
                       for k, v in config["parameters"].items()}
        self.config = deepcopy(config)
        self.reference_id = config["profile_id"] + ":" + hashlib.sha256(
            json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


def load_block_execution(filename=None, config_dir=None):
    base = Path(config_dir) if config_dir is not None else Path(__file__).resolve().parents[2] / "configs"
    path = Path(filename) if filename is not None else base / "execution/blocks.json"
    if filename is not None and not path.is_absolute() and not path.is_file():
        path = base / "execution" / path
    return BlockExecution(json.loads(path.read_text(encoding="utf-8")))
