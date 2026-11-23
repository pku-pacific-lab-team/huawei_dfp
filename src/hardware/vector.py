"""Versioned Vector execution assumptions, separate from GEMM policies."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .parameters import read_parameter


ROOT = Path(__file__).resolve().parents[2]
SCOPES = {"ascend": "single_core", "tpu": "single_tensorcore", "tsp": "single_hem"}
PATHS = {"ascend": "hbm_ub_bypass_l2", "tpu": "hbm_vmem_bypass_cmem", "tsp": "resident_mem_stream_vxm"}


class VectorExecution:
    def __init__(self, config, architecture):
        if architecture not in {"ascend", "tpu", "tsp", "tsp_cim"}:
            raise ValueError("Unsupported Vector architecture")
        if not isinstance(config, dict) or type(config.get("schema_version")) is not int or config["schema_version"] not in (1, 2):
            raise ValueError("Expected Vector execution schema 1 or 2")
        expected = {"schema_version", "model", "profile_id", "policies", "parameters"}
        if config["schema_version"] == 2:
            expected.add("composite")
        if not isinstance(config, dict) or set(config) != expected:
            raise ValueError("Vector execution profile fields mismatch")
        if config["model"] != "vector_serial":
            raise ValueError("Expected model vector_serial")
        if not isinstance(config["profile_id"], str) or not config["profile_id"].strip():
            raise ValueError("Vector profile_id is required")
        if not isinstance(config["policies"], dict) or set(config["policies"]) != set(SCOPES):
            raise ValueError("Expected Ascend, TPU and shared TSP Vector policies")
        for arch, policy in config["policies"].items():
            fields = {"scope", "path", "schedule", "tile_elements", "transfer_startup_cycles",
                      "compute_startup_cycles", "status", "source", "version", "conditions"}
            if not isinstance(policy, dict) or set(policy) != fields:
                raise ValueError("Vector policy fields mismatch")
            if (policy["scope"], policy["path"], policy["schedule"], policy["status"]) != (SCOPES[arch], PATHS[arch], "serial_tiles_and_stages", "assumed"):
                raise ValueError("Unsupported Vector execution policy")
            for field in ("tile_elements", "transfer_startup_cycles", "compute_startup_cycles"):
                if type(policy[field]) is not int or policy[field] < (1 if field == "tile_elements" else 0):
                    raise ValueError(f"Invalid Vector {field}")
            for field in ("source", "version", "conditions"):
                if not isinstance(policy[field], str) or not policy[field].strip():
                    raise ValueError(f"Vector policy requires {field}")
        params = config["parameters"]
        if not isinstance(params, dict) or set(params) != {"ascend_ub_read_bpc", "ascend_ub_write_bpc"}:
            raise ValueError("Vector profile requires explicit Ascend UB services")
        self.values = {name: read_parameter(raw, name, "byte/cycle", "core", required=False)
                       for name, raw in params.items()}
        self.composite = deepcopy(config.get("composite"))
        self.composite_values = {}
        if config["schema_version"] == 2:
            policy = self.composite
            fields = {"schedule", "reduction", "storage", "status", "source", "version", "conditions", "parameters"}
            if not isinstance(policy, dict) or set(policy) != fields:
                raise ValueError("Composite policy fields mismatch")
            if (policy["schedule"], policy["reduction"], policy["storage"], policy["status"]) != (
                    "serial_rows_passes_tiles", "balanced_tile_tree_serial_merge",
                    "rmsnorm_reread_softmax_output_workspace", "assumed"):
                raise ValueError("Unsupported composite execution policy")
            for field in ("source", "version", "conditions"):
                if not isinstance(policy[field], str) or not policy[field].strip():
                    raise ValueError(f"Composite policy requires {field}")
            costs = {"fp32_sub_cost", "fp32_max_cost", "fp32_rcp_cost", "reduction_round_cycles"}
            if not isinstance(policy["parameters"], dict) or set(policy["parameters"]) != costs:
                raise ValueError("Composite costs require SUB, MAX, RCP and reduction dependency latency")
            for name, raw in policy["parameters"].items():
                self.composite_values[name] = read_parameter(raw, name,
                    "cycle" if name == "reduction_round_cycles" else "equivalent_op/op", "task",
                    integer=name == "reduction_round_cycles", required=False)
        self.architecture = architecture
        self.config = deepcopy(config)
        self.policy = deepcopy(config["policies"]["tsp" if architecture == "tsp_cim" else architecture])
        self.reference_id = config["profile_id"] + ":" + hashlib.sha256(
            json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


def load_vector_execution(architecture, filename=None, config_dir=None):
    base = Path(config_dir) if config_dir is not None else ROOT / "configs"
    path = Path(filename) if filename is not None else base / "execution" / "vector.json"
    if filename is not None and not path.is_absolute() and not path.is_file():
        path = base / "execution" / path
    return VectorExecution(json.loads(path.read_text(encoding="utf-8")), architecture)
