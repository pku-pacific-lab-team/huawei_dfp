"""Versioned Decode scheduling assumptions, separate from hardware calibration."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path


class AttentionExecution:
    def __init__(self, config):
        fixed = {"schema_version": 1, "schedule": "serial_query_heads", "cache": "preallocated_append_only",
                 "reuse": "no_cross_query_head_reuse", "layout": "symbolic_padded_head_views", "status": "assumed"}
        if not isinstance(config, dict) or set(config) != set(fixed) | {"profile_id", "source", "conditions"}:
            raise ValueError("Expected attention execution profile schema 1")
        if type(config["schema_version"]) is not int or any(config[k] != v for k, v in fixed.items()):
            raise ValueError("Unsupported attention execution policy")
        for key in ("profile_id", "source", "conditions"):
            if not isinstance(config[key], str) or not config[key].strip():
                raise ValueError(f"Attention profile requires {key}")
        self.config = deepcopy(config)
        self.reference_id = config["profile_id"] + ":" + hashlib.sha256(
            json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


def load_attention_execution(filename=None, config_dir=None):
    base = Path(config_dir) if config_dir is not None else Path(__file__).resolve().parents[2] / "configs"
    path = Path(filename) if filename is not None else base / "execution/attention.json"
    if filename is not None and not path.is_absolute() and not path.is_file():
        path = base / "execution" / path
    return AttentionExecution(json.loads(path.read_text(encoding="utf-8")))
