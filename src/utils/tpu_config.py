"""Load a versioned TPU v4 specification without silent parameter defaults."""
import json
from pathlib import Path

from ..hardware.tpu import TPUSpec
from .workload import resolve_config_path


class TPUConfigLoader:
    def __init__(self, config_dir=None):
        self.config_dir = Path(config_dir) if config_dir else Path(__file__).resolve().parents[2] / "configs"

    def load_hardware(self, filename="tpu_v4_reference.json"):
        path = resolve_config_path(filename, self.config_dir, "hardware")
        return TPUSpec(json.loads(path.read_text(encoding="utf-8")))
