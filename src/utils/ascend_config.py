"""Load the explicit, versioned Ascend hardware specification."""
import json
from pathlib import Path

from ..hardware.ascend import AscendSpec
from .workload import resolve_config_path


class AscendConfigLoader:
    def __init__(self, config_dir=None):
        self.config_dir = Path(config_dir) if config_dir else Path(__file__).resolve().parents[2] / "configs"

    def load_hardware(self, filename="ascend910_reference.json"):
        path = resolve_config_path(filename, self.config_dir, "hardware")
        return AscendSpec(json.loads(path.read_text(encoding="utf-8")))
