"""
TSP Configuration Loader
"""
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ..hardware.tsp import TSPSpec
from .workload import load_workload_config, parse_workloads, resolve_config_path


class TSPConfigLoader:
    """Load native TSP hardware and workload configurations."""

    def __init__(self, config_dir: Optional[str] = None):
        if config_dir:
            self.config_dir = Path(config_dir)
        else:
            self.config_dir = Path(__file__).parent.parent.parent / "configs"

    def _load_json(self, file_path: Path) -> Dict[str, Any]:
        with open(file_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def load_hardware_config(self, filename: str = "tsp.json") -> Dict[str, Any]:
        return self._load_json(resolve_config_path(filename, self.config_dir, "hardware"))

    def load_workload_config(self, filename: str = "tiny_gemm.json") -> Dict[str, Any]:
        return load_workload_config(filename, self.config_dir)

    def create_tsp_spec(self, hw_config: Dict[str, Any]) -> TSPSpec:
        return TSPSpec.from_config(hw_config)

    def get_workload(self, workload_config: Dict[str, Any]) -> Tuple[int, int, int]:
        return parse_workloads(workload_config)[0].shape
