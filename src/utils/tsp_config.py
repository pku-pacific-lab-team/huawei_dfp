"""
TSP Configuration Loader
"""
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ..hardware.tsp import TSPSpec


class TSPConfigLoader:
    """Load TSP hardware, energy, and workload configurations."""

    def __init__(self, config_dir: Optional[str] = None):
        if config_dir:
            self.config_dir = Path(config_dir)
        else:
            self.config_dir = Path(__file__).parent.parent.parent / "configs"

    def _load_json(self, file_path: Path) -> Dict[str, Any]:
        with open(file_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def load_energy_config(self, filename: str = "7nm_tsp.json") -> Dict[str, float]:
        config = self._load_json(self.config_dir / "energy" / filename)
        return config.get("parameters", config)

    def load_area_config(self, filename: str = "7nm_tsp.json") -> Dict[str, float]:
        config = self._load_json(self.config_dir / "area" / filename)
        return config.get("parameters", config)

    def load_hardware_config(self, filename: str = "tsp.json") -> Dict[str, Any]:
        return self._load_json(self.config_dir / "hardware" / filename)

    def load_workload_config(self, filename: str = "default_gemm.json") -> Dict[str, Any]:
        return self._load_json(self.config_dir / "workload" / filename)

    def create_tsp_spec(self, hw_config: Dict[str, Any]) -> TSPSpec:
        return TSPSpec.from_config(hw_config)

    def get_workload(self, workload_config: Dict[str, Any]) -> Tuple[int, int, int]:
        gemm = workload_config.get("gemm", {})
        return (
            gemm.get("M", 1024),
            gemm.get("K", 4096),
            gemm.get("N", 1024),
        )
