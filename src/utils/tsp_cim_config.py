"""
TSP-CIM Configuration Loader

加载和解析 TSP-CIM 混合架构的配置文件
"""
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

from ..hardware.tsp_cim import TSPCIMSpec


class TSPCIMConfigLoader:
    """
    TSP-CIM 配置加载器

    加载硬件、能量和工作负载配置文件
    """

    def __init__(self, config_dir: Optional[str] = None):
        if config_dir:
            self.config_dir = Path(config_dir)
        else:
            self.config_dir = Path(__file__).parent.parent.parent / "configs"

    def _load_json(self, file_path: Path) -> Dict[str, Any]:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def load_energy_config(
        self, filename: str = "7nm_tsp_cim.json"
    ) -> Dict[str, float]:
        file_path = self.config_dir / "energy" / filename
        config = self._load_json(file_path)
        return config.get("parameters", config)

    def load_hardware_config(
        self, filename: str = "tsp_cim.json"
    ) -> Dict[str, Any]:
        file_path = self.config_dir / "hardware" / filename
        return self._load_json(file_path)

    def load_workload_config(
        self, filename: str = "default_gemm.json"
    ) -> Dict[str, Any]:
        file_path = self.config_dir / "workload" / filename
        return self._load_json(file_path)

    def create_tsp_spec(self, hw_config: Dict[str, Any]) -> TSPCIMSpec:
        return TSPCIMSpec.from_config(hw_config)

    def get_workload(
        self, workload_config: Dict[str, Any]
    ) -> Tuple[int, int, int]:
        gemm = workload_config.get("gemm", {})
        return (
            gemm.get("M", 1024),
            gemm.get("K", 4096),
            gemm.get("N", 1024),
        )
