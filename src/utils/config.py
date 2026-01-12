"""
Configuration Loader

加载和解析JSON配置文件
"""
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

from ..hardware.macro import CIMMacroSpec


class ConfigLoader:
    """
    配置文件加载器

    支持加载三种类型的配置：
    - 能量模型配置
    - 硬件参数配置
    - 工作负载配置
    """

    def __init__(self, config_dir: Optional[str] = None):
        """
        初始化配置加载器

        Args:
            config_dir: 配置文件目录路径（可选）
        """
        if config_dir:
            self.config_dir = Path(config_dir)
        else:
            # 默认使用项目根目录下的configs文件夹
            self.config_dir = Path(__file__).parent.parent.parent / "configs"

    def _load_json(self, file_path: Path) -> Dict[str, Any]:
        """
        加载JSON文件

        Args:
            file_path: JSON文件路径

        Returns:
            解析后的字典
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def load_energy_config(self, filename: str = "default_22nm.json") -> Dict[str, float]:
        """
        加载能量模型配置

        Args:
            filename: 配置文件名

        Returns:
            能量参数字典
        """
        file_path = self.config_dir / "energy" / filename
        config = self._load_json(file_path)
        return config.get("parameters", config)

    def load_hardware_config(self, filename: str = "default_core.json") -> Dict[str, Any]:
        """
        加载硬件参数配置

        Args:
            filename: 配置文件名

        Returns:
            硬件配置字典
        """
        file_path = self.config_dir / "hardware" / filename
        return self._load_json(file_path)

    def load_workload_config(self, filename: str = "default_gemm.json") -> Dict[str, Any]:
        """
        加载工作负载配置

        Args:
            filename: 配置文件名

        Returns:
            工作负载配置字典
        """
        file_path = self.config_dir / "workload" / filename
        return self._load_json(file_path)

    def create_macro_spec(self, hardware_config: Dict[str, Any]) -> CIMMacroSpec:
        """
        从硬件配置创建CIMMacroSpec实例

        Args:
            hardware_config: 硬件配置字典

        Returns:
            CIMMacroSpec实例
        """
        macro_config = hardware_config.get("macro", {})
        return CIMMacroSpec.from_dict(macro_config)

    def get_workload(self, workload_config: Dict[str, Any]) -> Tuple[int, int, int]:
        """
        从工作负载配置获取GEMM维度

        Args:
            workload_config: 工作负载配置字典

        Returns:
            (M, K, N) 元组
        """
        gemm = workload_config.get("gemm", {})
        return (gemm.get("M", 1024), gemm.get("K", 4096), gemm.get("N", 1024))

    def get_l0_params(self, hardware_config: Dict[str, Any]) -> Dict[str, int]:
        """
        从硬件配置获取L0 Buffer参数

        Args:
            hardware_config: 硬件配置字典

        Returns:
            L0参数字典
        """
        memory = hardware_config.get("memory", {})
        return {
            "input": memory.get("size_l0_input_buffer_bytes", 4096),
            "output": memory.get("size_l0_output_buffer_bytes", 4096)
        }

    def get_bandwidth_params(self, hardware_config: Dict[str, Any]) -> Dict[str, float]:
        """
        从硬件配置获取带宽参数

        Args:
            hardware_config: 硬件配置字典

        Returns:
            带宽参数字典
        """
        bw = hardware_config.get("bandwidth", {})
        return {
            "noc_to_l1": bw.get("bw_noc_to_l1_gbps", 64.0),
            "l1_to_l0": bw.get("bw_l1_to_l0_gbps", 128.0)
        }


def load_all_configs(
    config_dir: Optional[str] = None,
    energy_file: str = "default_22nm.json",
    hardware_file: str = "default_core.json",
    workload_file: str = "default_gemm.json"
) -> Tuple[Dict[str, float], Dict[str, Any], Dict[str, Any]]:
    """
    便捷函数：一次性加载所有配置

    Args:
        config_dir: 配置目录路径
        energy_file: 能量配置文件名
        hardware_file: 硬件配置文件名
        workload_file: 工作负载配置文件名

    Returns:
        (energy_params, hardware_config, workload_config) 元组
    """
    loader = ConfigLoader(config_dir)
    energy_params = loader.load_energy_config(energy_file)
    hardware_config = loader.load_hardware_config(hardware_file)
    workload_config = loader.load_workload_config(workload_file)
    return energy_params, hardware_config, workload_config
