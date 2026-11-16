"""
Configuration Loader

加载和解析JSON配置文件
"""
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

from ..hardware.cube import CubeSpec


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

    def get_l0_cube_params(self, hardware_config: Dict[str, Any]) -> Dict[str, int]:
        """
        从硬件配置获取Cube专用的L0 Buffer参数（A/B/C三个独立buffer）

        Args:
            hardware_config: 硬件配置字典

        Returns:
            L0 Cube参数字典，包含：
            - size_l0_A_input_buffer_bytes
            - size_l0_B_input_buffer_bytes
            - size_l0_C_output_buffer_bytes
        """
        memory = hardware_config.get("memory", {})
        return {
            "size_l0_A_input_buffer_bytes": memory.get("size_l0_a_input_buffer_bytes", 4096),
            "size_l0_B_input_buffer_bytes": memory.get("size_l0_b_input_buffer_bytes", 4096),
            "size_l0_C_output_buffer_bytes": memory.get("size_l0_c_output_buffer_bytes", 4096),
        }

    def get_l1_cube_params(self, hardware_config: Dict[str, Any]) -> Dict[str, int]:
        """
        从硬件配置获取Cube专用的L1 Buffer参数

        Args:
            hardware_config: 硬件配置字典

        Returns:
            L1 Cube参数字典，包含：
            - size_l1_buffer_bytes: L1 Buffer大小（字节）
            
        注意：Cube架构中L1只对应1个core，主要用于存储更大的数据块
        """
        memory = hardware_config.get("memory", {})
        return {
            "size_l1_buffer_bytes": memory.get("size_l1_buffer_bytes", 8388608),  # 默认8MB
        }

    def get_cube_bandwidth_params(self, hardware_config: Dict[str, Any]) -> Dict[str, float]:
        """
        从硬件配置获取Cube专用的带宽参数

        Args:
            hardware_config: 硬件配置字典

        Returns:
            Cube带宽参数字典，包含：
            - bw_l0_to_cube_gbps: L0到Cube的带宽（Gbps）
            - bw_l1_to_l0_gbps: L1到L0的带宽（Gbps）
        """
        bw = hardware_config.get("bandwidth", {})
        
        # L0到Cube的带宽
        if "bw_l0_to_cube_gbps" in bw:
            bw_l0_to_cube = bw["bw_l0_to_cube_gbps"]
        elif "bw_l0_b_to_l0_c_gbps" in bw:
            bw_l0_to_cube = bw["bw_l0_b_to_l0_c_gbps"]
        else:
            bw_l0_to_cube = 128.0  # 默认值
        
        # L1到L0的带宽
        if "bw_l1_to_l0_gbps" in bw:
            bw_l1_to_l0 = bw["bw_l1_to_l0_gbps"]
        elif "bw_noc_to_l0_a_gbps" in bw:
            # 如果没有明确的L1到L0带宽，可以使用NoC到L0的带宽作为近似
            bw_l1_to_l0 = bw["bw_noc_to_l0_a_gbps"]
        else:
            bw_l1_to_l0 = 128.0  # 默认值
        
        return {
            "bw_l0_to_cube_gbps": bw_l0_to_cube,
            "bw_l1_to_l0_gbps": bw_l1_to_l0
        }

    def get_clock_freq(self, hardware_config: Dict[str, Any]) -> float:
        """
        从硬件配置获取时钟频率

        Args:
            hardware_config: 硬件配置字典

        Returns:
            时钟频率（GHz）
        """
        # 支持多种配置格式：
        # - Cube.clock_freq_ghz (ascend.json)
        # - core.clock_freq_ghz (default_core.json)
        cube_config = hardware_config.get("Cube", {})
        core_config = hardware_config.get("core", {})
        
        # 使用 get 方法并提供默认值，避免 0 值被误判
        if "clock_freq_ghz" in cube_config:
            return cube_config["clock_freq_ghz"]
        if "clock_freq_ghz" in core_config:
            return core_config["clock_freq_ghz"]
        return 1.0  # 默认值

    def create_cube_spec(self, hardware_config: Dict[str, Any]) -> CubeSpec:
        """
        从硬件配置创建CubeSpec实例

        Args:
            hardware_config: 硬件配置字典

        Returns:
            CubeSpec实例
        """
        # 尝试从多种可能的配置位置读取
        cube_config = hardware_config.get("Cube", {})
        macro_config = hardware_config.get("macro", {})
        
        # 如果 Cube 配置中有 cube 规格，使用它；否则使用 macro 配置或默认值
        cube_spec_config = cube_config.get("cube", {}) or macro_config
        
        return CubeSpec.from_dict(cube_spec_config)


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
