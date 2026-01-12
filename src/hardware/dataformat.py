"""
Data Format Specifications

定义支持的数据格式（INT8/BF16/MXINT8/FP8/MXFP8_M4）的属性和开销参数
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class DataFormatSpec:
    """
    数据格式规格类

    Attributes:
        name: 格式名称
        storage_bytes: 每元素存储字节数
        mantissa_bits: 尾数位数
        guard_bits: 保护位数（防止溢出）
        effective_compute_bits: 有效计算位数 (= mantissa + guard)
    """
    name: str
    storage_bytes: float
    mantissa_bits: int
    guard_bits: int

    @property
    def effective_compute_bits(self) -> int:
        """有效计算位数"""
        return self.mantissa_bits + self.guard_bits

    def get_activity_factor(self, hardware_bits: int = 8, base_factor: float = 0.3) -> float:
        """
        计算CIM能耗活动因子

        Args:
            hardware_bits: 硬件物理位宽，默认8
            base_factor: 基础活动因子，默认0.3

        Returns:
            活动因子 (0.0 ~ 1.0)
        """
        if self.effective_compute_bits <= hardware_bits:
            data_activity_ratio = self.effective_compute_bits / hardware_bits
            return base_factor + (1 - base_factor) * data_activity_ratio
        else:
            # e.g., BF16 需要多次迭代，每次都是满负荷
            return 1.0


# 支持的数据格式定义
FORMAT_SPECS: Dict[str, DataFormatSpec] = {
    "INT8": DataFormatSpec(
        name="INT8",
        storage_bytes=1.0,
        mantissa_bits=8,
        guard_bits=0
    ),
    "BF16": DataFormatSpec(
        name="BF16",
        storage_bytes=2.0,
        mantissa_bits=8,
        guard_bits=7  # 8位尾数 + 7保护位 = 15位有效计算
    ),
    "MXINT8": DataFormatSpec(
        name="MXINT8",
        storage_bytes=1.03125,  # 共享指数开销: 1 + 1/32
        mantissa_bits=8,
        guard_bits=2  # 共享指数限制了动态范围
    ),
    "FP8": DataFormatSpec(
        name="FP8",
        storage_bytes=1.0,
        mantissa_bits=4,
        guard_bits=5  # 4位尾数 + 5保护位 = 9位有效计算
    ),
    "MXFP8_M4": DataFormatSpec(
        name="MXFP8_M4",
        storage_bytes=1.03125,  # 共享指数开销
        mantissa_bits=4,
        guard_bits=5  # 共享指数限制了动态范围
    ),
}


def get_format_overheads(data_format: str, custom_overheads: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """
    获取指定数据格式的开销参数

    不同数据格式在预处理、计算、后处理阶段有不同的开销：
    - pre_latency_cycles_per_input: 每输入元素的预处理延迟（周期）
    - pre_energy_pj_per_bit: 每比特预处理能耗（pJ）
    - macro_energy_scale: Macro计算能耗缩放因子
    - post_throughput_scale: 后处理吞吐量缩放因子
    - post_energy_scale: 后处理能耗缩放因子

    Args:
        data_format: 数据格式名称
        custom_overheads: 自定义开销参数（可选）

    Returns:
        开销参数字典
    """
    # 默认开销（INT8）
    overheads = {
        "pre_latency_cycles_per_input": 0.0,
        "pre_energy_pj_per_bit": 0.0,
        "macro_energy_scale": 1.0,
        "post_throughput_scale": 1.0,
        "post_energy_scale": 1.0,
    }

    # 根据数据格式更新开销
    format_specific_overheads = {
        "INT8": {},  # INT8 使用默认值
        "BF16": {
            "pre_latency_cycles_per_input": 0.3,
            "pre_energy_pj_per_bit": 0.05,
            "post_throughput_scale": 0.9,
            "post_energy_scale": 1.4,
        },
        "MXINT8": {
            "pre_latency_cycles_per_input": 0.2,
            "pre_energy_pj_per_bit": 0.03,
            "macro_energy_scale": 1.1,
            "post_throughput_scale": 0.95,
            "post_energy_scale": 1.2,
        },
        "FP8": {
            "pre_latency_cycles_per_input": 0.25,
            "pre_energy_pj_per_bit": 0.04,
            "post_throughput_scale": 0.9,
            "post_energy_scale": 1.3,
        },
        "MXFP8_M4": {
            "pre_latency_cycles_per_input": 0.15,
            "pre_energy_pj_per_bit": 0.025,
            "macro_energy_scale": 1.05,
            "post_throughput_scale": 0.95,
            "post_energy_scale": 1.15,
        },
    }

    if data_format in format_specific_overheads:
        overheads.update(format_specific_overheads[data_format])

    # 应用自定义开销
    if custom_overheads:
        overheads.update(custom_overheads)

    return overheads


def get_format_spec(data_format: str) -> DataFormatSpec:
    """
    获取数据格式规格

    Args:
        data_format: 数据格式名称

    Returns:
        DataFormatSpec实例

    Raises:
        ValueError: 不支持的数据格式
    """
    if data_format not in FORMAT_SPECS:
        raise ValueError(f"Unsupported data format: {data_format}. "
                        f"Supported formats: {list(FORMAT_SPECS.keys())}")
    return FORMAT_SPECS[data_format]
