"""Ascend、TPU、TSP、CIM-TSP GEMM 分析工程。"""
from .hardware import CIMMacroSpec, KNOWN_FORMATS

__version__ = "1.0.0"
__all__ = [
    "CIMMacroSpec",
    "KNOWN_FORMATS",
]
