"""
Digital CIM Simulator Package
"""
from .hardware import CIMMacroSpec, DataFormatSpec, FORMAT_SPECS, get_format_overheads
from .core import CIMCoreSimulator, TilingEngine
from .dse import CIMDesignExplorer
from .utils import ConfigLoader, ResultReporter

__version__ = "1.0.0"
__all__ = [
    "CIMMacroSpec",
    "DataFormatSpec",
    "FORMAT_SPECS",
    "get_format_overheads",
    "CIMCoreSimulator",
    "TilingEngine",
    "CIMDesignExplorer",
    "ConfigLoader",
    "ResultReporter",
]
