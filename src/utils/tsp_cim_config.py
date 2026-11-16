"""Resolve independent Macro/topology profiles and the shared native TSP reference."""
# 解析三份配置的引用：CIM 拓扑、Macro 规格、原生 TSP 共享系统；跨配置约束交给 Spec。
import json
from pathlib import Path

from ..hardware.tsp_cim import TSPCIMSpec
from ..hardware.macro import CIMMacroSpec
from .tsp_config import TSPConfigLoader
from .workload import load_workload_config, parse_workloads, resolve_config_path


def resolve_macro_reference(config, hardware_path):
    """Resolve the single Macro profile relative to its owning config file."""
    # 引用相对于所属硬件配置解析，移动整套配置目录后仍能找到同一份 Macro profile。
    # _macro_profile 是加载器解析出的内部数据，避免各入口各自维护一套 Macro 默认值。
    if "macro" in config:
        raise ValueError("Inline legacy macro config is unsupported; use macro_reference")
    reference = config.get("macro_reference")
    if not isinstance(reference, str) or not reference:
        raise ValueError("macro_reference is required")
    path = Path(reference)
    if not path.is_absolute():
        path = Path(hardware_path).parent / path
    with path.open(encoding="utf-8") as stream:
        profile = json.load(stream)
    CIMMacroSpec.from_dict(profile)
    config["_macro_profile"] = profile
    return config


class TSPCIMConfigLoader:
    def __init__(self, config_dir=None):
        self.config_dir = Path(config_dir) if config_dir else Path(__file__).resolve().parents[2] / "configs"

    def load_hardware_config(self, filename="tsp_cim.json"):
        path = resolve_config_path(filename, self.config_dir, "hardware")
        with path.open(encoding="utf-8") as stream:
            config = json.load(stream)
        config = resolve_macro_reference(config, path)
        reference = config.get("native_reference")
        if not isinstance(reference, str) or not reference:
            raise ValueError("CIM config must explicitly name native_reference")
        native_path = Path(reference)
        # 与 Macro 引用一致，以当前硬件配置所在目录为基准，而非进程工作目录。
        if not native_path.is_absolute():
            native_path = path.parent / native_path
        config["_native_reference_config"] = TSPConfigLoader(self.config_dir).load_hardware_config(str(native_path.resolve()))
        # No injected defaults, duplicate native fields, or historical SRAM overlay.
        TSPCIMSpec(config)
        return config

    def create_tsp_spec(self, hardware_config):
        return TSPCIMSpec(hardware_config)

    def load_workload_config(self, filename="tiny_gemm.json"):
        return load_workload_config(filename, self.config_dir)

    def get_workload(self, workload_config):
        return parse_workloads(workload_config)[0].shape
