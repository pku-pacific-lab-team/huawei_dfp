"""Source-traceable INT8 TSP-CIM Macro specification; no legacy defaults."""
# Macro 的几何和动作口径只有这一份定义，通用 CIM 与 CIM-TSP 都引用它。
from copy import deepcopy
import hashlib
import json
from types import MappingProxyType

from .parameters import read_parameter


class CIMMacroSpec:
    """One Macro: four 512-byte contexts, K16 x N32, bit-serial INT8."""
    # Macro 只定义几何、动作和启动间隔；任务时间由 Cluster 层计算。
    FIXED = {
        "banks_per_context": (16, "count", "context"),
        "subarrays_per_bank": (16, "count", "bank"),
        "columns_per_subarray": (16, "bit", "subarray"),
        "weight_contexts": (4, "count", "macro"),
        "weight_bits": (8, "bit", "operand"),
        "activation_bits": (8, "bit", "operand"),
        "accumulator_bits": (32, "bit", "result"),
        "input_ii_cycles": (8, "cycle", "macro"),
        "write_wordline_bytes": (32, "byte", "macro"),
    }

    def __init__(self, config):
        if not isinstance(config, dict) or set(config) - {
            "schema_version", "profile_id", "description", "parameters"
        }:
            raise ValueError("Macro requires a versioned profile; legacy geometry is unsupported")
        if type(config.get("schema_version")) is not int or config["schema_version"] != 2:
            raise ValueError("Macro schema_version must be 2; remove retired local timing parameters")
        if not isinstance(config.get("profile_id"), str) or not config["profile_id"].strip():
            raise ValueError("Macro profile_id is required")
        params = config.get("parameters")
        if not isinstance(params, dict) or set(params) != set(self.FIXED):
            raise ValueError("Macro parameters must explicitly specify geometry and action contracts only")
        values = {}
        for name, (expected, unit, scope) in self.FIXED.items():
            value = read_parameter(params[name], name, unit, scope, integer=True)
            if value != expected:
                raise ValueError(f"{name}: this INT8 profile requires {expected}")
            values[name] = value
        self.values = MappingProxyType(values)
        # 冻结运行时数值并保存配置副本，避免调用方修改配置后悄悄改变规格或来源标识。
        self._config = deepcopy(config)

    @classmethod
    def from_dict(cls, config):
        return cls(config)

    def to_dict(self):
        return deepcopy(self._config)

    @property
    def reference_id(self):
        digest = hashlib.sha256(json.dumps(
            self._config, sort_keys=True, allow_nan=False).encode()).hexdigest()
        return f"{self._config['profile_id']}:{digest}"

    @property
    def block_k_dim(self):
        return self.values["subarrays_per_bank"]

    @property
    def block_n_dim(self):
        # 每个 subarray 的 16 bit 放两个 INT8 权重，16 个 bank 因此覆盖 32 个输出列。
        return self.values["banks_per_context"] * (
            self.values["columns_per_subarray"] // self.values["weight_bits"])

    @property
    def context_bytes(self):
        # 一套权重是 bank×subarray×bit/8；乘 context 数后才是 Macro 总驻留容量。
        return (self.values["banks_per_context"] * self.values["subarrays_per_bank"]
                * self.values["columns_per_subarray"] // 8)

    @property
    def resident_weight_bytes(self):
        return self.context_bytes * self.values["weight_contexts"]

    @property
    def macs_per_launch(self):
        return self.block_k_dim * self.block_n_dim

    @property
    def peak_mac_per_cycle_int8(self):
        # 512 MAC/次除以 8 cycle 的启动间隔得到稳态规格值，不能当作首结果延迟。
        return self.macs_per_launch // self.values["input_ii_cycles"]

    def derived(self):
        return {
            "block_k": self.block_k_dim, "block_n": self.block_n_dim,
            "weight_contexts": self.values["weight_contexts"],
            "context_bytes": self.context_bytes,
            "resident_weight_bytes": self.resident_weight_bytes,
            "wordlines_per_context": self.context_bytes // self.values["write_wordline_bytes"],
            "macs_per_launch": self.macs_per_launch,
            "peak_mac_per_cycle_int8": self.peak_mac_per_cycle_int8,
            "input_ii_cycles": self.values["input_ii_cycles"],
            "output_bytes_per_launch": self.block_n_dim * self.values["accumulator_bits"] // 8,
        }

    def __repr__(self):
        return "CIMMacroSpec(INT8, K16xN32, contexts=4, II=8)"
