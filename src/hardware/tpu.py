"""TPU v4 MXU/TensorCore/chip specification, inventory and input validation."""
from .execution import read_execution
from .documentation import without_documentation
# 规格层负责参数校验、库存派生和预留服务占用估算；GEMM 调度/计数在 src/core/tpu。
from dataclasses import asdict, dataclass
from fractions import Fraction
import math


# name: (unit, scope, integer). All fields are explicit, including unknowns.
PARAMETERS = {
    "clock_hz": ("Hz", "chip", False),
    "tensorcores_per_chip": ("unit", "chip", True),
    "mxus_per_tensorcore": ("unit", "tensorcore", True),
    "mxu_rows": ("element", "mxu", True),
    "mxu_columns": ("element", "mxu", True),
    "int8_macs_per_cell_cycle": ("MAC/cycle", "mxu_cell", False),
    "vector_base_ops_per_cycle": ("op/cycle", "tensorcore", False),
    "vector_int32_add_cost": ("equivalent_op/op", "tensorcore", False),
    "vector_fp32_basic_op_cost": ("equivalent_op/op", "tensorcore", False),
    "vector_fp32_exp_cost": ("equivalent_op/op", "tensorcore", False),
    "vector_fp32_rsqrt_cost": ("equivalent_op/op", "tensorcore", False),
    "vmem_physical_bytes": ("byte", "tensorcore", True),
    "vmem_usable_bytes": ("byte", "tensorcore", True),
    "vmem_bandwidth_bytes_per_second": ("byte/s", "tensorcore", False),
    "cmem_physical_bytes": ("byte", "chip", True),
    "cmem_usable_bytes": ("byte", "chip", True),
    "cmem_read_bandwidth_bytes_per_second": ("byte/s", "chip", False),
    "cmem_write_bandwidth_bytes_per_second": ("byte/s", "chip", False),
    "hbm_physical_bytes": ("byte", "chip", True),
    "hbm_usable_bytes": ("byte", "chip", True),
    "hbm_bandwidth_bytes_per_second": ("byte/s", "chip", False),
}


VECTOR_OPERATION_COSTS = {
    "int32_add": "vector_int32_add_cost",
    "fp32_add": "vector_fp32_basic_op_cost",
    "fp32_mul": "vector_fp32_basic_op_cost",
    "fp32_exp": "vector_fp32_exp_cost",
    "fp32_rsqrt": "vector_fp32_rsqrt_cost",
}
VECTOR_SERVICE_PARAMETERS = ("vector_base_ops_per_cycle", *dict.fromkeys(VECTOR_OPERATION_COSTS.values()))


def require_fields(raw, fields, name):
    if not isinstance(raw, dict) or set(raw) != set(fields):
        raise ValueError(f"{name}: expected fields {sorted(fields)}")


def _known_product(*values):
    # 任一依赖未知，派生值也必须未知，不能把缺项当作 0 或擅自补默认值。
    return None if any(value is None for value in values) else math.prod(values)


@dataclass(frozen=True)
class Parameter:
    value: float | int | None
    unit: str
    scope: str
    status: str
    source: str
    version: str
    conditions: str

    @classmethod
    def read(cls, raw, name, unit, scope, integer):
        require_fields(raw, cls.__dataclass_fields__, name)
        p = cls(**raw)
        if p.unit != unit or p.scope != scope:
            raise ValueError(f"{name}: invalid unit or scope")
        if p.status not in ("documented", "measured", "assumed", "unknown"):
            raise ValueError(f"{name}: invalid provenance status")
        if any(not isinstance(v, str) or not v.strip() for v in (p.source, p.version, p.conditions)):
            raise ValueError(f"{name}: source, version and conditions are required")
        if p.value is None:
            if p.status != "unknown":
                raise ValueError(f"{name}: null requires unknown status")
        elif (p.status == "unknown" or type(p.value) not in (int, float)
              or not math.isfinite(p.value) or p.value <= 0
              or (integer and type(p.value) is not int)):
            raise ValueError(f"{name}: expected positive finite {'integer' if integer else 'number'}")
        return p


class TPUSpec:
    def __init__(self, config):
        config = without_documentation(config)
        require_fields(config, ("execution", "schema_version", "model", "description", "parameters"), "hardware")
        if type(config["schema_version"]) is not int or config["schema_version"] != 12 or config["model"] != "tpu_v4_count":
            raise ValueError("Expected schema 12, model tpu_v4_count; archive nominal_int8_tops and the three VPU inventory parameters from schema 11")
        self.execution = read_execution(config.get("execution"), "tpu")
        if not isinstance(config["description"], str) or not config["description"].strip():
            raise ValueError("description is required")
        require_fields(config["parameters"], PARAMETERS, "parameters")
        self.parameters = {name: Parameter.read(config["parameters"][name], name, *rules)
                           for name, rules in PARAMETERS.items()}
        self.values = {name: p.value for name, p in self.parameters.items()}
        for memory in ("vmem", "cmem", "hbm"):
            physical, usable = (self.values[f"{memory}_{kind}_bytes"] for kind in ("physical", "usable"))
            if usable is not None and (physical is None or usable > physical):
                raise ValueError(f"{memory}: usable capacity requires physical capacity and cannot exceed it")

    def require_count_capacities(self):
        # count 不要求所有吞吐参数已知，但必须知道所用 VMEM/HBM 的可用容量。
        needed = ("tensorcores_per_chip", "vmem_usable_bytes", "hbm_usable_bytes")
        missing = [name for name in needed if self.values[name] is None]
        if missing:
            raise ValueError("Counting parameters missing: " + ", ".join(missing))

    def missing_parameters(self):
        return [name for name, p in self.parameters.items() if p.value is None]

    def require_timing_inputs(self):
        # HBM 是 byte/s，换成周期需要时钟；内部端口细节由上层显式近似，不在这里补值。
        needed = ("clock_hz", "tensorcores_per_chip", "mxus_per_tensorcore", "mxu_rows", "mxu_columns",
                  "int8_macs_per_cell_cycle", "hbm_bandwidth_bytes_per_second", "vmem_bandwidth_bytes_per_second")
        missing = [name for name in needed if self.values[name] is None]
        if missing:
            raise ValueError("TPU timing parameters missing: " + ", ".join(missing))

    def provenance(self):
        return {name: asdict(p) for name, p in self.parameters.items()}

    def vector_service(self):
        """One compute service; operation rates are alternatives, not additive."""
        v = self.values
        base = v["vector_base_ops_per_cycle"]
        inputs = VECTOR_SERVICE_PARAMETERS
        return {
            "scope": "tensorcore", "status": "analytical_compute_service",
            "included_in_gemm": False, "resource_policy": "one_shared_vector_service",
            "gemm_operations": [], "nonlinear_included_in_gemm": False,
            "special_function_model": "algorithmic_work_proxy_not_native_instruction_timing",
            "base_ops_per_cycle": base,
            "operation_costs": {op: v[name] for op, name in VECTOR_OPERATION_COSTS.items()},
            "operation_rates_per_cycle": {
                op: None if base is None or v[name] is None else float(Fraction(str(base)) / Fraction(str(v[name])))
                for op, name in VECTOR_OPERATION_COSTS.items()
            },
            "assumed_inputs": [name for name in inputs if self.parameters[name].status == "assumed"],
            "missing_parameters": [name for name in inputs if v[name] is None],
            "conditions": "Weighted work shares one service and rounds once. Compute-only ideal estimate; "
                          "no memory, padding, reduction, instruction latency or dispatch model. "
                          "EXP/RSQRT use explicit total equivalent-work costs, independently configured from basic-op cost. "
                          "FP32 denotes operand format, not correctly rounded or exceptional-value accuracy. "
                          "No numerical evaluation or separate SFU service. FMA, DIV and full nonlinear operators "
                          "remain unsupported. GEMM accumulation stays in the MXU; no Vector charge is added.",
        }

    def vector_service_cycles(self, operation_counts):
        """Ideal shared-service occupancy, separate from the GEMM simulator and wall time."""
        if not isinstance(operation_counts, dict):
            raise ValueError("Vector operation counts must be a dictionary")
        unknown = set(operation_counts) - VECTOR_OPERATION_COSTS.keys()
        if unknown:
            raise ValueError("Unsupported Vector operations: " + ", ".join(sorted(map(str, unknown))))
        if any(type(count) is not int or count < 0 for count in operation_counts.values()):
            raise ValueError("Vector operation counts must be nonnegative integers")
        active = {op: count for op, count in operation_counts.items() if count}
        if not active:
            return 0
        needed = ["vector_base_ops_per_cycle", *dict.fromkeys(VECTOR_OPERATION_COSTS[op] for op in active)]
        missing = [name for name in needed if self.values[name] is None]
        if missing:
            raise ValueError("Vector service parameters missing: " + ", ".join(missing))
        work = sum(count * Fraction(str(self.values[VECTOR_OPERATION_COSTS[op]])) for op, count in active.items())
        return math.ceil(work / Fraction(str(self.values["vector_base_ops_per_cycle"])))

    def derived(self):
        v = self.values
        mxu_macs = _known_product(v["mxu_rows"], v["mxu_columns"], v["int8_macs_per_cell_cycle"])
        # 单元数逐层乘上去得到规格峰值，不代表执行模型已经映射到所有 MXU/核。
        core_macs = _known_product(mxu_macs, v["mxus_per_tensorcore"])
        chip_macs = _known_product(core_macs, v["tensorcores_per_chip"])

        def peak(macs):
            rate = _known_product(macs, v["clock_hz"])
            return None if rate is None else 2 * rate / 1e12

        return {
            "mac_ops": 2,
            "mxu_int8_macs_per_cycle": mxu_macs,
            "mxu_int8_peak_tops": peak(mxu_macs),
            "core_int8_macs_per_cycle": core_macs,
            "core_int8_peak_tops": peak(core_macs),
            "chip_int8_macs_per_cycle": chip_macs,
            "chip_int8_peak_tops": peak(chip_macs),
            "vector_service": self.vector_service(),
            "bandwidth_gbit_per_second": {
                name: None if v[name] is None else v[name] * 8 / 1e9
                for name in PARAMETERS if name.endswith("bytes_per_second")
            },
            "status": "specification_arithmetic_only; assumed_inputs_remain_assumed",
            "assumed_inputs": [name for name, p in self.parameters.items() if p.status == "assumed"],
        }

    def hierarchy(self):
        """Structural multiplicities and specification peaks, not active units."""
        v = self.values
        d = self.derived()
        return {
            "mxu": {
                "scope": "mxu", "rows": v["mxu_rows"], "columns": v["mxu_columns"],
                "cells": _known_product(v["mxu_rows"], v["mxu_columns"]),
                "int8_macs_per_cycle": d["mxu_int8_macs_per_cycle"],
                "int8_peak_tops": d["mxu_int8_peak_tops"],
            },
            "tensorcore": {
                "scope": "tensorcore", "mxus_per_tensorcore": v["mxus_per_tensorcore"],
                "int8_macs_per_cycle": d["core_int8_macs_per_cycle"],
                "int8_peak_tops": d["core_int8_peak_tops"],
            },
            "chip": {
                "scope": "chip", "tensorcores_per_chip": v["tensorcores_per_chip"],
                "mxus_per_chip": _known_product(v["tensorcores_per_chip"], v["mxus_per_tensorcore"]),
                "int8_macs_per_cycle": d["chip_int8_macs_per_cycle"],
                "int8_peak_tops": d["chip_int8_peak_tops"],
            },
            "status": "specification_only; no_execution_mapping",
            "peak_status": d["status"],
            "peak_assumed_inputs": [name for name in (
                "mxu_rows", "mxu_columns", "int8_macs_per_cell_cycle", "mxus_per_tensorcore",
                "tensorcores_per_chip", "clock_hz") if self.parameters[name].status == "assumed"],
        }

    def tensorcore_inventory(self):
        """Physical resources private to one TensorCore; no pooled capacity."""
        v = self.values
        return {
            "scope": "tensorcore",
            "mxus": v["mxus_per_tensorcore"],
            "private_vmem_bytes": v["vmem_physical_bytes"],
            "status": "partial_physical_inventory; no_unknown_components_treated_as_zero",
        }

    def inventory(self):
        """Chip totals: aggregate private resources, count shared resources once."""
        # 先汇总各核私有资源，再加入只计一份的 CMEM/HBM 等共享资源，避免重复乘核数。
        # 私有 VMEM 的合计不能作为某一个 TensorCore 的可用容量。
        v = self.values
        local = self.tensorcore_inventory()
        private = {name: _known_product(local[name], v["tensorcores_per_chip"]) for name in (
            "mxus", "private_vmem_bytes")}
        return {
            "scope": "chip",
            "tensorcores": v["tensorcores_per_chip"],
            **private,
            "shared_cmem_bytes": v["cmem_physical_bytes"],
            "external_hbm_bytes": v["hbm_physical_bytes"],
            "total_area_mm2": None,
            "status": "partial_physical_inventory; no_unknown_components_treated_as_zero",
        }
