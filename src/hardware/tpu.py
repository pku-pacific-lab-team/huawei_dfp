"""TPU v4 MXU/TensorCore/chip specification, inventory and input validation."""
# 规格层负责参数校验、层级和库存派生；任务计数在 src/core/tpu，不在此处执行。
from dataclasses import asdict, dataclass
import math


# name: (unit, scope, integer). All fields are explicit, including unknowns.
PARAMETERS = {
    "clock_hz": ("Hz", "chip", False),
    "tensorcores_per_chip": ("unit", "chip", True),
    "mxus_per_tensorcore": ("unit", "tensorcore", True),
    "mxu_rows": ("element", "mxu", True),
    "mxu_columns": ("element", "mxu", True),
    "int8_macs_per_cell_cycle": ("MAC/cycle", "mxu_cell", False),
    "nominal_int8_tops": ("TOPS", "chip", False),
    "vpus_per_tensorcore": ("unit", "tensorcore", True),
    "vpu_lanes": ("lane", "vpu", True),
    "vpu_alus_per_lane": ("unit", "lane", True),
    "vector_int32_adds_per_cycle": ("op/cycle", "tensorcore", False),
    "vector_fp32_ops_per_cycle": ("op/cycle", "tensorcore", False),
    "scalar_units_per_tensorcore": ("unit", "tensorcore", True),
    "scalar_ops_per_cycle": ("op/cycle", "tensorcore", False),
    "sfu_units_per_tensorcore": ("unit", "tensorcore", True),
    "sfu_fp32_ops_per_cycle": ("op/cycle", "tensorcore", False),
    "vmem_physical_bytes": ("byte", "tensorcore", True),
    "vmem_usable_bytes": ("byte", "tensorcore", True),
    "vmem_bandwidth_bytes_per_second": ("byte/s", "tensorcore", False),
    "cmem_physical_bytes": ("byte", "chip", True),
    "cmem_usable_bytes": ("byte", "chip", True),
    "cmem_bandwidth_bytes_per_second": ("byte/s", "chip", False),
    "smem_physical_bytes": ("byte", "tensorcore", True),
    "register_file_physical_bytes": ("byte", "tensorcore", True),
    "sparsecores_per_chip": ("unit", "chip", True),
    "spmem_physical_bytes": ("byte", "chip", True),
    "hbm_physical_bytes": ("byte", "chip", True),
    "hbm_usable_bytes": ("byte", "chip", True),
    "hbm_bandwidth_bytes_per_second": ("byte/s", "chip", False),
    "ici_links_per_chip": ("link", "chip", True),
    "ici_link_reported_bytes_per_second": ("byte/s", "link", False),
    "ici_link_one_way_bytes_per_second": ("byte/s", "link", False),
    "process_node_nm": ("nm", "die", False),
    "die_area_upper_bound_mm2": ("mm2", "die", False),
}


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
        require_fields(config, ("schema_version", "model", "description", "parameters", "interconnect"), "hardware")
        if type(config["schema_version"]) is not int or config["schema_version"] != 1 or config["model"] != "tpu_v4_count":
            raise ValueError("Expected schema 1, model tpu_v4_count")
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
        interconnect = config["interconnect"]
        require_fields(interconnect, ("topology", "source", "version", "conditions"), "interconnect")
        if any(not isinstance(v, str) or not v.strip() for v in interconnect.values()):
            raise ValueError("interconnect metadata must be nonempty strings")
        self.interconnect = dict(interconnect)

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
                  "int8_macs_per_cell_cycle", "hbm_bandwidth_bytes_per_second")
        missing = [name for name in needed if self.values[name] is None]
        if missing:
            raise ValueError("TPU timing parameters missing: " + ", ".join(missing))

    def provenance(self):
        return {name: asdict(p) for name, p in self.parameters.items()}

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
            "nominal_chip_int8_tops": v["nominal_int8_tops"],
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
            "vpus": v["vpus_per_tensorcore"],
            "vpu_alus": _known_product(v["vpus_per_tensorcore"], v["vpu_lanes"], v["vpu_alus_per_lane"]),
            "scalar_units": v["scalar_units_per_tensorcore"],
            "sfu_units": v["sfu_units_per_tensorcore"],
            "private_vmem_bytes": v["vmem_physical_bytes"],
            "private_smem_bytes": v["smem_physical_bytes"],
            "private_register_file_bytes": v["register_file_physical_bytes"],
            "status": "partial_physical_inventory; no_unknown_components_treated_as_zero",
        }

    def inventory(self):
        """Chip totals: aggregate private resources, count shared resources once."""
        # 先汇总各核私有资源，再加入只计一份的 CMEM/HBM 等共享资源，避免重复乘核数。
        # 私有 VMEM 的合计不能作为某一个 TensorCore 的可用容量。
        v = self.values
        local = self.tensorcore_inventory()
        private = {name: _known_product(local[name], v["tensorcores_per_chip"]) for name in (
            "mxus", "vpus", "vpu_alus", "scalar_units", "sfu_units", "private_vmem_bytes",
            "private_smem_bytes", "private_register_file_bytes")}
        return {
            "scope": "chip",
            "tensorcores": v["tensorcores_per_chip"],
            **private,
            "sparsecores": v["sparsecores_per_chip"],
            "shared_cmem_bytes": v["cmem_physical_bytes"],
            "sparse_spmem_bytes": v["spmem_physical_bytes"],
            "external_hbm_bytes": v["hbm_physical_bytes"],
            "ici_links": v["ici_links_per_chip"],
            "die_area_upper_bound_mm2": v["die_area_upper_bound_mm2"],
            "die_area_bound_operator": "<",
            "total_area_mm2": None,
            "status": "partial_physical_inventory; no_unknown_components_treated_as_zero",
        }
