"""Reviewed Core/Cluster/Plane topology; task domains and chip inventory are distinct."""
from .execution import read_execution
from .documentation import without_documentation
# 阅读层级：Macro → Core → 单 SuperLane 上的 Cluster slice → 跨 SuperLane 的 domain。
# 同一 slice 沿 N 拼接，沿 SuperLane 方向的 K 贡献相加；一个 Plane 内有多个独立域。
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType

from .macro import CIMMacroSpec
from .tsp import TSPSpec, VXM_SERVICE_PARAMETERS, VXM_OPERATION_COSTS
from .parameters import read_parameter


# A cluster_slice is the five-Core group on ONE SuperLane.
# A cluster_domain spans the same group index across all 20 SuperLanes.
FIXED = {
    "macros_per_core": (4, "count", "core"),
    "cores_per_cluster": (5, "count", "cluster_slice"),
    "clusters_per_superlane": (4, "count", "superlane_in_plane"),
    "superlanes_per_plane": (20, "count", "plane"),
    "planes_per_hemisphere": (2, "count", "hemisphere"),
    "hemispheres": (2, "count", "chip"),
    "int32_writer_bpc": (256, "byte/cycle", "writer_superlane_plane"),
    "streams_per_plane": (16, "count", "superlane_plane_direction"),
    "lw_cores_per_cycle": (2, "core/cycle", "superlane_plane"),
    "core_stagger_cycles": (1, "cycle", "cluster_slice"),
}
SHARED_PARAMETERS = (
    "superlanes", "lanes_per_superlane", "hemispheres", "planes_per_hemisphere",
    "vxm_alus_per_lane", "mem_slices_per_hemisphere", "mem_tiles_per_slice",
    "memory_tile_bytes", "banks_per_memory_tile", "usable_mem_bytes", "sr_coordinates",
    "streams_per_direction", "stream_width_bytes", "clock_ghz", "mem_read_bpc",
    "mem_write_bpc", *VXM_SERVICE_PARAMETERS, "vxm_add_latency_cycles",
)


@dataclass(frozen=True)
class CIMCoreSpec:
    # Core 内的 Macro 共用同一段 K，各自负责不同输出列，因此只扩大 N。
    macro: CIMMacroSpec
    macros: int

    @property
    def k(self):
        return self.macro.block_k_dim

    @property
    def n(self):
        return self.macros * self.macro.block_n_dim


@dataclass(frozen=True)
class CIMClusterSpec:
    """One local Cluster slice; N expands, K does not."""
    core: CIMCoreSpec
    cores: int

    @property
    def macros(self):
        return self.cores * self.core.macros

    @property
    def n(self):
        return self.cores * self.core.n


@dataclass(frozen=True)
class CIMPlaneSpec:
    cluster: CIMClusterSpec
    superlanes: int
    clusters_per_superlane: int

    @property
    def domain_k(self):
        # 同一域在每条 SuperLane 上覆盖不同 K16；20 条共同覆盖 K320、保持相同 N640。
        return self.superlanes * self.cluster.core.k

    @property
    def domain_macros(self):
        return self.superlanes * self.cluster.macros

    @property
    def macros(self):
        return self.clusters_per_superlane * self.domain_macros


class TSPCIMSpec:
    def __init__(self, config):
        config = without_documentation(config)
        self.execution = read_execution(config.get("execution"), "tsp_cim")
        required = {"execution", "schema_version", "model", "profile_id", "description", "supported_formats",
                    "native_reference", "macro_reference", "parameters",
                    "_native_reference_config", "_macro_profile"}
        if not isinstance(config, dict) or set(config) != required:
            raise ValueError("CIM schema 7 requires resolved profiles; legacy fields conflict with Core/Cluster/Plane topology")
        config["_native_reference_config"] = without_documentation(config["_native_reference_config"])
        if type(config["schema_version"]) is not int or config["schema_version"] != 7:
            raise ValueError("CIM schema_version must be 7; archive the four NIM inventory parameters from schema 6")
        if config["model"] != "tsp_cim_cluster_reference" or config["supported_formats"] != ["INT8"]:
            raise ValueError("CIM Core/Cluster/Plane reference supports INT8 only")
        for key in ("profile_id", "description", "native_reference", "macro_reference"):
            if not isinstance(config[key], str) or not config[key].strip():
                raise ValueError(f"{key} is required")
        params = config["parameters"]
        if not isinstance(params, dict) or set(params) != set(FIXED):
            raise ValueError("CIM parameters must explicitly declare the supported topology only")
        values = {}
        for name, (expected, unit, scope) in FIXED.items():
            values[name] = read_parameter(params[name], name, unit, scope, integer=True)
            if values[name] != expected:
                raise ValueError(f"{name}: reviewed profile requires {expected}")
        self.values = MappingProxyType(values)
        self._config = deepcopy(config)
        self.macro_spec = CIMMacroSpec(config["_macro_profile"])
        # 拓扑、Macro、原生共享系统三份配置在此会合，先检查几何和带宽分配是否一致。
        self.native_reference_spec = TSPSpec(config["_native_reference_config"])
        native = self.native_reference_spec
        for field, native_field in (("superlanes_per_plane", "superlanes"),
                                    ("hemispheres", "hemispheres"),
                                    ("planes_per_hemisphere", "planes_per_hemisphere")):
            if values[field] != native.values[native_field]:
                raise ValueError(f"{field} conflicts with native reference")
        if native.n_lanes_per_superlane != self.macro_spec.block_k_dim:
            raise ValueError("Macro K conflicts with native SuperLane width")
        if native.stream_width_bytes != native.total_lanes:
            raise ValueError("Native stream width must contain one byte per lane")
        self.core = CIMCoreSpec(self.macro_spec, values["macros_per_core"])
        self.cluster = CIMClusterSpec(self.core, values["cores_per_cluster"])
        self.plane = CIMPlaneSpec(self.cluster, values["superlanes_per_plane"],
                                 values["clusters_per_superlane"])
        self.mem_total_bytes = native.mem_total_bytes
        self.clock_freq_ghz = native.clock_freq_ghz
        stream_budget = values["streams_per_plane"] * values["planes_per_hemisphere"]
        # 两个 Plane 共享本侧方向上的 stream 预算；LW 按每 SuperLane 的捕获宽度检查。
        if stream_budget > native.n_streams_per_direction:
            raise ValueError("Plane stream allocation exceeds shared native direction budget")
        if self.lw_bpc_per_superlane_plane > values["streams_per_plane"] * native.n_lanes_per_superlane:
            raise ValueError("LW design target exceeds allocated stream capture width")

    @classmethod
    def from_config(cls, config):
        return cls(config)

    @property
    def reference_id(self):
        # Identify content, not the path used to find referenced files.
        content = {k: v for k, v in self._config.items()
                   if k not in {"native_reference", "macro_reference"}}
        return self._config["profile_id"] + ":" + hashlib.sha256(
            json.dumps(content, sort_keys=True, allow_nan=False).encode()).hexdigest()

    @property
    def total_planes_chip(self):
        return self.values["hemispheres"] * self.values["planes_per_hemisphere"]

    @property
    def total_macros_chip(self):
        return self.total_planes_chip * self.plane.macros

    @property
    def macro_peak_mac_per_cycle_int8(self):
        return self.macro_spec.peak_mac_per_cycle_int8

    @property
    def lw_bpc_per_superlane_plane(self):
        return self.values["lw_cores_per_cycle"] * self.core.macros * self.macro_spec.values["write_wordline_bytes"]

    def hierarchy(self):
        return {
            "macro": {"k": self.core.k, "n": self.macro_spec.block_n_dim},
            "core": {"macros": self.core.macros, "k": self.core.k, "n": self.core.n,
                     "organization": "disjoint N shards; no inter-Macro N reduction"},
            "cluster_slice": {"cores": self.cluster.cores, "macros": self.cluster.macros,
                              "k": self.core.k, "n": self.cluster.n, "scope": "one_superlane"},
            "cluster_domain": {"superlanes": self.plane.superlanes, "cores": self.plane.superlanes * self.cluster.cores,
                               "macros": self.plane.domain_macros, "k": self.plane.domain_k, "n": self.cluster.n,
                               "reduction": "same N shard across SuperLanes; no reduction across Cluster domains"},
            "plane": {"cluster_domains": self.plane.clusters_per_superlane,
                      "cluster_slices": self.plane.superlanes * self.plane.clusters_per_superlane,
                      "cores": self.plane.macros // self.core.macros, "macros": self.plane.macros},
            "hemisphere": {"planes": self.values["planes_per_hemisphere"],
                           "cluster_domains": self.values["planes_per_hemisphere"] * self.plane.clusters_per_superlane,
                           "cores": self.plane.macros * self.values["planes_per_hemisphere"] // self.core.macros,
                           "macros": self.plane.macros * self.values["planes_per_hemisphere"]},
            "chip": {"hemispheres": self.values["hemispheres"], "planes": self.total_planes_chip,
                     "cluster_domains": self.total_planes_chip * self.plane.clusters_per_superlane,
                     "cores": self.total_macros_chip // self.core.macros, "macros": self.total_macros_chip},
        }

    def inventory(self):
        # active_context 只计当前一套权重，resident 计四套；物理权重 SRAM 应使用后者。
        native = self.native_reference_spec
        resident = self.total_macros_chip * self.macro_spec.resident_weight_bytes
        return {
            "scope": "chip", **self.hierarchy()["chip"],
            "macro_weight_active_context_bytes": self.total_macros_chip * self.macro_spec.context_bytes,
            "macro_weight_resident_bytes": resident,
            "global_mem_bytes": self.mem_total_bytes,
            "known_mem_plus_macro_bytes": self.mem_total_bytes + resident,
            "vxm_alus": native.total_vxm_alus, "mem_slices": native.mem_n_slices,
            "memory_tiles": native.mem_n_slices * native.values["mem_tiles_per_slice"],
        }

    def derived(self):
        def quantity(value, unit, scope, formula):
            return {"value": value, "unit": unit, "scope": scope, "formula": formula,
                    "basis": "architecture arithmetic; not simulated throughput or calibrated service"}
        result = {}
        for scope, macros in (("core", self.core.macros), ("cluster_slice", self.cluster.macros),
                               ("cluster_domain", self.plane.domain_macros), ("plane", self.plane.macros),
                               ("hemisphere", self.plane.macros * self.values["planes_per_hemisphere"]),
                               ("chip", self.total_macros_chip)):
            peak = macros * self.macro_peak_mac_per_cycle_int8
            result["int8_mac_per_cycle_" + scope] = quantity(peak, "MAC/cycle", scope, "macros * Macro steady-state MAC/cycle")
            result["int8_tops_" + scope] = quantity(
                None if self.clock_freq_ghz is None else peak * 2 * self.clock_freq_ghz / 1000,
                "TOPS", scope, "MAC/cycle * 2 ops/MAC * native.clock_ghz / 1000")
        native = self.native_reference_spec
        result["stream_bpc_per_direction"] = quantity(native.stream_bw_per_direction_bytes_per_cycle,
                "byte/cycle", "direction", "native.streams_per_direction * native.stream_width_bytes")
        result["lw_capture_bpc_per_superlane_plane"] = quantity(self.lw_bpc_per_superlane_plane,
                "byte/cycle", "superlane_plane", "lw_cores_per_cycle * macros_per_core * Macro.wordline_bytes")
        result["lw_capture_bpc_per_plane"] = quantity(self.lw_bpc_per_superlane_plane * self.plane.superlanes,
                "byte/cycle", "plane", "per-SuperLane capture width * superlanes; not measured programming bandwidth")
        result["lw_capture_bpc_per_chip"] = quantity(
                self.lw_bpc_per_superlane_plane * self.plane.superlanes * self.total_planes_chip,
                "byte/cycle", "chip", "per-Plane distributed LW capture width * planes_per_chip")
        # 文档的 LW 轮转覆盖一条 SuperLane 上四个 Cluster 的全部 20 Core。
        # 每轮给各 Macro 写一条 wordline；160 拍是完整轮转预算，不是本次任务装载时间。
        cores_per_row = self.cluster.cores * self.plane.clusters_per_superlane
        issue_width = self.values["lw_cores_per_cycle"]
        scan_cycles = (cores_per_row + issue_width - 1) // issue_width
        wordlines = self.macro_spec.context_bytes // self.macro_spec.values["write_wordline_bytes"]
        result["lw_scan_cycles_per_wordline"] = quantity(scan_cycles, "cycle", "superlane_plane",
                "ceil(cores_per_cluster * clusters_per_superlane / lw_cores_per_cycle)")
        result["lw_full_context_scan_cycles"] = quantity(scan_cycles * wordlines, "cycle", "superlane_plane",
                "lw_scan_cycles_per_wordline * Macro.wordlines_per_context")
        lw_source = self._config["parameters"]["lw_cores_per_cycle"]
        for name in ("lw_capture_bpc_per_chip", "lw_scan_cycles_per_wordline", "lw_full_context_scan_cycles"):
            result[name].update(source=lw_source["source"], source_version=lw_source["source_version"],
                                basis="documented LW schedule arithmetic; excludes transport stalls and physical write latency")
        # 按 §3.4.6 区分每个 writer SuperLane 的目标与 Plane/半片聚合服务。
        result["int32_writer_bpc_per_writer_superlane_plane"] = quantity(self.values["int32_writer_bpc"],
                "byte/cycle", "writer_superlane_plane", "distributed writer target; shared MEM/stream limits still apply")
        result["nim_output_target_bpc_per_hemisphere"] = quantity(
                self.values["int32_writer_bpc"] * self.plane.superlanes * self.values["planes_per_hemisphere"],
                "byte/cycle", "hemisphere", "per-writer target * 20 transposed writer SuperLanes * 2 Planes; not effective bandwidth")
        result["vxm_service"] = self.native_reference_spec.vxm_service()
        return result

    def provenance(self):
        return {
            "topology": deepcopy(self._config["parameters"]),
            "macro": self.macro_spec.to_dict(),
            "shared_system": {k: deepcopy(self.native_reference_spec.parameters[k]) for k in SHARED_PARAMETERS},
            "native_reference_id": self.native_reference_spec.reference_id,
            "topology_reference_id": self.reference_id,
        }

    def missing_parameters(self):
        # FP32 operation costs are visible in provenance/service, not required by INT8 GEMM.
        unused_costs = {name for op, name in VXM_OPERATION_COSTS.items() if op != "int32_add"}
        provenance = self.provenance()
        return ([f"{group}.{name}" for group in ("topology", "shared_system")
                 for name, item in provenance[group].items()
                 if item["value"] is None and name not in unused_costs]
                + [f"macro.{name}" for name, item in provenance["macro"]["parameters"].items()
                   if item["value"] is None])
