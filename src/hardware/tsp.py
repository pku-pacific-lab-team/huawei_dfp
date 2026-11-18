"""Issue #2 TSP specifications; single-hemisphere analytical execution."""
# 规格层统一原生 TSP 的 MEM、stream、MXM 和 VXM 口径，CIM-TSP 也引用其中共享部分。
import copy
import hashlib
import json
import math
from typing import Dict, List, Tuple
from .parameters import read_parameter

# unit, scope, integer, zero_allowed, required_value
PARAMETERS = {
    "superlanes": ("count", "chip", True, False, True),
    "lanes_per_superlane": ("count", "superlane", True, False, True),
    "spare_superlanes": ("count", "chip", True, True, False),
    "hemispheres": ("count", "chip", True, False, True),
    "planes_per_hemisphere": ("count", "hemisphere", True, False, True),
    "plane_dim": ("count", "plane", True, False, True),
    "supercell_dim": ("count", "supercell", True, False, True),
    "vxm_alus_per_lane": ("count", "lane", True, False, True),
    "mem_slices_per_hemisphere": ("count", "hemisphere", True, False, True),
    "mem_tiles_per_slice": ("count", "slice", True, False, True),
    "memory_tile_bytes": ("byte", "memory_tile", True, False, True),
    "banks_per_memory_tile": ("count", "memory_tile", True, False, True),
    "usable_mem_bytes": ("byte", "chip", True, False, True),
    "physical_sram_bytes": ("byte", "chip", True, False, False),
    "local_sram_bytes": ("byte", "chip", True, True, False),
    "sr_coordinates": ("count", "superlane", True, False, True),
    "streams_per_direction": ("count", "chip_stream_direction", True, False, True),
    "stream_width_bytes": ("byte", "chip_stream_cross_section", True, False, True),
    "clock_ghz": ("GHz", "chip", False, False, False),
    "nominal_int8_tops": ("TOPS", "chip", False, False, False),
    "mem_read_bpc": ("byte/cycle", "slice", False, False, False),
    "mem_write_bpc": ("byte/cycle", "slice", False, False, False),
    "mem_shared_bpc": ("byte/cycle", "slice", False, False, False),
    "install_bpc": ("byte/cycle", "hemisphere", False, False, False),
    "install_first_cycles": ("cycle", "hemisphere", True, False, False),
    "mxm_first_cycles": ("cycle", "hemisphere", True, False, False),
    "mxm_row_ii": ("cycle", "hemisphere", True, False, False),
    "row_io_latency_cycles": ("cycle", "hemisphere", True, True, False),
    "vxm_add_ops_per_cycle": ("op/cycle", "allocation", False, False, False),
    "vxm_add_first_cycles": ("cycle", "allocation", True, False, False),
    "nonlinear_ops_per_cycle": ("op/cycle", "chip", False, False, False),
    "offchip_capacity_bytes": ("byte", "chip", True, True, False),
    "offchip_bandwidth_bpc": ("byte/cycle", "chip", False, False, False),
    "interchip_bandwidth_bpc": ("byte/cycle", "chip", False, False, False),
}


class TSPSpec:
    def __init__(self, config):
        required = {"schema_version", "model", "profile_id", "description", "parameters", "memory_tile_capacity", "supported_formats"}
        if not isinstance(config, dict) or set(config) != required:
            raise ValueError(f"TSP reference schema 2 requires {sorted(required)}")
        if config["schema_version"] != 2 or config["model"] != "tsp_analytical_reference":
            raise ValueError("Expected TSP reference schema 2")
        if config["memory_tile_capacity"] != "includes_both_banks":
            raise ValueError("memory_tile_capacity must include both banks; never multiply capacity by bank count")
        if config["supported_formats"] != ["INT8"]:
            raise ValueError("Reference execution currently supports INT8 only")
        if not isinstance(config["profile_id"], str) or not config["profile_id"].strip():
            raise ValueError("profile_id required")
        if set(config["parameters"]) != set(PARAMETERS):
            raise ValueError(f"Parameter keys mismatch: {sorted(set(PARAMETERS) ^ set(config['parameters']))}")
        self.config = copy.deepcopy(config)
        self.parameters = self.config["parameters"]
        self.values = {}
        for name, (unit, scope, integer, zero, required) in PARAMETERS.items():
            self.values[name] = read_parameter(self.parameters[name], name, unit, scope,
                                              integer=integer, zero=zero, required=required)
        v = self.values
        self.n_superlanes = v["superlanes"]
        self.n_lanes_per_superlane = v["lanes_per_superlane"]
        self.n_spare_superlanes = v["spare_superlanes"]
        self.n_hemispheres = v["hemispheres"]
        self.n_mxm_planes_per_hemisphere = v["planes_per_hemisphere"]
        self.mxm_plane_dim = v["plane_dim"]
        self.supercell_dim = v["supercell_dim"]
        self.vxm_alus_per_lane = v["vxm_alus_per_lane"]
        self.n_mem_slices_per_hemisphere = v["mem_slices_per_hemisphere"]
        self.mem_n_slices = self.n_hemispheres * self.n_mem_slices_per_hemisphere
        self.mem_total_bytes = self.global_mem_sram_bytes = v["usable_mem_bytes"]
        self.physical_total_sram_bytes = v["physical_sram_bytes"]
        self.reserved_local_sram_bytes = v["local_sram_bytes"]
        self.n_streams_per_direction = v["streams_per_direction"]
        self.stream_width_bytes = v["stream_width_bytes"]
        self.clock_freq_ghz = v["clock_ghz"]
        capacity = self.mem_n_slices * v["mem_tiles_per_slice"] * v["memory_tile_bytes"]
        # memory_tile_bytes 已包含 tile 内所有 bank，容量总和不能再乘 bank 数。
        if self.mem_total_bytes > capacity or self.mem_total_bytes % self.mem_n_slices:
            raise ValueError("usable_mem_bytes must fit declared memory tiles and divide equally among slices")
        if v["mem_tiles_per_slice"] != self.n_superlanes or self.mxm_plane_dim != self.total_lanes:
            raise ValueError("TSP plane/lane and slice/superlane geometry is inconsistent")
        if self.mxm_plane_dim % self.supercell_dim or self.supercell_dim != self.n_lanes_per_superlane:
            raise ValueError("Supercell geometry is inconsistent")
        if v["memory_tile_bytes"] % v["banks_per_memory_tile"]:
            raise ValueError("Memory tile capacity must partition into its declared banks")
        if v["vxm_add_ops_per_cycle"] is not None and v["vxm_add_ops_per_cycle"] > self.total_vxm_alus:
            raise ValueError("Single-task INT32 add budget exceeds one operation per configured VXM ALU per cycle")
        physical, local = self.physical_total_sram_bytes, self.reserved_local_sram_bytes
        if physical is not None and physical < self.mem_total_bytes + (local or 0):
            raise ValueError("Physical SRAM cannot be smaller than known usable/local SRAM")
        self.profile_id = config["profile_id"]
        self.reference_id = self.profile_id + ":" + hashlib.sha256(
            json.dumps(self.config, sort_keys=True, allow_nan=False).encode()).hexdigest()[:16]

    @classmethod
    def from_config(cls, config):
        return cls(config)

    @property
    def total_lanes(self):
        # SuperLane/lane 是整片通道坐标，不随左右两个 hemisphere 再复制一份。
        return self.n_superlanes * self.n_lanes_per_superlane

    @property
    def total_vxm_alus(self):
        return self.total_lanes * self.vxm_alus_per_lane

    @property
    def total_mxm_planes_chip(self):
        return self.n_hemispheres * self.n_mxm_planes_per_hemisphere

    @property
    def supercells_k_per_plane(self):
        return self.mxm_plane_dim // self.supercell_dim

    supercells_n_per_plane = supercells_k_per_plane

    @property
    def supercells_per_plane(self):
        return self.supercells_k_per_plane ** 2

    @property
    def total_supercells_chip(self):
        return self.supercells_per_plane * self.total_mxm_planes_chip

    @property
    def mem_slice_capacity_bytes(self):
        return self.mem_total_bytes // self.mem_n_slices

    @property
    def stream_bw_per_direction_bytes_per_cycle(self):
        return self.n_streams_per_direction * self.stream_width_bytes

    @property
    def global_mem_sram_mib(self):
        return self.mem_total_bytes / 2**20

    @property
    def physical_total_sram_mib(self):
        return None if self.physical_total_sram_bytes is None else self.physical_total_sram_bytes / 2**20

    @property
    def reserved_local_sram_mib(self):
        return None if self.reserved_local_sram_bytes is None else self.reserved_local_sram_bytes / 2**20

    def logical_mxm_planes_per_hemisphere(self, data_format):
        if data_format != "INT8":
            raise ValueError("Reference MXM supports INT8 only")
        return self.n_mxm_planes_per_hemisphere

    def tile_n_capacity_per_hemisphere(self, data_format):
        # 本侧 plane 沿 N 负责不同列，所以 N 容量可相加，K 高度仍是一个 plane 的高度。
        return self.logical_mxm_planes_per_hemisphere(data_format) * self.mxm_plane_dim

    def peak_tops(self, data_format="INT8"):
        self.logical_mxm_planes_per_hemisphere(data_format)
        return None if self.clock_freq_ghz is None else self.total_mxm_planes_chip * self.mxm_plane_dim**2 * 2 * self.clock_freq_ghz / 1000

    def area_inventory(self):
        return {"scope": "chip", "mxm_planes": self.total_mxm_planes_chip,
                "supercells": self.total_supercells_chip, "vxm_alus": self.total_vxm_alus,
                "mem_slices": self.mem_n_slices, "active_memory_tiles": self.mem_n_slices * self.values["mem_tiles_per_slice"],
                "global_mem_sram_mib": self.global_mem_sram_mib,
                "local_sram_mib": self.reserved_local_sram_mib,
                "physical_sram_mib": self.physical_total_sram_mib,
                "spare_superlanes": self.n_spare_superlanes,
                "sxm_physical_units": None, "stream_physical_endpoints": None,
                "clock_domains": None, "control_domains": None}

    def derived(self):
        # 派生值一并携带依赖及其核验状态，计算出一个数字不意味着假设已经变成实测。
        definitions = {
            "lanes_chip": (self.total_lanes, "lane", "superlanes * lanes_per_superlane", ["superlanes", "lanes_per_superlane"]),
            "planes_chip": (self.total_mxm_planes_chip, "plane", "hemispheres * planes_per_hemisphere", ["hemispheres", "planes_per_hemisphere"]),
            "vxm_alus_chip": (self.total_vxm_alus, "ALU", "lanes_chip * vxm_alus_per_lane", ["superlanes", "lanes_per_superlane", "vxm_alus_per_lane"]),
            "mem_slice_bytes": (self.mem_slice_capacity_bytes, "byte", "usable_mem_bytes / (hemispheres * mem_slices_per_hemisphere)", ["usable_mem_bytes", "hemispheres", "mem_slices_per_hemisphere"]),
            "memory_tile_capacity_chip_bytes": (self.mem_n_slices * self.values["mem_tiles_per_slice"] * self.values["memory_tile_bytes"], "byte", "hemispheres * slices_per_hemisphere * tiles_per_slice * memory_tile_bytes; bank count not multiplied", ["hemispheres", "mem_slices_per_hemisphere", "mem_tiles_per_slice", "memory_tile_bytes"]),
            "stream_bpc_per_direction": (self.stream_bw_per_direction_bytes_per_cycle, "byte/cycle", "streams_per_direction * stream_width_bytes", ["streams_per_direction", "stream_width_bytes"]),
            "geometric_int8_tops_chip": (self.peak_tops(), "TOPS", "planes_chip * plane_dim**2 * 2 * clock_ghz / 1000", ["hemispheres", "planes_per_hemisphere", "plane_dim", "clock_ghz"]),
        }
        return {key: {"value": value, "unit": unit, "formula": formula, "dependencies": deps,
                      "dependency_status": {d: self.parameters[d]["status"] for d in deps},
                      "dependency_verification": {d: self.parameters[d]["verification"] for d in deps}}
                for key, (value, unit, formula, deps) in definitions.items()}

    def __repr__(self):
        return f"TSPSpec(reference={self.profile_id}, MEM={self.global_mem_sram_mib:g} MiB, planes/chip={self.total_mxm_planes_chip})"

    def validate_sr_coordinate(self, coordinate):
        # SR 仅校验坐标范围，不把坐标总数解释成额外传输跳数或固定延迟。
        if type(coordinate) is not int or not 0 <= coordinate < self.values["sr_coordinates"]:
            raise ValueError("SR coordinate is outside the declared coordinate space")
        return coordinate

    def lane_range_for_superlane(self, superlane: int) -> Tuple[int, int]:
        if type(superlane) is not int or not 0 <= superlane < self.n_superlanes:
            raise ValueError("Superlane index is outside active inventory")
        lane_start = superlane * self.n_lanes_per_superlane
        lane_end = lane_start + self.n_lanes_per_superlane - 1
        return lane_start, lane_end

    def describe_supercell_usage(
        self,
        k_actual: int,
        n_actual: int,
        data_format: str,
        hemisphere: int = 0,
    ) -> Dict[str, object]:
        if type(hemisphere) is not int or not 0 <= hemisphere < self.n_hemispheres:
            raise ValueError("Hemisphere index outside chip inventory")
        if (type(k_actual) is not int or type(n_actual) is not int or
                not 1 <= k_actual <= self.mxm_plane_dim or
                not 1 <= n_actual <= self.tile_n_capacity_per_hemisphere(data_format)):
            raise ValueError("Tile exceeds the single-hemisphere MXM geometry")
        active_supercell_rows = min(
            self.supercells_k_per_plane,
            max(1, math.ceil(max(0, k_actual) / self.supercell_dim)),
        )
        active_supercell_cols_total = min(
            self.logical_mxm_planes_per_hemisphere(data_format) * self.supercells_n_per_plane,
            max(1, math.ceil(max(0, n_actual) / self.supercell_dim)),
        )

        planes: List[Dict[str, object]] = []
        # 以下描述几何覆盖范围，供阅读结构与映射；并不执行逐 supercell 的周期事件。
        remaining_cols = active_supercell_cols_total
        logical_planes = self.logical_mxm_planes_per_hemisphere(data_format)

        for plane_idx in range(logical_planes):
            # N 列块依次铺到多个 plane；相同 K 行贯穿各 plane，N 列互不重叠。
            plane_col_blocks = min(self.supercells_n_per_plane, remaining_cols)
            base_n_block = plane_idx * self.supercells_n_per_plane
            rows: List[Dict[str, object]] = []

            for superlane in range(self.supercells_k_per_plane):
                lane_start, lane_end = self.lane_range_for_superlane(superlane)
                row_active = superlane < active_supercell_rows
                k_start = superlane * self.supercell_dim
                k_end = min(
                    self.mxm_plane_dim - 1,
                    k_start + self.supercell_dim - 1,
                )

                supercells = []
                for col_idx in range(self.supercells_n_per_plane):
                    n_block_idx = base_n_block + col_idx
                    col_active = col_idx < plane_col_blocks
                    active = row_active and col_active
                    n_start = n_block_idx * self.supercell_dim
                    n_end = n_start + self.supercell_dim - 1
                    supercells.append({
                        "hemisphere": hemisphere,
                        "plane": plane_idx,
                        "superlane": superlane,
                        "lane_start": lane_start,
                        "lane_end": lane_end,
                        "supercell_row": superlane,
                        "supercell_col": col_idx,
                        "active": active,
                        "k_start": k_start,
                        "k_end": min(k_end, k_actual - 1) if active else k_end,
                        "n_start": n_start,
                        "n_end": min(n_end, n_actual - 1) if active else n_end,
                        "reduction_source_row": max(0, superlane - 1),
                        "reduction_sink_row": min(
                            active_supercell_rows - 1,
                            superlane + 1,
                        ) if active_supercell_rows > 0 else 0,
                    })

                rows.append({
                    "superlane": superlane,
                    "lane_start": lane_start,
                    "lane_end": lane_end,
                    "active": row_active,
                    "k_start": k_start,
                    "k_end": min(k_end, k_actual - 1) if row_active else k_end,
                    "supercells": supercells,
                })

            planes.append({
                "plane": plane_idx,
                "active": plane_col_blocks > 0,
                "active_supercell_cols": plane_col_blocks,
                "rows": rows,
            })
            remaining_cols -= plane_col_blocks

        return {
            "hemisphere": hemisphere,
            "logical_planes": logical_planes,
            "active_supercell_rows": active_supercell_rows,
            "active_supercell_cols_total": active_supercell_cols_total,
            "supercell_dim": self.supercell_dim,
            "reduction_chain_depth": active_supercell_rows,
            "planes": planes,
        }
