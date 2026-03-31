"""
TSP-CIM Hardware Specification (V2)

定义 TSP-CIM 混合架构的硬件规格，含物理布局模型：
- Groq TSP 功能切片空间布局 (VXM | MEM[0..43] | SXM | CIM | C2C × 2 hemispheres)
- Stream register hop 延迟建模
- MEM 切片分配策略 (activation vs weight)
- 显式 macro/superlane/lane 拓扑
- CIM Macro 替换 MXM 矩阵乘单元
"""
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .macro import CIMMacroSpec


@dataclass(frozen=True)
class MacroPlacement:
    """单个 CIM macro 在单个 hemisphere 内的逻辑放置。"""
    hemisphere: int
    superlane: int
    macro_idx: int
    macro_id: int
    k_group: int
    n_group: int
    lane_start: int
    lane_end: int
    n_start: int
    n_end: int
    k_start: int
    k_end: int
    dispatch_hops_from_sxm: int


class TSPCIMSpec:
    """
    TSP-CIM 硬件规格类（含物理布局）

    物理布局 (每半球，从中心向外):
        VXM | SRF | MEM[0] | SRF | MEM[1] | ... | MEM[43] | SRF | SXM | SRF | CIM | SRF | C2C

    - MEM[0] 最靠近 VXM（芯片中心）
    - MEM[43] 最靠近 SXM/CIM（芯片边缘）
    - 每对相邻功能切片间有 SRF 站点，1 hop = 1 cycle transit delay
    - 两个半球 (East/West) 完全对称，独立运行
    """

    def __init__(
        self,
        macro_spec: CIMMacroSpec,
        # Superlane 结构
        n_superlanes: int = 20,
        n_lanes_per_superlane: int = 16,
        n_macros_per_superlane: int = 4,
        # MEM
        mem_total_bytes: int = 220 * 1024 * 1024,
        mem_n_slices: int = 88,
        mem_bw_tbps: float = 80.0,
        stream_bw_tbps: float = 20.0,
        physical_total_sram_bytes: Optional[int] = None,
        reserved_local_sram_bytes: int = 0,
        # VXM
        vxm_alus_per_lane: int = 16,
        # Clock & programming
        clock_freq_ghz: float = 0.9,
        weight_prog_bw_gbps: float = 64.0,
        # Physical layout
        n_mem_slices_per_hemisphere: int = 44,
        n_hemispheres: int = 2,
        # Stream parameters
        n_streams_per_direction: int = 32,
        stream_width_bytes: int = 320,
        # MEM allocation strategy
        act_mem_start_slice: int = 0,
        wt_mem_start_slice: int = 28,
        # CIM fabric topology
        macro_arrangement: str = "linear_daisy_chain",
        macro_link_hops: int = 1,
        macro_k_groups_per_superlane: int = 1,
    ):
        self.n_superlanes = n_superlanes
        self.n_lanes_per_superlane = n_lanes_per_superlane
        self.n_macros_per_superlane = n_macros_per_superlane
        self.macro_spec = macro_spec

        # MEM
        self.global_mem_sram_bytes = mem_total_bytes
        self.reserved_local_sram_bytes = reserved_local_sram_bytes
        self.physical_total_sram_bytes = (
            physical_total_sram_bytes
            if physical_total_sram_bytes is not None
            else (self.global_mem_sram_bytes + self.reserved_local_sram_bytes)
        )
        self.mem_total_bytes = self.global_mem_sram_bytes
        self.mem_n_slices = mem_n_slices
        self.mem_bw_tbps = mem_bw_tbps
        self.stream_bw_tbps = stream_bw_tbps

        # VXM
        self.vxm_alus_per_lane = vxm_alus_per_lane

        # Clock & programming
        self.clock_freq_ghz = clock_freq_ghz
        self.weight_prog_bw_gbps = weight_prog_bw_gbps

        # Physical layout
        self.n_mem_slices_per_hemisphere = n_mem_slices_per_hemisphere
        self.n_hemispheres = n_hemispheres

        # Stream parameters
        self.n_streams_per_direction = n_streams_per_direction
        self.stream_width_bytes = stream_width_bytes

        # MEM allocation strategy
        self.act_mem_start_slice = act_mem_start_slice
        self.wt_mem_start_slice = wt_mem_start_slice

        # CIM fabric topology
        self.macro_arrangement = macro_arrangement
        self.macro_link_hops = macro_link_hops
        self.macro_k_groups_per_superlane = macro_k_groups_per_superlane

        if self.macro_k_groups_per_superlane <= 0:
            raise ValueError("macro_k_groups_per_superlane must be >= 1")
        if self.n_macros_per_superlane % self.macro_k_groups_per_superlane != 0:
            raise ValueError(
                "n_macros_per_superlane must be divisible by macro_k_groups_per_superlane"
            )

    # ---- Derived parameters ----

    @property
    def total_lanes(self) -> int:
        return self.n_superlanes * self.n_lanes_per_superlane

    @property
    def total_macros(self) -> int:
        return self.n_superlanes * self.n_macros_per_superlane

    @property
    def total_vxm_alus(self) -> int:
        return self.total_lanes * self.vxm_alus_per_lane

    @property
    def chip_block_k(self) -> int:
        return self.macro_k_groups_per_superlane * self.macro_spec.block_k_dim

    @property
    def chip_block_n(self) -> int:
        return self.macro_n_groups_per_superlane * self.macro_spec.block_n_dim

    @property
    def macro_n_groups_per_superlane(self) -> int:
        return self.n_macros_per_superlane // self.macro_k_groups_per_superlane

    @property
    def stagger_cycles(self) -> int:
        """20 superlane 阶梯执行的额外延迟"""
        return self.n_superlanes - 1  # 19

    @property
    def macros_per_hemisphere(self) -> int:
        return self.total_macros

    @property
    def total_macros_chip(self) -> int:
        return self.total_macros * self.n_hemispheres

    @property
    def macro_weight_ratio_vs_supercell(self) -> float:
        """以 MXM 16×16 supercell 为参考的权重容量比。"""
        supercell_weights = 16 * 16
        return (
            (self.macro_spec.block_k_dim * self.macro_spec.block_n_dim) /
            supercell_weights
        )

    @property
    def macro_peak_mac_per_cycle_int8(self) -> float:
        """
        当前 intra-macro INT8 模型下，单个 macro 的稳态峰值 MAC/cycle。
        保持 intra-macro 时序不变，仅用于跨架构口径比较。
        """
        return (
            self.macro_spec.block_k_dim * self.macro_spec.block_n_dim
        ) / (self.macro_spec.n_loc_per_subarray * 8)

    @property
    def macro_peak_ratio_vs_supercell_int8(self) -> float:
        """
        以 MXM supercell ≈ 16×16 = 256 MAC/cycle 为参考的峰值吞吐比。
        """
        return self.macro_peak_mac_per_cycle_int8 / (16 * 16)

    @property
    def supercell_peak_mac_per_cycle_int8(self) -> float:
        return 16 * 16

    @property
    def macros_per_supercell_peak_equivalent_int8(self) -> float:
        ratio = self.macro_peak_ratio_vs_supercell_int8
        return (1.0 / ratio) if ratio > 0 else float("inf")

    @property
    def peak_equivalent_supercells_per_superlane_int8(self) -> float:
        return self.n_macros_per_superlane * self.macro_peak_ratio_vs_supercell_int8

    @property
    def peak_equivalent_supercells_chip_int8(self) -> float:
        return self.total_macros_chip * self.macro_peak_ratio_vs_supercell_int8

    @property
    def effective_n_columns_per_lane(self) -> float:
        """
        inter-macro 级别下，每条 superlane 的共享 lane 带最终覆盖的 N 宽度。
        这里给出“每 lane 对应的有效 N 列宽”作为解释口径。
        """
        return self.chip_block_n / self.n_lanes_per_superlane

    # ---- Bandwidth helpers ----

    @property
    def stream_bw_bytes_per_cycle(self) -> float:
        """Total stream BW (all directions combined)"""
        return (self.stream_bw_tbps * 1e12) / (self.clock_freq_ghz * 1e9)

    @property
    def stream_bw_per_direction_bytes_per_cycle(self) -> float:
        """Stream BW per direction = n_streams × stream_width"""
        return self.n_streams_per_direction * self.stream_width_bytes

    @property
    def mem_bw_bytes_per_cycle(self) -> float:
        return (self.mem_bw_tbps * 1e12) / (self.clock_freq_ghz * 1e9)

    @property
    def weight_prog_bytes_per_cycle(self) -> float:
        return (self.weight_prog_bw_gbps * 1e9) / (self.clock_freq_ghz * 1e9) / 8

    @property
    def mem_slice_capacity_bytes(self) -> int:
        """每个 MEM 切片的容量"""
        return self.mem_total_bytes // self.mem_n_slices

    @property
    def physical_total_sram_mib(self) -> float:
        return self.physical_total_sram_bytes / (1024 * 1024)

    @property
    def global_mem_sram_mib(self) -> float:
        return self.global_mem_sram_bytes / (1024 * 1024)

    @property
    def reserved_local_sram_mib(self) -> float:
        return self.reserved_local_sram_bytes / (1024 * 1024)

    @property
    def fabric_retention_domains(self) -> int:
        return max(1, self.total_macros_chip // 16)

    @property
    def fabric_logic_domains(self) -> int:
        return max(1, (self.total_macros_chip // 32) + (self.n_superlanes * self.n_hemispheres))

    @property
    def clock_domains_total(self) -> int:
        return max(1, self.fabric_logic_domains)

    @property
    def control_domains_total(self) -> int:
        return max(1, self.fabric_logic_domains + self.n_hemispheres)

    # ---- Explicit CIM topology ----

    @property
    def supercell_dim(self) -> int:
        return 16

    @property
    def supercells_k_per_plane(self) -> int:
        return 320 // self.supercell_dim

    @property
    def supercells_n_per_plane(self) -> int:
        return 320 // self.supercell_dim

    @property
    def n_mxm_planes_per_hemisphere(self) -> int:
        return 2

    @property
    def macro_supercell_k_span(self) -> int:
        return max(1, math.ceil(self.macro_spec.block_k_dim / self.supercell_dim))

    @property
    def macro_supercell_n_span(self) -> int:
        return max(1, math.ceil(self.macro_spec.block_n_dim / self.supercell_dim))

    @property
    def macro_rows_per_plane(self) -> int:
        return math.ceil(self.supercells_k_per_plane / self.macro_supercell_k_span)

    @property
    def macro_cols_per_plane(self) -> int:
        return math.ceil(self.supercells_n_per_plane / self.macro_supercell_n_span)

    @property
    def macro_slots_per_plane(self) -> int:
        return self.macro_rows_per_plane * self.macro_cols_per_plane

    @property
    def macro_slots_per_sheet(self) -> int:
        return self.macro_slots_per_plane * self.n_mxm_planes_per_hemisphere

    @property
    def macro_sheets_per_hemisphere(self) -> int:
        return max(1, math.ceil(self.macros_per_hemisphere / self.macro_slots_per_sheet))

    @property
    def macro_sheet_stack_n(self) -> int:
        return max(1, min(self.macro_n_groups_per_superlane, self.macro_sheets_per_hemisphere))

    @property
    def macro_sheet_stack_k(self) -> int:
        return max(1, math.ceil(self.macro_sheets_per_hemisphere / self.macro_sheet_stack_n))

    @property
    def tile_k_capacity(self) -> int:
        return self.macro_sheet_stack_k * self.supercells_k_per_plane * self.supercell_dim

    @property
    def tile_n_capacity(self) -> int:
        return (
            self.macro_sheet_stack_n *
            self.n_mxm_planes_per_hemisphere *
            self.supercells_n_per_plane *
            self.supercell_dim
        )

    def lane_range_for_superlane(self, superlane: int) -> Tuple[int, int]:
        lane_start = superlane * self.n_lanes_per_superlane
        lane_end = lane_start + self.n_lanes_per_superlane - 1
        return lane_start, lane_end

    def describe_macro_usage(
        self,
        k_actual: int,
        n_actual: int,
        hemisphere: int = 0,
    ) -> Dict[str, object]:
        active_sheet_k = min(
            self.macro_sheet_stack_k,
            max(1, math.ceil(max(0, k_actual) / (self.supercells_k_per_plane * self.supercell_dim))),
        )
        active_sheet_n = min(
            self.macro_sheet_stack_n,
            max(1, math.ceil(max(0, n_actual) / (
                self.n_mxm_planes_per_hemisphere *
                self.supercells_n_per_plane *
                self.supercell_dim
            ))),
        )
        active_macro_k_groups = max(1, math.ceil(max(0, k_actual) / self.macro_spec.block_k_dim))
        active_macro_n_groups = max(1, math.ceil(max(0, n_actual) / self.macro_spec.block_n_dim))

        sheets: List[Dict[str, object]] = []
        active_macros = 0
        max_dispatch_hops = 0
        max_reduce_hops = 0
        active_superlane_spans = 0
        active_k_parallel_slots = set()
        active_n_parallel_slots = set()

        macro_id_base = hemisphere * self.macros_per_hemisphere
        macro_id = macro_id_base

        for sheet_k in range(active_sheet_k):
            k_sheet_base = sheet_k * self.supercells_k_per_plane * self.supercell_dim
            k_sheet_actual = min(
                self.supercells_k_per_plane * self.supercell_dim,
                max(0, k_actual - k_sheet_base),
            )
            for sheet_n in range(active_sheet_n):
                n_sheet_base = (
                    sheet_n *
                    self.n_mxm_planes_per_hemisphere *
                    self.supercells_n_per_plane *
                    self.supercell_dim
                )
                sheet_planes = []
                sheet_active_macros = 0

                for plane in range(self.n_mxm_planes_per_hemisphere):
                    plane_n_base = n_sheet_base + plane * self.supercells_n_per_plane * self.supercell_dim
                    plane_n_actual = min(
                        self.supercells_n_per_plane * self.supercell_dim,
                        max(0, n_actual - plane_n_base),
                    )
                    plane_active_supercell_cols = min(
                        self.supercells_n_per_plane,
                        math.ceil(plane_n_actual / self.supercell_dim) if plane_n_actual > 0 else 0,
                    )
                    macro_rows = []

                    for macro_row in range(self.macro_rows_per_plane):
                        supercell_k_start = macro_row * self.macro_supercell_k_span
                        supercell_k_end = min(
                            self.supercells_k_per_plane - 1,
                            supercell_k_start + self.macro_supercell_k_span - 1,
                        )
                        row_k_base = k_sheet_base + supercell_k_start * self.supercell_dim
                        row_k_actual = min(
                            self.macro_spec.block_k_dim,
                            max(0, k_actual - row_k_base),
                        )
                        row_active = row_k_actual > 0
                        row_macros = []

                        for macro_col in range(self.macro_cols_per_plane):
                            supercell_n_start = macro_col * self.macro_supercell_n_span
                            supercell_n_end = min(
                                self.supercells_n_per_plane - 1,
                                supercell_n_start + self.macro_supercell_n_span - 1,
                            )
                            macro_n_base = plane_n_base + supercell_n_start * self.supercell_dim
                            macro_n_actual = min(
                                self.macro_spec.block_n_dim,
                                max(0, n_actual - macro_n_base),
                            )
                            macro_active = row_active and macro_n_actual > 0
                            superlane_start = min(self.n_superlanes - 1, supercell_k_start)
                            superlane_end = min(self.n_superlanes - 1, supercell_k_end)
                            lane_start, _ = self.lane_range_for_superlane(superlane_start)
                            _, lane_end = self.lane_range_for_superlane(superlane_end)
                            dispatch_hops = (
                                sheet_k + sheet_n + macro_row + macro_col
                            ) * self.macro_link_hops
                            reduction_hops = (
                                sheet_k * self.macro_rows_per_plane + macro_row
                            ) * self.macro_link_hops

                            if macro_active:
                                active_macros += 1
                                sheet_active_macros += 1
                                max_dispatch_hops = max(max_dispatch_hops, dispatch_hops)
                                max_reduce_hops = max(max_reduce_hops, reduction_hops)
                                active_k_parallel_slots.add((sheet_k, macro_row))
                                active_n_parallel_slots.add((sheet_n, plane, macro_col))

                            row_macros.append({
                                "macro_id": macro_id,
                                "sheet_k": sheet_k,
                                "sheet_n": sheet_n,
                                "plane": plane,
                                "macro_row": macro_row,
                                "macro_col": macro_col,
                                "active": macro_active,
                                "supercell_k_start": supercell_k_start,
                                "supercell_k_end": supercell_k_end,
                                "supercell_n_start": supercell_n_start,
                                "supercell_n_end": supercell_n_end,
                                "superlane_start": superlane_start,
                                "superlane_end": superlane_end,
                                "lane_start": lane_start,
                                "lane_end": lane_end,
                                "k_start": row_k_base,
                                "k_end": row_k_base + row_k_actual - 1 if macro_active else row_k_base - 1,
                                "k_capacity": self.macro_spec.block_k_dim,
                                "k_actual": row_k_actual if macro_active else 0,
                                "n_start": macro_n_base,
                                "n_end": macro_n_base + macro_n_actual - 1 if macro_active else macro_n_base - 1,
                                "n_capacity": self.macro_spec.block_n_dim,
                                "n_actual": macro_n_actual if macro_active else 0,
                                "dispatch_hops_from_sxm": dispatch_hops,
                                "reduction_hops_to_collector": reduction_hops,
                            })
                            macro_id += 1

                        if row_active:
                            active_superlane_spans = max(active_superlane_spans, superlane_end + 1)
                        macro_rows.append({
                            "macro_row": macro_row,
                            "active": row_active,
                            "supercell_k_start": supercell_k_start,
                            "supercell_k_end": supercell_k_end,
                            "k_actual": row_k_actual if row_active else 0,
                            "macros": row_macros,
                        })

                    sheet_planes.append({
                        "plane": plane,
                        "active": plane_n_actual > 0 and k_sheet_actual > 0,
                        "active_supercell_cols": plane_active_supercell_cols,
                        "macro_rows": macro_rows,
                    })

                sheets.append({
                    "sheet_id": len(sheets),
                    "sheet_k": sheet_k,
                    "sheet_n": sheet_n,
                    "active": sheet_active_macros > 0,
                    "k_base": k_sheet_base,
                    "n_base": n_sheet_base,
                    "planes": sheet_planes,
                })

        active_k_parallel_groups = max(1, len(active_k_parallel_slots))
        active_n_parallel_groups = max(1, len(active_n_parallel_slots))
        serial_k_depth = max(
            1,
            math.ceil(active_macro_k_groups / active_k_parallel_groups),
        )
        serial_n_depth = max(
            1,
            math.ceil(active_macro_n_groups / active_n_parallel_groups),
        )

        return {
            "hemisphere": hemisphere,
            "macro_arrangement": self.macro_arrangement,
            "macro_link_hops": self.macro_link_hops,
            "active_k_groups": active_macro_k_groups,
            "active_n_groups": active_macro_n_groups,
            "active_k_parallel_groups": active_k_parallel_groups,
            "active_n_parallel_groups": active_n_parallel_groups,
            "serial_k_depth": serial_k_depth,
            "serial_n_depth": serial_n_depth,
            "macro_k_groups_per_superlane": self.macro_k_groups_per_superlane,
            "macro_n_groups_per_superlane": self.macro_n_groups_per_superlane,
            "macro_supercell_k_span": self.macro_supercell_k_span,
            "macro_supercell_n_span": self.macro_supercell_n_span,
            "macro_rows_per_plane": self.macro_rows_per_plane,
            "macro_cols_per_plane": self.macro_cols_per_plane,
            "macro_sheet_stack_k": self.macro_sheet_stack_k,
            "macro_sheet_stack_n": self.macro_sheet_stack_n,
            "macro_sheets_per_hemisphere": self.macro_sheets_per_hemisphere,
            "supercell_overlay": (
                f"one macro covers {self.macro_supercell_k_span}x{self.macro_supercell_n_span} "
                f"native 16x16 supercells"
            ),
            "lane_binding": (
                "macro rows span adjacent superlanes; macro columns span adjacent "
                "supercell columns within a plane"
            ),
            "n_partitioning": "macro columns overlay adjacent native supercells along N",
            "k_partitioning": "macro rows overlay adjacent native supercells along K and reduce along the K-chain",
            "active_superlanes": min(
                self.n_superlanes,
                math.ceil(min(max(0, k_actual), self.supercells_k_per_plane * self.supercell_dim) / self.supercell_dim),
            ),
            "active_macros": active_macros,
            "active_superlane_spans": active_superlane_spans,
            "max_dispatch_hops": max_dispatch_hops,
            "max_reduce_hops": max_reduce_hops,
            "sheets": sheets,
        }

    def estimate_activation_route(
        self,
        mem_slice: int,
        usage: Dict[str, object],
        act_bytes: float,
    ) -> Dict[str, float]:
        ext_hops = self.hops_mem_to_cim(mem_slice)
        active_k_groups = max(1, int(usage["active_k_groups"]))
        bytes_per_row = act_bytes / active_k_groups if act_bytes > 0 else 0.0
        internal_byte_hops = 0.0
        max_internal_hops = 0

        for sheet in usage["sheets"]:
            if not sheet["active"]:
                continue
            for plane in sheet["planes"]:
                if not plane["active"]:
                    continue
                for row in plane["macro_rows"]:
                    active_macros = [m for m in row["macros"] if m["active"]]
                    if not active_macros:
                        continue
                    max_internal_hops = max(
                        max_internal_hops,
                        max(m["dispatch_hops_from_sxm"] for m in active_macros),
                    )
                    internal_byte_hops += bytes_per_row * sum(
                        m["dispatch_hops_from_sxm"] for m in active_macros
                    )

        return {
            "external_hops": float(ext_hops),
            "external_byte_hops": act_bytes * ext_hops,
            "max_internal_hops": float(max_internal_hops),
            "max_total_hops": float(ext_hops + max_internal_hops),
            "fabric_injection_bytes": act_bytes,
            "internal_byte_hops": internal_byte_hops,
            "total_byte_hops": act_bytes * ext_hops + internal_byte_hops,
        }

    def estimate_weight_route(
        self,
        mem_slice: int,
        usage: Dict[str, object],
        k_actual: int,
        storage_bytes: float,
    ) -> Dict[str, float]:
        ext_hops = self.hops_mem_to_cim(mem_slice)
        internal_byte_hops = 0.0
        max_internal_hops = 0
        max_macro_bytes = 0.0
        total_weight_bytes = 0.0
        active_macros = 0

        for sheet in usage["sheets"]:
            for plane in sheet["planes"]:
                for row in plane["macro_rows"]:
                    for macro in row["macros"]:
                        if not macro["active"]:
                            continue
                        active_macros += 1
                        macro_weight_bytes = macro["k_actual"] * macro["n_actual"] * storage_bytes
                        total_weight_bytes += macro_weight_bytes
                        max_macro_bytes = max(max_macro_bytes, macro_weight_bytes)
                        max_internal_hops = max(max_internal_hops, macro["dispatch_hops_from_sxm"])
                        internal_byte_hops += macro_weight_bytes * macro["dispatch_hops_from_sxm"]

        return {
            "external_hops": float(ext_hops),
            "external_byte_hops": total_weight_bytes * ext_hops,
            "max_internal_hops": float(max_internal_hops),
            "max_total_hops": float(ext_hops + max_internal_hops),
            "fabric_injection_bytes": total_weight_bytes,
            "internal_byte_hops": internal_byte_hops,
            "total_byte_hops": total_weight_bytes * ext_hops + internal_byte_hops,
            "max_macro_bytes": max_macro_bytes,
            "active_macros": float(active_macros),
            "total_weight_bytes": total_weight_bytes,
        }

    def estimate_result_route(
        self,
        usage: Dict[str, object],
        m_actual: int,
        psum_bytes: int = 4,
    ) -> Dict[str, float]:
        ext_hops = self.hops_cim_to_vxm()
        internal_byte_hops = 0.0
        max_internal_hops = 0
        total_partial_result_bytes = 0.0
        reduced_result_bytes = 0.0

        for sheet in usage["sheets"]:
            sheet_result_bytes = 0.0
            for plane in sheet["planes"]:
                collector_n = 0.0
                for row in plane["macro_rows"]:
                    row_n = 0.0
                    for macro in row["macros"]:
                        if not macro["active"]:
                            continue
                        macro_result_bytes = m_actual * macro["n_actual"] * psum_bytes
                        total_partial_result_bytes += macro_result_bytes
                        row_n += macro["n_actual"]
                        max_internal_hops = max(max_internal_hops, macro["reduction_hops_to_collector"])
                        internal_byte_hops += (
                            macro_result_bytes * macro["reduction_hops_to_collector"]
                        )
                    collector_n = max(collector_n, row_n)
                sheet_result_bytes += m_actual * collector_n * psum_bytes
            reduced_result_bytes += sheet_result_bytes

        return {
            "external_hops": float(ext_hops),
            "external_byte_hops": reduced_result_bytes * ext_hops,
            "max_internal_hops": float(max_internal_hops),
            "max_total_hops": float(ext_hops + max_internal_hops),
            "internal_byte_hops": internal_byte_hops,
            "reduction_bytes": max(0.0, total_partial_result_bytes - reduced_result_bytes),
            "total_byte_hops": reduced_result_bytes * ext_hops + internal_byte_hops,
            "total_result_bytes": reduced_result_bytes,
            "total_partial_result_bytes": total_partial_result_bytes,
        }

    def summarize_topology(
        self,
        k_actual: Optional[int] = None,
        n_actual: Optional[int] = None,
    ) -> Dict[str, object]:
        if k_actual is None:
            k_actual = self.tile_k_capacity
        if n_actual is None:
            n_actual = self.tile_n_capacity
        usage = self.describe_macro_usage(k_actual, n_actual, hemisphere=0)
        return {
            "macro_arrangement": self.macro_arrangement,
            "macro_link_hops": self.macro_link_hops,
            "macro_k_groups_per_superlane": self.macro_k_groups_per_superlane,
            "macro_n_groups_per_superlane": self.macro_n_groups_per_superlane,
            "macros_per_superlane": self.n_macros_per_superlane,
            "macros_per_hemisphere": self.macros_per_hemisphere,
            "lanes_per_superlane": self.n_lanes_per_superlane,
            "active_k_groups": usage["active_k_groups"],
            "active_n_groups": usage["active_n_groups"],
            "active_k_parallel_groups": usage["active_k_parallel_groups"],
            "active_n_parallel_groups": usage["active_n_parallel_groups"],
            "serial_k_depth": usage["serial_k_depth"],
            "serial_n_depth": usage["serial_n_depth"],
            "active_superlanes": usage["active_superlanes"],
            "active_macros": usage["active_macros"],
            "max_chain_hops": usage["max_dispatch_hops"],
            "macro_supercell_k_span": self.macro_supercell_k_span,
            "macro_supercell_n_span": self.macro_supercell_n_span,
            "macro_rows_per_plane": self.macro_rows_per_plane,
            "macro_cols_per_plane": self.macro_cols_per_plane,
            "macro_sheets_per_hemisphere": self.macro_sheets_per_hemisphere,
            "macro_sheet_stack_k": self.macro_sheet_stack_k,
            "macro_sheet_stack_n": self.macro_sheet_stack_n,
            "tile_k_capacity": self.tile_k_capacity,
            "tile_n_capacity": self.tile_n_capacity,
            "sheets": usage["sheets"],
        }

    def sram_summary(self) -> Dict[str, float]:
        return {
            "physical_total_sram_mib": self.physical_total_sram_mib,
            "global_mem_sram_mib": self.global_mem_sram_mib,
            "local_reserved_sram_mib": self.reserved_local_sram_mib,
            "mem_slice_capacity_mib": self.mem_slice_capacity_bytes / (1024 * 1024),
        }

    def area_inventory(self) -> Dict[str, float]:
        return {
            "global_mem_sram_mib": self.global_mem_sram_mib,
            "local_sram_mib": self.reserved_local_sram_mib,
            "cim_macros": self.total_macros_chip,
            "vxm_alus": self.total_vxm_alus,
            "stream_endpoints": self.n_streams_per_direction * self.n_hemispheres * 2,
            "fabric_domains": self.fabric_retention_domains,
            "clock_domains": self.clock_domains_total,
            "control_domains": self.control_domains_total,
            "routing_complexity_factor": 1.2,
        }

    # ---- Physical layout: hop calculations ----

    def hops_vxm_to_mem(self, mem_slice: int) -> int:
        """VXM 到 MEM[mem_slice] 的 stream hop 数
        MEM[0] 最近 VXM (1 hop), MEM[43] 最远 (44 hops)
        """
        return mem_slice + 1

    def hops_mem_to_vxm(self, mem_slice: int) -> int:
        """MEM[mem_slice] 到 VXM 的 stream hop 数 (inward)"""
        return mem_slice + 1

    def hops_mem_to_cim(self, mem_slice: int) -> int:
        """MEM[mem_slice] 到 CIM 的 stream hop 数 (outward)
        经过剩余 MEM 切片 + SXM(1 hop) + CIM 入口(1 hop)
        """
        remaining_mem = (self.n_mem_slices_per_hemisphere - 1) - mem_slice
        return remaining_mem + 2  # +1 SXM, +1 CIM

    def hops_cim_to_vxm(self) -> int:
        """CIM 到 VXM 的 stream hop 数 (inward, chaining 路径)
        CIM 出口(1) + SXM(1) + 所有 MEM 切片(44) = 46
        """
        return 1 + 1 + self.n_mem_slices_per_hemisphere

    def hops_cim_to_mem(self, mem_slice: int) -> int:
        """CIM 到 MEM[mem_slice] 的 stream hop 数 (inward)"""
        remaining_mem = (self.n_mem_slices_per_hemisphere - 1) - mem_slice
        return 1 + 1 + remaining_mem  # CIM出口 + SXM + 剩余MEM

    def avg_hops_act_to_cim(self, n_act_slices: int) -> float:
        """activation 从 MEM 到 CIM 的平均 hop 数"""
        if n_act_slices <= 0:
            return self.hops_mem_to_cim(self.act_mem_start_slice)
        total = 0
        for i in range(n_act_slices):
            slice_idx = self.act_mem_start_slice + i
            total += self.hops_mem_to_cim(slice_idx)
        return total / n_act_slices

    def avg_hops_wt_to_cim(self, n_wt_slices: int) -> float:
        """weight 从 MEM 到 CIM 的平均 hop 数"""
        if n_wt_slices <= 0:
            return self.hops_mem_to_cim(self.wt_mem_start_slice)
        total = 0
        for i in range(n_wt_slices):
            slice_idx = self.wt_mem_start_slice + i
            slice_idx = min(slice_idx, self.n_mem_slices_per_hemisphere - 1)
            total += self.hops_mem_to_cim(slice_idx)
        return total / n_wt_slices

    def avg_hops_vxm_to_result_mem(self, n_result_slices: int,
                                    result_mem_start: int = 0) -> float:
        """VXM 到结果 MEM 的平均 hop 数"""
        if n_result_slices <= 0:
            return self.hops_vxm_to_mem(result_mem_start)
        total = 0
        for i in range(n_result_slices):
            slice_idx = result_mem_start + i
            total += self.hops_vxm_to_mem(slice_idx)
        return total / n_result_slices

    # ---- MEM slice allocation ----

    def calc_mem_slices_needed(self, data_bytes: int) -> int:
        """计算存储 data_bytes 需要多少 MEM 切片"""
        cap = self.mem_slice_capacity_bytes
        if cap <= 0:
            return 1
        return max(1, -(-data_bytes // cap))  # ceil division

    # ---- Factory ----

    @classmethod
    def from_config(cls, hw_config: dict) -> "TSPCIMSpec":
        sl = hw_config.get("superlane", {})
        macro_cfg = hw_config.get("macro", {})
        mem = hw_config.get("mem", {})
        sram = hw_config.get("sram", {})
        vxm = hw_config.get("vxm", {})
        layout = hw_config.get("layout", {})
        streams = hw_config.get("streams", {})
        fabric = hw_config.get("cim_fabric", {})

        macro_spec = CIMMacroSpec.from_dict(macro_cfg)

        return cls(
            macro_spec=macro_spec,
            n_superlanes=sl.get("n_superlanes", 20),
            n_lanes_per_superlane=sl.get("n_lanes_per_superlane", 16),
            n_macros_per_superlane=sl.get("n_macros_per_superlane", 4),
            mem_total_bytes=sram.get(
                "global_mem_bytes",
                mem.get("total_bytes", 220 * 1024 * 1024),
            ),
            mem_n_slices=mem.get("n_slices", 88),
            mem_bw_tbps=mem.get("bw_tbps", 80.0),
            stream_bw_tbps=mem.get("stream_bw_tbps", 20.0),
            physical_total_sram_bytes=sram.get(
                "physical_total_bytes",
                mem.get("total_bytes", 220 * 1024 * 1024),
            ),
            reserved_local_sram_bytes=sram.get("local_reserved_bytes", 0),
            vxm_alus_per_lane=vxm.get("alus_per_lane", 16),
            clock_freq_ghz=hw_config.get("clock_freq_ghz", 0.9),
            weight_prog_bw_gbps=hw_config.get("weight_prog_bw_gbps", 64.0),
            n_mem_slices_per_hemisphere=layout.get("n_mem_slices_per_hemisphere", 44),
            n_hemispheres=layout.get("n_hemispheres", 2),
            n_streams_per_direction=streams.get("n_streams_per_direction", 32),
            stream_width_bytes=streams.get("stream_width_bytes", 320),
            act_mem_start_slice=layout.get("act_mem_start_slice", 0),
            wt_mem_start_slice=layout.get("wt_mem_start_slice", 28),
            macro_arrangement=fabric.get("macro_arrangement", "grid_broadcast_reduce"),
            macro_link_hops=fabric.get("macro_link_hops", 1),
            macro_k_groups_per_superlane=fabric.get("macro_k_groups_per_superlane", 1),
        )

    def __repr__(self) -> str:
        return (
            f"TSPCIMSpec(superlanes={self.n_superlanes}, "
            f"lanes/sl={self.n_lanes_per_superlane}, "
            f"macros/sl={self.n_macros_per_superlane}, "
            f"total_macros={self.total_macros}, "
            f"MEM={self.mem_total_bytes // (1024*1024)}MB, "
            f"SRAM={self.physical_total_sram_bytes // (1024*1024)}MB, "
            f"VXM_ALUs={self.total_vxm_alus}, "
            f"freq={self.clock_freq_ghz}GHz, "
            f"hops_CIM→VXM={self.hops_cim_to_vxm()})"
        )
