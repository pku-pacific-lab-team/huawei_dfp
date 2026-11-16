"""
TSP Hardware Specification

Paper-faithful single-chip Groq TSP baseline:
- 20 superlanes, 16 lanes per superlane
- 2 mirrored hemispheres
- 2 MXM planes per hemisphere (4 total)
- 320x320 MXM plane geometry
- explicit 20x20 supercell mesh per plane
- 88 MEM slices (44 per hemisphere), 220 MiB total
"""
import math
from typing import Dict, List, Optional, Tuple


class TSPSpec:
    """Native Groq TSP hardware specification."""

    def __init__(
        self,
        n_superlanes: int = 20,
        n_lanes_per_superlane: int = 16,
        n_spare_superlanes: int = 1,
        n_mxm_planes_per_hemisphere: int = 2,
        mxm_plane_dim: int = 320,
        mxm_pipeline_depth: int = 20,
        weight_install_cycles: int = 40,
        mem_total_bytes: int = 220 * 1024 * 1024,
        mem_n_slices: int = 88,
        mem_bw_tbps: float = 80.0,
        physical_total_sram_bytes: Optional[int] = None,
        reserved_local_sram_bytes: int = 0,
        vxm_alus_per_lane: int = 16,
        clock_freq_ghz: float = 0.9,
        n_mem_slices_per_hemisphere: int = 44,
        n_hemispheres: int = 2,
        n_streams_per_direction: int = 32,
        stream_width_bytes: int = 320,
        act_mem_start_slice: int = 0,
        wt_mem_start_slice: int = 28,
    ):
        self.n_superlanes = n_superlanes
        self.n_lanes_per_superlane = n_lanes_per_superlane
        self.n_spare_superlanes = n_spare_superlanes
        self.n_mxm_planes_per_hemisphere = n_mxm_planes_per_hemisphere
        self.mxm_plane_dim = mxm_plane_dim
        self.mxm_pipeline_depth = mxm_pipeline_depth
        self.weight_install_cycles = weight_install_cycles
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
        self.vxm_alus_per_lane = vxm_alus_per_lane
        self.clock_freq_ghz = clock_freq_ghz
        self.n_mem_slices_per_hemisphere = n_mem_slices_per_hemisphere
        self.n_hemispheres = n_hemispheres
        self.n_streams_per_direction = n_streams_per_direction
        self.stream_width_bytes = stream_width_bytes
        self.act_mem_start_slice = act_mem_start_slice
        self.wt_mem_start_slice = wt_mem_start_slice

    @property
    def supercell_dim(self) -> int:
        return 16

    @property
    def supercells_k_per_plane(self) -> int:
        return self.mxm_plane_dim // self.supercell_dim

    @property
    def supercells_n_per_plane(self) -> int:
        return self.mxm_plane_dim // self.supercell_dim

    @property
    def total_lanes(self) -> int:
        return self.n_superlanes * self.n_lanes_per_superlane

    @property
    def total_vxm_alus(self) -> int:
        return self.total_lanes * self.vxm_alus_per_lane

    @property
    def vxm_alus_per_hemisphere(self) -> int:
        return self.total_vxm_alus // self.n_hemispheres

    @property
    def total_mxm_planes_chip(self) -> int:
        return self.n_mxm_planes_per_hemisphere * self.n_hemispheres

    @property
    def mxm_weights_per_plane(self) -> int:
        return self.mxm_plane_dim * self.mxm_plane_dim

    @property
    def supercells_per_plane(self) -> int:
        return self.supercells_k_per_plane * self.supercells_n_per_plane

    @property
    def total_supercells_chip(self) -> int:
        return self.supercells_per_plane * self.total_mxm_planes_chip

    @property
    def mem_slice_capacity_bytes(self) -> int:
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
    def stagger_cycles(self) -> int:
        return self.n_superlanes - 1

    @property
    def stream_bw_per_direction_bytes_per_cycle(self) -> int:
        return self.n_streams_per_direction * self.stream_width_bytes

    @property
    def weight_install_bytes_per_cycle(self) -> float:
        full_plane_bytes = (
            self.n_mxm_planes_per_hemisphere *
            self.mxm_weights_per_plane
        )
        return full_plane_bytes / self.weight_install_cycles

    def logical_mxm_planes_per_hemisphere(self, data_format: str) -> int:
        if data_format in {"FP16", "BF16", "INT16"}:
            return max(1, self.n_mxm_planes_per_hemisphere // 2)
        return self.n_mxm_planes_per_hemisphere

    def plane_tandem_factor(self, data_format: str) -> int:
        if data_format in {"FP16", "BF16", "INT16"}:
            return 2
        return 1

    def peak_ops_per_cycle_per_hemisphere(self, data_format: str) -> int:
        if data_format in {"FP16", "BF16", "INT16"}:
            return self.logical_mxm_planes_per_hemisphere(data_format) * (
                self.mxm_plane_dim * self.mxm_plane_dim
            )
        return self.n_mxm_planes_per_hemisphere * (
            self.mxm_plane_dim * self.mxm_plane_dim * 2
        )

    def peak_ops_per_cycle_total(self, data_format: str) -> int:
        return self.peak_ops_per_cycle_per_hemisphere(data_format) * self.n_hemispheres

    def peak_tops(self, data_format: str) -> float:
        return (
            self.peak_ops_per_cycle_total(data_format) *
            self.clock_freq_ghz / 1000.0
        )

    def tile_n_capacity_per_hemisphere(self, data_format: str) -> int:
        return self.logical_mxm_planes_per_hemisphere(data_format) * self.mxm_plane_dim

    def lane_range_for_superlane(self, superlane: int) -> Tuple[int, int]:
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
        active_supercell_rows = min(
            self.supercells_k_per_plane,
            max(1, math.ceil(max(0, k_actual) / self.supercell_dim)),
        )
        active_supercell_cols_total = min(
            self.logical_mxm_planes_per_hemisphere(data_format) * self.supercells_n_per_plane,
            max(1, math.ceil(max(0, n_actual) / self.supercell_dim)),
        )

        planes: List[Dict[str, object]] = []
        remaining_cols = active_supercell_cols_total
        logical_planes = self.logical_mxm_planes_per_hemisphere(data_format)

        for plane_idx in range(logical_planes):
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

    def summarize_topology(
        self,
        data_format: str,
        k_actual: int,
        n_actual: int,
    ) -> Dict[str, int]:
        usage = self.describe_supercell_usage(
            k_actual=k_actual,
            n_actual=n_actual,
            data_format=data_format,
            hemisphere=0,
        )
        active_logical_planes = min(
            self.logical_mxm_planes_per_hemisphere(data_format),
            max(1, -(-max(0, n_actual) // self.mxm_plane_dim)),
        )
        return {
            "superlanes": self.n_superlanes,
            "lanes_per_superlane": self.n_lanes_per_superlane,
            "spare_superlanes": self.n_spare_superlanes,
            "mxm_planes_per_hemisphere": self.n_mxm_planes_per_hemisphere,
            "logical_mxm_planes_per_hemisphere": self.logical_mxm_planes_per_hemisphere(data_format),
            "active_logical_planes": active_logical_planes,
            "plane_tandem_factor": self.plane_tandem_factor(data_format),
            "supercells_per_plane": self.supercells_per_plane,
            "supercells_per_hemisphere": self.supercells_per_plane * self.n_mxm_planes_per_hemisphere,
            "total_supercells_chip": self.total_supercells_chip,
            "mxm_plane_dim": self.mxm_plane_dim,
            "supercell_dim": self.supercell_dim,
            "active_supercell_rows": usage["active_supercell_rows"],
            "active_supercell_cols_total": usage["active_supercell_cols_total"],
            "planes": usage["planes"],
        }

    def hops_vxm_to_mem(self, mem_slice: int) -> int:
        return mem_slice + 1

    def hops_mem_to_vxm(self, mem_slice: int) -> int:
        return mem_slice + 1

    def hops_mem_to_mxm(self, mem_slice: int) -> int:
        remaining_mem = (self.n_mem_slices_per_hemisphere - 1) - mem_slice
        return remaining_mem + 2

    def hops_mxm_to_vxm(self) -> int:
        return 1 + 1 + self.n_mem_slices_per_hemisphere

    def avg_hops_act_to_mxm(self, n_act_slices: int) -> float:
        if n_act_slices <= 0:
            return float(self.hops_mem_to_mxm(self.act_mem_start_slice))
        total = 0.0
        for offset in range(n_act_slices):
            total += self.hops_mem_to_mxm(self.act_mem_start_slice + offset)
        return total / n_act_slices

    def avg_hops_wt_to_mxm(self, n_wt_slices: int) -> float:
        if n_wt_slices <= 0:
            return float(self.hops_mem_to_mxm(self.wt_mem_start_slice))
        total = 0.0
        for offset in range(n_wt_slices):
            slice_idx = min(
                self.wt_mem_start_slice + offset,
                self.n_mem_slices_per_hemisphere - 1,
            )
            total += self.hops_mem_to_mxm(slice_idx)
        return total / n_wt_slices

    def avg_hops_vxm_to_result_mem(
        self,
        n_result_slices: int,
        result_mem_start: int = 0,
    ) -> float:
        if n_result_slices <= 0:
            return float(self.hops_vxm_to_mem(result_mem_start))
        total = 0.0
        for offset in range(n_result_slices):
            total += self.hops_vxm_to_mem(result_mem_start + offset)
        return total / n_result_slices

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
            "mxm_planes": self.total_mxm_planes_chip,
            "mem_slices": self.mem_n_slices,
            "sxm_units": self.n_hemispheres,
            "vxm_alus": self.total_vxm_alus,
            "stream_endpoints": self.n_streams_per_direction * self.n_hemispheres * 2,
            "clock_domains": self.total_mxm_planes_chip,
            "control_domains": self.total_mxm_planes_chip + self.n_hemispheres,
            "routing_complexity_factor": 1.0,
        }

    @classmethod
    def from_config(cls, hw_config: Dict[str, object]) -> "TSPSpec":
        superlane = hw_config.get("superlane", {})
        mxm = hw_config.get("mxm", {})
        mem = hw_config.get("mem", {})
        sram = hw_config.get("sram", {})
        vxm = hw_config.get("vxm", {})
        layout = hw_config.get("layout", {})
        streams = hw_config.get("streams", {})

        return cls(
            n_superlanes=superlane.get("n_superlanes", 20),
            n_lanes_per_superlane=superlane.get("n_lanes_per_superlane", 16),
            n_spare_superlanes=superlane.get("n_spare_superlanes", 1),
            n_mxm_planes_per_hemisphere=mxm.get("n_planes_per_hemisphere", 2),
            mxm_plane_dim=mxm.get("plane_dim", 320),
            mxm_pipeline_depth=mxm.get("pipeline_depth", 20),
            weight_install_cycles=mxm.get("weight_install_cycles", 40),
            mem_total_bytes=sram.get(
                "global_mem_bytes",
                mem.get("total_bytes", 220 * 1024 * 1024),
            ),
            mem_n_slices=mem.get("n_slices", 88),
            mem_bw_tbps=mem.get("bw_tbps", 80.0),
            physical_total_sram_bytes=sram.get(
                "physical_total_bytes",
                mem.get("total_bytes", 220 * 1024 * 1024),
            ),
            reserved_local_sram_bytes=sram.get("local_reserved_bytes", 0),
            vxm_alus_per_lane=vxm.get("alus_per_lane", 16),
            clock_freq_ghz=hw_config.get("clock_freq_ghz", 0.9),
            n_mem_slices_per_hemisphere=layout.get("n_mem_slices_per_hemisphere", 44),
            n_hemispheres=layout.get("n_hemispheres", 2),
            n_streams_per_direction=streams.get("n_streams_per_direction", 32),
            stream_width_bytes=streams.get("stream_width_bytes", 320),
            act_mem_start_slice=layout.get("act_mem_start_slice", 0),
            wt_mem_start_slice=layout.get("wt_mem_start_slice", 28),
        )

    def __repr__(self) -> str:
        return (
            f"TSPSpec(superlanes={self.n_superlanes}, "
            f"lanes/sl={self.n_lanes_per_superlane}, "
            f"MXM_planes/hem={self.n_mxm_planes_per_hemisphere}, "
            f"MEM={self.mem_total_bytes // (1024 * 1024)}MiB, "
            f"SRAM={self.physical_total_sram_bytes // (1024 * 1024)}MiB, "
            f"VXM_ALUs={self.total_vxm_alus}, "
            f"freq={self.clock_freq_ghz}GHz)"
        )
