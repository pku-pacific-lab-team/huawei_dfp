"""
TSP-CIM Hardware Specification (V2)

定义 TSP-CIM 混合架构的硬件规格，含物理布局模型：
- Groq TSP 功能切片空间布局 (VXM | MEM[0..43] | SXM | CIM | C2C × 2 hemispheres)
- Stream register hop 延迟建模
- MEM 切片分配策略 (activation vs weight)
- CIM Macro 替换 MXM 矩阵乘单元
"""
from typing import Optional

from .macro import CIMMacroSpec


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
    ):
        self.n_superlanes = n_superlanes
        self.n_lanes_per_superlane = n_lanes_per_superlane
        self.n_macros_per_superlane = n_macros_per_superlane
        self.macro_spec = macro_spec

        # MEM
        self.mem_total_bytes = mem_total_bytes
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
        return self.macro_spec.block_k_dim

    @property
    def chip_block_n(self) -> int:
        return self.n_macros_per_superlane * self.macro_spec.block_n_dim

    @property
    def stagger_cycles(self) -> int:
        """20 superlane 阶梯执行的额外延迟"""
        return self.n_superlanes - 1  # 19

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
        vxm = hw_config.get("vxm", {})
        layout = hw_config.get("layout", {})
        streams = hw_config.get("streams", {})

        macro_spec = CIMMacroSpec.from_dict(macro_cfg)

        return cls(
            macro_spec=macro_spec,
            n_superlanes=sl.get("n_superlanes", 20),
            n_lanes_per_superlane=sl.get("n_lanes_per_superlane", 16),
            n_macros_per_superlane=sl.get("n_macros_per_superlane", 4),
            mem_total_bytes=mem.get("total_bytes", 220 * 1024 * 1024),
            mem_n_slices=mem.get("n_slices", 88),
            mem_bw_tbps=mem.get("bw_tbps", 80.0),
            stream_bw_tbps=mem.get("stream_bw_tbps", 20.0),
            vxm_alus_per_lane=vxm.get("alus_per_lane", 16),
            clock_freq_ghz=hw_config.get("clock_freq_ghz", 0.9),
            weight_prog_bw_gbps=hw_config.get("weight_prog_bw_gbps", 64.0),
            n_mem_slices_per_hemisphere=layout.get("n_mem_slices_per_hemisphere", 44),
            n_hemispheres=layout.get("n_hemispheres", 2),
            n_streams_per_direction=streams.get("n_streams_per_direction", 32),
            stream_width_bytes=streams.get("stream_width_bytes", 320),
            act_mem_start_slice=layout.get("act_mem_start_slice", 0),
            wt_mem_start_slice=layout.get("wt_mem_start_slice", 28),
        )

    def __repr__(self) -> str:
        return (
            f"TSPCIMSpec(superlanes={self.n_superlanes}, "
            f"lanes/sl={self.n_lanes_per_superlane}, "
            f"macros/sl={self.n_macros_per_superlane}, "
            f"total_macros={self.total_macros}, "
            f"MEM={self.mem_total_bytes // (1024*1024)}MB, "
            f"VXM_ALUs={self.total_vxm_alus}, "
            f"freq={self.clock_freq_ghz}GHz, "
            f"hops_CIM→VXM={self.hops_cim_to_vxm()})"
        )
