"""
CIM Core Simulator

支持 INT8 / MXINT8 / BF16 / FP8 等数据格式的 CIM 仿真器
包含 Pre-alignment 和 Post-processing 的开销建模
"""
import math
import itertools
from typing import Dict, Any, Union, Tuple, Optional

from ..hardware.macro import CIMMacroSpec
from ..hardware.dataformat import FORMAT_SPECS, get_format_overheads
from .tiling import TilingEngine


class CIMCoreSimulator:
    """
    CIM Core 仿真器

    仿真数字存内计算核心的性能和能耗，支持：
    - 多种数据格式 (INT8/BF16/MXINT8/FP8/MXFP8_M4)
    - 两级存储层次 (L0/L1)
    - 五级流水线仿真
    - 精细化能耗建模
    """

    def __init__(
        self,
        n_macros: int,
        macro_spec: CIMMacroSpec,
        size_l1_input_buffer_bytes: int,
        size_l1_output_buffer_bytes: int,
        size_l0_input_buffer_bytes: int,
        size_l0_output_buffer_bytes: int,
        bw_noc_to_l1_gbps: float,
        bw_l1_to_l0_gbps: float,
        throughput_vector_unit_gops: float,
        energy_params: Dict[str, float],
        clock_freq_ghz: float = 1.0,
        data_format: str = 'INT8',
        custom_overheads: Optional[Dict[str, float]] = None,
        verbose: bool = True
    ):
        """
        初始化CIM Core仿真器

        Args:
            n_macros: CIM Macro数量
            macro_spec: CIM Macro规格
            size_l1_input_buffer_bytes: L1输入Buffer大小（字节）
            size_l1_output_buffer_bytes: L1输出Buffer大小（字节）
            size_l0_input_buffer_bytes: L0输入Buffer大小（字节）
            size_l0_output_buffer_bytes: L0输出Buffer大小（字节）
            bw_noc_to_l1_gbps: NoC到L1带宽（Gbps）
            bw_l1_to_l0_gbps: L1到L0带宽（Gbps）
            throughput_vector_unit_gops: 向量单元吞吐量（GOPS）
            energy_params: 能量参数字典
            clock_freq_ghz: 时钟频率（GHz），默认1.0
            data_format: 数据格式，默认'INT8'
            custom_overheads: 自定义开销参数
            verbose: 是否打印详细信息
        """
        if verbose:
            print(f"--- 初始化CIM Core (Format: {data_format}) ---")

        self.n_macros = n_macros
        self.macro_spec = macro_spec
        self.size_l1_input = size_l1_input_buffer_bytes
        self.size_l0_input = size_l0_input_buffer_bytes
        self.size_l1_output = size_l1_output_buffer_bytes
        self.size_l0_output = size_l0_output_buffer_bytes
        self.bw_noc_to_l1_gbps = bw_noc_to_l1_gbps
        self.bw_l1_to_l0_gbps = bw_l1_to_l0_gbps
        self.throughput_vector_unit_gops = throughput_vector_unit_gops
        self.clock_freq_ghz = clock_freq_ghz
        self.energy_params = energy_params
        self.data_format = data_format
        self.verbose = verbose

        # --- 获取数据格式规格 ---
        if data_format not in FORMAT_SPECS:
            raise ValueError(f"Unsupported format: {data_format}")

        format_spec = FORMAT_SPECS[data_format]
        self.storage_bytes = format_spec.storage_bytes
        self.compute_bits = 8  # 硬件物理位宽固定为8
        self.effective_compute_bits = format_spec.effective_compute_bits
        self.activity_factor = format_spec.get_activity_factor(self.compute_bits)

        # --- 获取格式相关开销 ---
        self.overheads = get_format_overheads(data_format, custom_overheads)

        # --- 基础参数计算 ---
        self.ops_per_cycle_vector_unit = (
            (self.throughput_vector_unit_gops * 1e9) / (self.clock_freq_ghz * 1e9)
        )
        self.t_load_block_per_macro = self.macro_spec.block_k_dim
        self.t_local_readout = 1
        self.bytes_per_local_readout = (
            self.macro_spec.n_subarrays_per_bank *
            self.macro_spec.block_n_dim *
            self.macro_spec.n_cols_per_bank
        ) / 8

        # --- 初始化Tiling引擎 ---
        self.tiling_engine = TilingEngine(
            macro_spec=macro_spec,
            n_macros=n_macros,
            size_l0_input=size_l0_input_buffer_bytes,
            size_l0_output=size_l0_output_buffer_bytes,
            size_l1_input=size_l1_input_buffer_bytes,
            storage_bytes=self.storage_bytes
        )

        # --- 打印配置信息 ---
        if verbose:
            self._print_config(format_spec)

    def _print_config(self, format_spec) -> None:
        """打印仿真器配置信息"""
        print(f"--- Memory Hierarchy Setup ---")
        print(f"  L1 Buffer (Global): {self.size_l1_input/1024:.1f} KB")
        print(f"--- Data Format Specifics ({self.data_format}) ---")
        print(f"  Storage Precision: {self.storage_bytes:.4f} Bytes/item")
        print(f"  Mantissa/Guard Bits: {format_spec.mantissa_bits} / {format_spec.guard_bits}")
        print(f"  Effective Compute Bits (Latency): {self.effective_compute_bits} Bits")
        print(f"  CIM Energy Activity Factor: {self.activity_factor:.2f}")
        print("---------------------------------------------------\n")

    def _derive_tile_sizes(
        self,
        M: int,
        K: int,
        N: int,
        partition_strategy: Union[str, Tuple[int, int]],
        loop_order: str
    ) -> Tuple[Dict[str, int], Dict[str, int], Tuple[int, int]]:
        """
        计算两级Tile大小（委托给TilingEngine）
        """
        return self.tiling_engine.derive_tile_sizes(M, K, N, partition_strategy)

    def simulate(
        self,
        M: int,
        K: int,
        N: int,
        partition_strategy: Union[str, Tuple[int, int]],
        loop_order: str
    ) -> Dict[str, Any]:
        """
        执行仿真

        Args:
            M: 矩阵M维度
            K: 矩阵K维度
            N: 矩阵N维度
            partition_strategy: 并行划分策略
            loop_order: 循环顺序（如'MKN', 'KMN'等）

        Returns:
            仿真结果字典，包含延迟、能耗、吞吐量等指标
        """
        l0_dims, l1_dims, (pk, pn) = self._derive_tile_sizes(
            M, K, N, partition_strategy, loop_order
        )

        # 初始化流水线时间戳
        t_avail = {
            "NoC": 0,
            "Bus_L1_L0": 0,
            "Pre_Proc": 0,
            "Macro": 0,
            "Post_Proc": 0
        }
        t_l1_space_available = 0

        # 初始化能耗计数器
        energies_pj = {k: 0 for k in [
            "dram_read", "dram_write", "load_X_NoC", "load_W",
            "read_L1", "write_L0", "pre_process",
            "read_L0", "local_readout", "compute_macro",
            "accum_vector_unit", "read_psums", "write_psums"
        ]}

        # 计算带宽（字节/周期）
        bw_noc_bpc = (self.bw_noc_to_l1_gbps * 1e9) / (self.clock_freq_ghz * 1e9) / 8
        bw_bus_bpc = (self.bw_l1_to_l0_gbps * 1e9) / (self.clock_freq_ghz * 1e9) / 8

        # L1缓存标签
        current_l1_input_tag = None
        current_l1_weight_tag = None

        # Tile步长
        step_m_l1, step_k_l1, step_n_l1 = l1_dims['m'], l1_dims['k'], l1_dims['n']
        step_m_l0, step_k_l0, step_n_l0 = l0_dims['m'], l0_dims['k'], l0_dims['n']

        # 生成L1级循环迭代器
        ranges_l1 = {
            'M': range(0, M, step_m_l1),
            'K': range(0, K, step_k_l1),
            'N': range(0, N, step_n_l1)
        }
        outer_iter = itertools.product(
            ranges_l1[loop_order[0]],
            ranges_l1[loop_order[1]],
            ranges_l1[loop_order[2]]
        )

        # === 外层循环: L1 Tile ===
        for l1_indices in outer_iter:
            idx_map_l1 = {
                loop_order[0]: l1_indices[0],
                loop_order[1]: l1_indices[1],
                loop_order[2]: l1_indices[2]
            }
            m_l1_start = idx_map_l1['M']
            k_l1_start = idx_map_l1['K']
            n_l1_start = idx_map_l1['N']

            m_l1_actual = min(step_m_l1, M - m_l1_start)
            k_l1_actual = min(step_k_l1, K - k_l1_start)
            n_l1_actual = min(step_n_l1, N - n_l1_start)

            # === LEVEL 1: DRAM -> L1 Traffic (With Backpressure) ===
            input_tag = (m_l1_start, k_l1_start)
            input_miss = (current_l1_input_tag != input_tag)
            weight_tag = (k_l1_start, n_l1_start)
            weight_miss = (current_l1_weight_tag != weight_tag)

            if input_miss or weight_miss:
                start_noc = max(t_avail["NoC"], t_l1_space_available)
                current_noc_time = start_noc

                if input_miss:
                    bytes_x = m_l1_actual * k_l1_actual * self.storage_bytes
                    dur = math.ceil(bytes_x / bw_noc_bpc)
                    current_noc_time += dur
                    energies_pj["dram_read"] += bytes_x * self.energy_params["dram_read_power"]
                    energies_pj["load_X_NoC"] += bytes_x * self.energy_params["l1_sram_write_power"]
                    current_l1_input_tag = input_tag

                if weight_miss:
                    bytes_w = k_l1_actual * n_l1_actual * self.storage_bytes
                    dur = math.ceil(bytes_w / bw_noc_bpc)
                    current_noc_time += dur
                    energies_pj["dram_read"] += bytes_w * self.energy_params["dram_read_power"]
                    energies_pj["load_W"] += bytes_w * self.energy_params["sram_write_power"]
                    current_l1_weight_tag = weight_tag

                t_avail["NoC"] = current_noc_time

            # === LEVEL 2: L1 -> L0 Traffic & Compute (Sub Tiles) ===
            r_m_in = range(0, m_l1_actual, step_m_l0)
            r_k_in = range(0, k_l1_actual, step_k_l0)
            r_n_in = range(0, n_l1_actual, step_n_l0)

            for dm, dk, dn in itertools.product(r_m_in, r_k_in, r_n_in):
                m_tile = min(step_m_l0, m_l1_actual - dm)
                k_tile = min(step_k_l0, k_l1_actual - dk)
                n_tile = min(step_n_l0, n_l1_actual - dn)

                # --- [Action 1: L1 -> L0 Transfer] ---
                bytes_x_l0 = m_tile * k_tile * self.storage_bytes
                dur_bus = math.ceil(bytes_x_l0 / bw_bus_bpc)
                start_bus = max(t_avail["NoC"], t_avail["Bus_L1_L0"])
                t_avail["Bus_L1_L0"] = start_bus + dur_bus
                energies_pj["read_L1"] += bytes_x_l0 * self.energy_params["l1_sram_read_power"]
                energies_pj["write_L0"] += bytes_x_l0 * pn * self.energy_params["l0_sram_write_power"]

                # --- [Action 2: Pre-processing with Unified Overheads] ---
                num_inputs_to_pre_proc = m_tile * k_tile
                parallelism = pk * self.macro_spec.block_k_dim
                cycles_pre = math.ceil(
                    num_inputs_to_pre_proc *
                    self.overheads['pre_latency_cycles_per_input'] / parallelism
                )
                start_pre = max(t_avail["Bus_L1_L0"], t_avail["Pre_Proc"])
                t_avail["Pre_Proc"] = start_pre + cycles_pre
                bits_to_pre_proc = bytes_x_l0 * 8
                energies_pj["pre_process"] += bits_to_pre_proc * self.overheads['pre_energy_pj_per_bit']

                # --- [Action 3: Macro Compute with Corrected FP Model] ---
                num_switches = self.macro_spec.n_loc_per_subarray
                num_chunks = math.ceil(self.effective_compute_bits / self.compute_bits)
                dur_per_chunk = num_switches * (self.t_local_readout + m_tile * self.compute_bits)
                dur_macro = dur_per_chunk * num_chunks
                start_macro = max(t_avail["Pre_Proc"], t_avail["Macro"])
                t_avail["Macro"] = start_macro + dur_macro

                energies_pj["read_L0"] += bytes_x_l0 * pn * self.energy_params["l0_sram_read_power"]

                base_readout_energy = (
                    num_switches *
                    self.bytes_per_local_readout *
                    self.energy_params["cim_local_read_power"] *
                    self.n_macros
                )
                energies_pj["local_readout"] += (
                    base_readout_energy * self.activity_factor * num_chunks
                )

                base_compute_energy = (m_tile * k_tile * n_tile) * self.energy_params["tpu_flop_power"]
                ops_equivalent = base_compute_energy / self.energy_params["tpu_flop_power"] * \
                    (self.effective_compute_bits / 8.0)
                energy_compute = ops_equivalent * self.energy_params["tpu_flop_power"]
                energies_pj["compute_macro"] += (
                    energy_compute *
                    self.overheads['macro_energy_scale'] *
                    self.activity_factor
                )

                # === [Action 4: Post-processing with Corrected PSUM Logic] ===
                ops_mac_output = m_tile * n_tile
                reduction_penalty = 1.0 if pk == 1 else 1.5

                # 检查这是否是K维度的第一次计算
                is_first_k_chunk = (k_l1_start + dk == 0)

                # 1. PSUM Read (如果不是第一个K chunk)
                bytes_psum = ops_mac_output * 4  # PSUMs usually 32-bit
                if not is_first_k_chunk:
                    energies_pj["read_psums"] += (
                        bytes_psum * self.energy_params["l0_sram_read_power"]
                    )

                # 2. Accumulation in Vector Unit (延迟和能耗)
                ops_to_accumulate = ops_mac_output
                eff_ops_per_cycle = (
                    self.ops_per_cycle_vector_unit *
                    self.overheads['post_throughput_scale'] / reduction_penalty
                )
                dur_post = math.ceil(ops_to_accumulate / eff_ops_per_cycle) \
                    if eff_ops_per_cycle > 0 else 0
                start_post = max(t_avail["Macro"], t_avail["Post_Proc"])
                t_avail["Post_Proc"] = start_post + dur_post

                base_accum_energy = ops_to_accumulate * self.energy_params["vect_flop_power"]
                energies_pj["accum_vector_unit"] += (
                    base_accum_energy *
                    self.overheads['post_energy_scale'] *
                    reduction_penalty
                )

                # 3. PSUM Write (每次K迭代后都写回)
                energies_pj["write_psums"] += bytes_psum * self.energy_params["l0_sram_write_power"]

            # === [关键更新] Update Space Availability ===
            t_l1_space_available = t_avail["Macro"]

        # === 循环结束: 最终结果写回 (DRAM Write) ===
        output_bytes = 4  # 最终输出总是高精度
        total_out_bytes = M * N * output_bytes
        energies_pj["dram_write"] += total_out_bytes * self.energy_params["dram_write_power"]

        # === 汇总结果 ===
        total_latency = max(t_avail.values())
        total_energy_pj = sum(energies_pj.values())
        total_ops = M * K * N * 2  # MAC operation is 2 ops (multiply and add)
        throughput = (total_ops / total_latency * self.clock_freq_ghz) if total_latency > 0 else 0

        return {
            "Format": self.data_format,
            "L1_Tile_M": l1_dims['m'],
            "L0_Tile_M": l0_dims['m'],
            "latencies": t_avail,
            "energies_pj": energies_pj,
            "total_latency_cycles": total_latency,
            "total_energy_pj": total_energy_pj,
            "throughput_gops": throughput,
            "energy_efficiency_tops_per_watt": (
                (total_ops / 1e12) / (total_energy_pj / 1e12) if total_energy_pj > 0 else 0
            ),
            "bottleneck": max(t_avail, key=t_avail.get)
        }
