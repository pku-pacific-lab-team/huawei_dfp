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
        bw_l0_to_l1_output_gbps: Optional[float] = None,
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
            bw_l0_to_l1_output_gbps: L0→L1 Output独立带宽（Gbps），默认与bw_l1_to_l0_gbps相同
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
        self.bw_l0_to_l1_output_gbps = bw_l0_to_l1_output_gbps or bw_l1_to_l0_gbps
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
            "Weight_Prog": 0,       # 权重重编程 (L1 → SRAM bitcell)
            "Bus_L1_L0": 0,
            "Pre_Proc": 0,
            "Macro": 0,
            "Post_Proc": 0,
            "L1_Output": 0,         # L0 output → L1 output 传输
        }
        t_l1_space_available = 0

        # 初始化能耗计数器
        energies_pj = {k: 0 for k in [
            "dram_read", "dram_write", "load_X_NoC", "load_W",
            "weight_prog",          # 权重重编程能耗
            "read_L1", "write_L0", "pre_process",
            "read_L0", "local_readout", "compute_macro",
            "accum_vector_unit", "read_psums", "write_psums",
            "write_l1_output",      # L0 output → L1 output 写回能耗
        ]}

        # 计算带宽（字节/周期）
        bw_noc_bpc = (self.bw_noc_to_l1_gbps * 1e9) / (self.clock_freq_ghz * 1e9) / 8
        bw_bus_bpc = (self.bw_l1_to_l0_gbps * 1e9) / (self.clock_freq_ghz * 1e9) / 8
        bw_l0_to_l1_bpc = (self.bw_l0_to_l1_output_gbps * 1e9) / (self.clock_freq_ghz * 1e9) / 8

        # 权重重编程带宽
        weight_prog_bw = self.energy_params.get("weight_prog_bw_gbps", 0)
        weight_prog_bpc = (weight_prog_bw * 1e9 / (self.clock_freq_ghz * 1e9) / 8
                           ) if weight_prog_bw > 0 else 0

        # [P2-1] L1 多 tag 缓存追踪
        l1_input_tile_space = {}   # {(m, k): bytes}
        l1_weight_tile_space = {}  # {(k, n): bytes}
        l1_used_bytes = 0

        # [P0-2] L0 output PSUM 空间管理
        # per-macro: PSUM bytes = m_tile * block_n_dim * 4
        l0_output_tile_space = {}  # {(abs_m, abs_n): (t_write_complete, bytes_per_macro)}
        C_tile_K_progress = {}     # {(abs_m, abs_n): set(abs_k, ...)}

        # 统计信息
        l0_to_l1_output_stats = {
            'total_evict_count': 0, 'total_evict_bytes': 0,
            'tile_evict_count': {}, 'tile_evict_bytes': {},
        }
        l1_cache_stats = {
            'input_hits': 0, 'input_misses': 0,
            'weight_hits': 0, 'weight_misses': 0,
        }
        l0_input_cache_stats = {'hits': 0, 'misses': 0}

        # Tile步长
        step_m_l1, step_k_l1, step_n_l1 = l1_dims['m'], l1_dims['k'], l1_dims['n']
        step_m_l0, step_k_l0, step_n_l0 = l0_dims['m'], l0_dims['k'], l0_dims['n']

        # 预计算所有会被访问的绝对K位置（用于判断C tile是否完成）
        expected_K_set = set()
        for k_l1 in range(0, K, step_k_l1):
            k_l1_act = min(step_k_l1, K - k_l1)
            for dk in range(0, k_l1_act, step_k_l0):
                expected_K_set.add(k_l1 + dk)

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
            idx_map_l1 = {loop_order[i]: l1_indices[i] for i in range(3)}
            m_l1_start = idx_map_l1['M']
            k_l1_start = idx_map_l1['K']
            n_l1_start = idx_map_l1['N']

            m_l1_actual = min(step_m_l1, M - m_l1_start)
            k_l1_actual = min(step_k_l1, K - k_l1_start)
            n_l1_actual = min(step_n_l1, N - n_l1_start)

            # === LEVEL 1: DRAM -> L1 Traffic ===
            input_tag = (m_l1_start, k_l1_start)
            weight_tag = (k_l1_start, n_l1_start)

            # [P2-1] L1 多 tag 缓存命中检测
            input_miss = (input_tag not in l1_input_tile_space)
            weight_miss = (weight_tag not in l1_weight_tile_space)

            if input_miss:
                l1_cache_stats['input_misses'] += 1
            else:
                l1_cache_stats['input_hits'] += 1
            if weight_miss:
                l1_cache_stats['weight_misses'] += 1
            else:
                l1_cache_stats['weight_hits'] += 1

            if input_miss or weight_miss:
                start_noc = max(t_avail["NoC"], t_l1_space_available)
                current_noc_time = start_noc

                if input_miss:
                    bytes_x = m_l1_actual * k_l1_actual * self.storage_bytes
                    # L1 空间不足时驱逐最早的 input tile
                    while l1_used_bytes + bytes_x > self.size_l1_input and l1_input_tile_space:
                        oldest = next(iter(l1_input_tile_space))
                        l1_used_bytes -= l1_input_tile_space.pop(oldest)
                    dur = math.ceil(bytes_x / bw_noc_bpc)
                    current_noc_time += dur
                    energies_pj["dram_read"] += bytes_x * self.energy_params["dram_read_power"]
                    energies_pj["load_X_NoC"] += bytes_x * self.energy_params["l1_sram_write_power"]
                    l1_input_tile_space[input_tag] = bytes_x
                    l1_used_bytes += bytes_x

                if weight_miss:
                    bytes_w = k_l1_actual * n_l1_actual * self.storage_bytes
                    # L1 空间不足时驱逐最早的 weight tile
                    while l1_used_bytes + bytes_w > self.size_l1_input and l1_weight_tile_space:
                        oldest = next(iter(l1_weight_tile_space))
                        l1_used_bytes -= l1_weight_tile_space.pop(oldest)
                    dur = math.ceil(bytes_w / bw_noc_bpc)
                    current_noc_time += dur
                    energies_pj["dram_read"] += bytes_w * self.energy_params["dram_read_power"]
                    energies_pj["load_W"] += bytes_w * self.energy_params["sram_write_power"]
                    l1_weight_tile_space[weight_tag] = bytes_w
                    l1_used_bytes += bytes_w

                t_avail["NoC"] = current_noc_time

            # [P1-2] 权重重编程: L1 → SRAM bitcell 通过 Weight I/O
            if weight_miss and weight_prog_bpc > 0:
                bytes_per_macro_prog = (self.macro_spec.block_k_dim *
                                        self.macro_spec.block_n_dim *
                                        self.storage_bytes)
                dur_prog = math.ceil(bytes_per_macro_prog / weight_prog_bpc)
                start_prog = max(t_avail["NoC"], t_avail["Weight_Prog"])
                t_avail["Weight_Prog"] = start_prog + dur_prog
                weight_prog_power = self.energy_params.get("weight_prog_power", 0)
                energies_pj["weight_prog"] += (bytes_per_macro_prog *
                                               self.n_macros * weight_prog_power)

            # === LEVEL 2: L1 -> L0 Traffic & Compute (Sub Tiles) ===
            r_m_in = range(0, m_l1_actual, step_m_l0)
            r_k_in = range(0, k_l1_actual, step_k_l0)
            r_n_in = range(0, n_l1_actual, step_n_l0)

            # [P2-2] 内层循环尊重 loop_order
            ranges_l0 = {'M': r_m_in, 'K': r_k_in, 'N': r_n_in}
            inner_iter = itertools.product(
                ranges_l0[loop_order[0]],
                ranges_l0[loop_order[1]],
                ranges_l0[loop_order[2]]
            )

            # [P3] L0 输入数据复用追踪（每个 L1 tile 重置）
            l0_input_current_tag = None

            for l0_indices in inner_iter:
                l0_idx_map = {loop_order[i]: l0_indices[i] for i in range(3)}
                dm = l0_idx_map['M']
                dk = l0_idx_map['K']
                dn = l0_idx_map['N']

                m_tile = min(step_m_l0, m_l1_actual - dm)
                k_tile = min(step_k_l0, k_l1_actual - dk)
                n_tile = min(step_n_l0, n_l1_actual - dn)

                # --- [Action 1: L1 -> L0 Transfer] ---
                # [P3] 检查 L0 输入数据是否可复用
                l0_input_tag = (dm, dk)
                l0_input_hit = (l0_input_current_tag == l0_input_tag)
                bytes_x_l0 = m_tile * k_tile * self.storage_bytes

                if l0_input_hit:
                    # 数据已在 L0 中，跳过传输
                    l0_input_cache_stats['hits'] += 1
                else:
                    dur_bus = math.ceil(bytes_x_l0 / bw_bus_bpc)
                    start_bus = max(t_avail["NoC"], t_avail["Bus_L1_L0"])
                    t_avail["Bus_L1_L0"] = start_bus + dur_bus
                    energies_pj["read_L1"] += (bytes_x_l0 *
                                               self.energy_params["l1_sram_read_power"])
                    energies_pj["write_L0"] += (bytes_x_l0 * pn *
                                                self.energy_params["l0_sram_write_power"])
                    l0_input_current_tag = l0_input_tag
                    l0_input_cache_stats['misses'] += 1

                # --- [Action 2: Pre-processing] ---
                num_inputs_to_pre_proc = m_tile * k_tile
                parallelism = pk * self.macro_spec.block_k_dim
                cycles_pre = math.ceil(
                    num_inputs_to_pre_proc *
                    self.overheads['pre_latency_cycles_per_input'] / parallelism
                )
                start_pre = max(t_avail["Bus_L1_L0"], t_avail["Pre_Proc"])
                t_avail["Pre_Proc"] = start_pre + cycles_pre
                bits_to_pre_proc = bytes_x_l0 * 8
                energies_pj["pre_process"] += (bits_to_pre_proc *
                                               self.overheads['pre_energy_pj_per_bit'])

                # --- [Action 3: Macro Compute] ---
                num_switches = self.macro_spec.n_loc_per_subarray
                num_chunks = math.ceil(self.effective_compute_bits / self.compute_bits)
                dur_per_chunk = num_switches * (self.t_local_readout +
                                                m_tile * self.compute_bits)
                dur_macro = dur_per_chunk * num_chunks
                # [P1-2] Macro 必须等待权重编程完成
                start_macro = max(t_avail["Pre_Proc"], t_avail["Macro"],
                                  t_avail["Weight_Prog"])
                t_avail["Macro"] = start_macro + dur_macro

                energies_pj["read_L0"] += (bytes_x_l0 * pn *
                                           self.energy_params["l0_sram_read_power"])

                base_readout_energy = (
                    num_switches *
                    self.bytes_per_local_readout *
                    self.energy_params["cim_local_read_power"] *
                    self.n_macros
                )
                energies_pj["local_readout"] += (
                    base_readout_energy * self.activity_factor * num_chunks
                )

                base_compute_energy = ((m_tile * k_tile * n_tile) *
                                       self.energy_params["tpu_flop_power"])
                ops_equivalent = (base_compute_energy /
                                  self.energy_params["tpu_flop_power"] *
                                  (self.effective_compute_bits / 8.0))
                energy_compute = ops_equivalent * self.energy_params["tpu_flop_power"]
                energies_pj["compute_macro"] += (
                    energy_compute *
                    self.overheads['macro_energy_scale'] *
                    self.activity_factor
                )

                # === [Action 4: Post-processing + PSUM 空间管理] ===
                # 绝对 C tile 标识
                abs_m = m_l1_start + dm
                abs_n = n_l1_start + dn
                abs_k = k_l1_start + dk
                C_tile_tag = (abs_m, abs_n)

                if C_tile_tag not in C_tile_K_progress:
                    C_tile_K_progress[C_tile_tag] = set()

                # 判断是否是该 C tile 的第一次 K 迭代
                has_previous_K = len(C_tile_K_progress[C_tile_tag]) > 0

                ops_mac_output = m_tile * n_tile
                reduction_penalty = 1.0 if pk == 1 else 1.5
                bytes_psum_total = ops_mac_output * 4  # 所有 macro 合计
                psum_bytes_macro = m_tile * self.macro_spec.block_n_dim * 4  # 单 macro

                # [P0-2] PSUM 读取：从 L0 output 或 L1 output
                if has_previous_K:
                    if C_tile_tag in l0_output_tile_space:
                        # PSUM 仍在 L0 output 中
                        energies_pj["read_psums"] += (
                            bytes_psum_total *
                            self.energy_params["l0_sram_read_power"]
                        )
                    else:
                        # PSUM 已被 evict 到 L1 output，需从 L1 读回
                        dur_reload = math.ceil(bytes_psum_total / bw_l0_to_l1_bpc)
                        start_reload = max(t_avail["Macro"], t_avail["L1_Output"])
                        t_avail["L1_Output"] = start_reload + dur_reload
                        energies_pj["read_psums"] += (
                            bytes_psum_total *
                            self.energy_params["l1_sram_read_power"]
                        )
                        energies_pj["write_L0"] += (
                            bytes_psum_total *
                            self.energy_params["l0_sram_write_power"]
                        )

                # 累加（Vector Unit）
                ops_to_accumulate = ops_mac_output
                eff_ops_per_cycle = (
                    self.ops_per_cycle_vector_unit *
                    self.overheads['post_throughput_scale'] / reduction_penalty
                )
                dur_post = (math.ceil(ops_to_accumulate / eff_ops_per_cycle)
                            if eff_ops_per_cycle > 0 else 0)
                start_post = max(t_avail["Macro"], t_avail["Post_Proc"])
                t_avail["Post_Proc"] = start_post + dur_post

                base_accum_energy = (ops_to_accumulate *
                                     self.energy_params["vect_flop_power"])
                energies_pj["accum_vector_unit"] += (
                    base_accum_energy *
                    self.overheads['post_energy_scale'] *
                    reduction_penalty
                )

                # [P0-2] L0 output 空间管理：写入 PSUM 前检查空间
                if C_tile_tag not in l0_output_tile_space:
                    # 需要分配新空间，可能需要驱逐现有 tile
                    for evict_tag in [t for t in l0_output_tile_space
                                      if t != C_tile_tag]:
                        evict_write_time, evict_bytes = l0_output_tile_space[evict_tag]
                        # 归约后只有 pn 个 macro 有独立输出
                        evict_total = evict_bytes * pn

                        start_evict = max(t_avail["Post_Proc"],
                                          t_avail["L1_Output"],
                                          evict_write_time)
                        dur_evict = math.ceil(evict_total / bw_l0_to_l1_bpc)
                        t_avail["L1_Output"] = start_evict + dur_evict

                        energies_pj["read_psums"] += (
                            evict_total *
                            self.energy_params["l0_sram_read_power"]
                        )
                        energies_pj["write_l1_output"] += (
                            evict_total *
                            self.energy_params["l1_sram_write_power"]
                        )

                        l0_to_l1_output_stats['total_evict_count'] += 1
                        l0_to_l1_output_stats['total_evict_bytes'] += evict_total
                        if evict_tag not in l0_to_l1_output_stats['tile_evict_count']:
                            l0_to_l1_output_stats['tile_evict_count'][evict_tag] = 0
                            l0_to_l1_output_stats['tile_evict_bytes'][evict_tag] = 0
                        l0_to_l1_output_stats['tile_evict_count'][evict_tag] += 1
                        l0_to_l1_output_stats['tile_evict_bytes'][evict_tag] += evict_total

                        del l0_output_tile_space[evict_tag]
                        break  # L0 output 只能放一个 tile，驱逐一个即够

                # 写 PSUM 到 L0 output
                energies_pj["write_psums"] += (bytes_psum_total *
                                               self.energy_params["l0_sram_write_power"])
                l0_output_tile_space[C_tile_tag] = (t_avail["Post_Proc"],
                                                    psum_bytes_macro)

                # 记录 K 进度
                C_tile_K_progress[C_tile_tag].add(abs_k)

                # [P1-1] 检查 C tile 是否完成所有 K 维度
                is_complete = expected_K_set.issubset(
                    C_tile_K_progress[C_tile_tag])
                if is_complete and C_tile_tag in l0_output_tile_space:
                    # 完成的 C tile 从 L0 output 转移到 L1 output
                    done_write_time, done_bytes = l0_output_tile_space[C_tile_tag]
                    # 归约后只有 pn 个 macro 有独立输出
                    done_total = done_bytes * pn

                    start_transfer = max(t_avail["Post_Proc"],
                                         t_avail["L1_Output"],
                                         done_write_time)
                    dur_transfer = math.ceil(done_total / bw_l0_to_l1_bpc)
                    t_avail["L1_Output"] = start_transfer + dur_transfer

                    energies_pj["read_psums"] += (
                        done_total * self.energy_params["l0_sram_read_power"]
                    )
                    energies_pj["write_l1_output"] += (
                        done_total * self.energy_params["l1_sram_write_power"]
                    )

                    l0_to_l1_output_stats['total_evict_count'] += 1
                    l0_to_l1_output_stats['total_evict_bytes'] += done_total
                    if C_tile_tag not in l0_to_l1_output_stats['tile_evict_count']:
                        l0_to_l1_output_stats['tile_evict_count'][C_tile_tag] = 0
                        l0_to_l1_output_stats['tile_evict_bytes'][C_tile_tag] = 0
                    l0_to_l1_output_stats['tile_evict_count'][C_tile_tag] += 1
                    l0_to_l1_output_stats['tile_evict_bytes'][C_tile_tag] += done_total

                    del l0_output_tile_space[C_tile_tag]

            # === [关键更新] Update Space Availability ===
            t_l1_space_available = t_avail["Macro"]

        # === 循环结束后：清理 L0 output 中剩余的 tile ===
        for tile_tag, (t_write, t_bytes) in list(l0_output_tile_space.items()):
            # 归约后只有 pn 个 macro 有独立输出
            total_bytes = t_bytes * pn
            start_evict = max(t_avail["Post_Proc"], t_avail["L1_Output"], t_write)
            dur_evict = math.ceil(total_bytes / bw_l0_to_l1_bpc)
            t_avail["L1_Output"] = start_evict + dur_evict

            energies_pj["read_psums"] += (total_bytes *
                                          self.energy_params["l0_sram_read_power"])
            energies_pj["write_l1_output"] += (total_bytes *
                                               self.energy_params["l1_sram_write_power"])

            l0_to_l1_output_stats['total_evict_count'] += 1
            l0_to_l1_output_stats['total_evict_bytes'] += total_bytes
            if tile_tag not in l0_to_l1_output_stats['tile_evict_count']:
                l0_to_l1_output_stats['tile_evict_count'][tile_tag] = 0
                l0_to_l1_output_stats['tile_evict_bytes'][tile_tag] = 0
            l0_to_l1_output_stats['tile_evict_count'][tile_tag] += 1
            l0_to_l1_output_stats['tile_evict_bytes'][tile_tag] += total_bytes
        l0_output_tile_space.clear()

        # === 最终结果写回 (DRAM Write) ===
        output_bytes = 4  # 最终输出总是高精度
        total_out_bytes = M * N * output_bytes
        energies_pj["dram_write"] += total_out_bytes * self.energy_params["dram_write_power"]

        # === 汇总结果 ===
        total_latency = max(t_avail.values())
        total_energy_pj = sum(energies_pj.values())
        total_ops = M * K * N * 2  # MAC operation is 2 ops (multiply and add)
        throughput = ((total_ops / total_latency * self.clock_freq_ghz)
                      if total_latency > 0 else 0)

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
                (total_ops / 1e12) / (total_energy_pj / 1e12)
                if total_energy_pj > 0 else 0
            ),
            "bottleneck": max(t_avail, key=t_avail.get),
            "l0_to_l1_output_stats": {
                "total_evict_count": l0_to_l1_output_stats['total_evict_count'],
                "total_evict_bytes": l0_to_l1_output_stats['total_evict_bytes'],
                "theoretical_min_bytes": M * N * 4,
            },
            "l1_cache_stats": l1_cache_stats,
            "l0_input_cache_stats": l0_input_cache_stats,
        }
