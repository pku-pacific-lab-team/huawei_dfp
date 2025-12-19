import math
import itertools


# --- 1. 硬件规格定义 ---
class CIMMacroSpec:
    def __init__(self, n_banks_internal=32, n_subarrays_per_bank=32, n_loc_per_subarray=4, n_cols_per_bank=8, p_input=8):
        self.n_banks_internal = n_banks_internal
        self.n_subarrays_per_bank = n_subarrays_per_bank
        self.n_loc_per_subarray = n_loc_per_subarray
        self.block_k_dim = n_subarrays_per_bank * n_loc_per_subarray
        self.block_n_dim = n_banks_internal
        self.n_cols_per_bank = n_cols_per_bank
        self.p_input = p_input  # 物理单元支持的输入位宽


# --- 2. 核心仿真器 ---
class CIMCoreSimulator:
    """
    支持 INT8 / MXINT8 / BF16 的 CIM 仿真器
    包含 Pre-alignment 和 Post-processing 的开销建模，支持流水线、tiling、双缓冲。
    """

    def __init__(self,
                 n_macros: int,
                 macro_spec: CIMMacroSpec,
                 size_input_buffer_bytes: int,
                 size_output_buffer_bytes: int,
                 bw_input_buffer_gbps: float,
                 throughput_vector_unit_gops: float,
                 energy_params: dict,
                 clock_freq_ghz: float = 1.0,
                 data_format: str = 'INT8',  # 'INT8', 'BF16', 'MXINT8'
                 overhead_params: dict = None,
                 double_buffer_input: bool = False,
                 double_buffer_weight: bool = False,
                 double_buffer_output: bool = False,
                 size_weight_buffer_bytes: int = None):

        print(f"--- 初始化CIM Core (Format: {data_format}) ---")
        self.n_macros = n_macros
        self.macro_spec = macro_spec
        self.size_input_buffer = size_input_buffer_bytes
        self.size_output_buffer = size_output_buffer_bytes
        self.bw_input_buffer_gbps = bw_input_buffer_gbps
        self.throughput_vector_unit_gops = throughput_vector_unit_gops
        self.clock_freq_ghz = clock_freq_ghz
        self.energy_params = energy_params
        self.data_format = data_format
        self.double_buffer_input = double_buffer_input
        self.double_buffer_weight = double_buffer_weight
        self.double_buffer_output = double_buffer_output
        # 权重buffer容量（可选），若未提供则视为不受约束
        self.size_weight_buffer = size_weight_buffer_bytes if size_weight_buffer_bytes is not None else float('inf')

        # --- A. 设置默认开销参数 ---
        default_overheads = {
            # 预处理/对齐带来的每bit能耗 (Digital Logic Power)
            "pre_align_energy_pj_per_bit": 0.0,
            # BF16/MXINT8 指数/Scale 比较的能量 (更细粒度)
            "pre_align_shift_energy_pj_per_bit": 0.0,
            # 预处理带来的额外延迟 (CIM Array计算前的 Pipeline Bubble 或 Setup time)
            "pre_align_latency_cycles_per_input": 0,
            # 额外的指数比较/移位建模
            "pre_align_exp_compare_cycles": 1,
            "pre_align_shift_avg_bits": 4,
            "pre_align_shift_parallel_inputs": None,  # 默认为 block_k_dim
            # BF16/MX 参数
            "bf16_exp_bits": 8,
            # MXINT8 参数
            "mx_scale_bits": 8,
            "mx_group_size": 32,

            # 后处理(Vector Unit)的能耗放大系数 (相对于INT累加)
            "post_process_energy_scale": 1.0,
            # 后处理的吞吐率折损系数 (例如浮点加法比INT加法慢)
            "post_process_throughput_scale": 1.0,
            # 额外的反量化/去scale 开销
            "post_dequant_cycles_per_output": 0.0,
            "post_dequant_energy_pj_per_bit": 0.0,
            "psum_bitwidth": 32  # PSUM/输出处理假定的位宽（常用 INT32/FP32 防溢出）
        }
        if overhead_params:
            default_overheads.update(overhead_params)
        self.overheads = default_overheads
        # 推导 PSUM 位宽对应的字节数，供 IO 压力估计使用（默认按 psum_bitwidth）
        self.psum_bytes = self.overheads.get("psum_bitwidth", 24) / 8.0

        # --- B. 根据数据格式推导 存储位宽 vs 计算位宽 ---
        if data_format == 'INT8':
            self.storage_bytes = 1.0    # 8-bit 存储
            self.compute_bits = 8       # 8-bit 计算
        elif data_format == 'BF16':
            self.storage_bytes = 2.0    # 16-bit 存储
            self.compute_bits = 8       # 核心只算 8-bit Mantissa
        elif data_format == 'MXINT8':
            # Microscaling: 8-bit 尾数 + 8-bit 共享指数(每32个元素共享)
            # 平均 bytes = 1 + 1/32 ~= 1.03
            self.storage_bytes = 1.03125
            self.compute_bits = 8       # 核心只算 8-bit 整数
        else:
            raise ValueError(f"Unsupported format: {data_format}")

        # --- C. 基础参数计算 ---
        self.bw_input_buffer_bytes_per_cycle = (self.bw_input_buffer_gbps * 1e9) / (self.clock_freq_ghz * 1e9)
        self.ops_per_cycle_vector_unit = (self.throughput_vector_unit_gops * 1e9) / (self.clock_freq_ghz * 1e9)
        # 简化假设：加载一个权重块的时间与 block_k_dim 成正比（可按带宽模型替换）
        self.t_load_block_per_macro = self.macro_spec.block_k_dim
        self.t_local_readout = 1

        self.t_block_compute_per_m = self.macro_spec.n_loc_per_subarray * (self.t_local_readout + self.compute_bits)
        self.bytes_per_local_readout = (self.macro_spec.n_subarrays_per_bank * self.macro_spec.block_n_dim * self.macro_spec.n_cols_per_bank) / 8

        print(f"  Storage Precision: {self.storage_bytes:.4f} Bytes/item (影响带宽)")
        print(f"  Compute Precision: {self.compute_bits} Bits (影响阵列计算延迟)")
        print(f"  Pre-Align Overhead: {self.overheads['pre_align_latency_cycles_per_input']} Cycles/Input, {self.overheads['pre_align_energy_pj_per_bit']} pJ/bit")
        print(f"  Post-Process Scale: Energy x{self.overheads['post_process_energy_scale']}, Throughput x{self.overheads['post_process_throughput_scale']}")
        print("---------------------------------------------------\n")

    def _select_m_tile(
        self,
        M,
        base_m_tile,
        m_tile_from_input,
        m_tile_from_output,
        tiling_policy="max_fit",
        k_tile=None,
        n_tile=None,
        *,
        alignment=16,
        neighbor_window=16,
        allow_base_slack=0.0,
    ):
        """
        常见 m_tile 策略：
        - max_fit: 选 <= base_limit 的最大候选
        - divisible/divisible_max: 优先整除 M 的最大候选，无整除则选 tail 最小
        - min_tail: 碎片最小
        - aligned_down: 对齐到 alignment 的最大候选
        - balanced_io: IO buffer 压力最小
        - auto/auto_score: 多目标打分（IO、tail、tile 数、对齐）
        """
        base_int = max(1, int(base_m_tile))
        base_limit = max(1, int(base_int * (1.0 + allow_base_slack)))

        # 1) 候选集
        candidates = set()
        for x in [base_int, int(m_tile_from_input), int(m_tile_from_output)]:
            candidates.add(max(1, x))
        for div in [2, 4, 8]:
            candidates.add(max(1, base_int // div))
        if alignment and alignment > 1:
            candidates.add(max(1, (base_int // alignment) * alignment))
            candidates.add(max(1, ((base_int // alignment) - 1) * alignment))
        for d in range(-neighbor_window, neighbor_window + 1):
            cand = base_int + d
            if 1 <= cand <= base_limit and M % cand == 0:
                candidates.add(cand)
        for x in [8, 16, 32, 64, 128, 256]:
            if x <= base_limit:
                candidates.add(x)

        # 2) 可行集
        feasible = [c for c in candidates if 1 <= c <= base_limit]
        if not feasible:
            return 1

        # 3) 度量
        def tail_frac(c: int) -> float:
            return (M % c) / c

        def num_tiles(c: int) -> int:
            return (M + c - 1) // c

        def align_penalty(c: int) -> float:
            if not alignment or alignment <= 1:
                return 0.0
            return 0.0 if (c % alignment == 0) else 1.0

        def io_pressure(c: int) -> float:
            """
            归一化后的最大压力：Max(Input, Weight, Output)
            - 输入占用用 storage_bytes
            - 权重占用用 storage_bytes（或独立 weight buffer）
            - 输出占用用 psum_bytes（默认按 psum_bitwidth/8）
            - 双缓冲把 0.5 视为满载，单缓冲 1.0 视为满载
            - 权重占用：bytes_wei_total = k * n * bpe，它是常数项。虽然不随 c 变化，但如果权重本身溢出 (norm_wei > 1.0)，会使得函数始终返回溢出值，从而在 tiling 搜索中淘汰该 (k, n) 组合。
            """
            if not (k_tile and n_tile):
                return 0.0
            in_buf = getattr(self, "size_input_buffer", 0) or 1e-9
            out_buf = getattr(self, "size_output_buffer", 0) or 1e-9
            wei_buf = getattr(self, "size_weight_buffer", float('inf')) or 1e-9

            bpe_in = float(getattr(self, "storage_bytes", 1.0))
            bpe_out = float(getattr(self, "psum_bytes", 4.0))
            bpe_wei = float(getattr(self, "storage_bytes", 1.0))

            bytes_in_total = c * k_tile * bpe_in
            bytes_out_total = c * n_tile * bpe_out
            bytes_wei_total = k_tile * n_tile * bpe_wei  # 权重tile大小，不随 c 变化

            raw_in = bytes_in_total / in_buf
            raw_out = bytes_out_total / out_buf
            raw_wei = bytes_wei_total / wei_buf

            scale_in = 2.0 if self.double_buffer_input else 1.0
            scale_out = 2.0 if self.double_buffer_output else 1.0
            scale_wei = 2.0 if self.double_buffer_weight else 1.0

            norm_in = raw_in * scale_in
            norm_out = raw_out * scale_out
            norm_wei = raw_wei * scale_wei
            return max(norm_in, norm_wei, norm_out)

        # IO 目标：统一倾向接近满载（归一化后 0.9）
        io_target = 0.90

        # 4) 基于 io_pressure 的可行集过滤（避免物理溢出）
        valid_feasible = [c for c in feasible if io_pressure(c) <= 1.001]
        if not valid_feasible:
            return 1

        # 策略
        if tiling_policy in ("divisible", "divisible_max"):
            divs = [c for c in valid_feasible if M % c == 0]
            if divs:
                return max(divs)
            return min(valid_feasible, key=tail_frac)

        if tiling_policy == "min_tail":
            return min(valid_feasible, key=lambda c: (tail_frac(c), -c))

        if tiling_policy == "aligned_down":
            aligned = [c for c in valid_feasible if align_penalty(c) == 0.0]
            if aligned:
                return max(aligned)
            return max(valid_feasible)

        if tiling_policy == "balanced_io":
            return min(valid_feasible, key=lambda c: (abs(io_pressure(c) - io_target), tail_frac(c), -c))

        if tiling_policy in ("auto", "auto_score"):
            w_io = 4.0
            w_tail = 1.0
            w_tiles = 0.2
            w_align = 0.5

            def score(c: int) -> float:
                p = io_pressure(c)
                if p > 1.001:
                    return float('inf')
                return (w_io * abs(p - io_target)
                        + w_tail * tail_frac(c)
                        + w_tiles * (num_tiles(c) / max(1, M) * 10)
                        + w_align * align_penalty(c))

            return min(valid_feasible, key=score)

        # 默认 max_fit
        return max(valid_feasible, key=lambda c: (c, -tail_frac(c), -1.0 * align_penalty(c)))

    def _derive_tile_sizes(self, M, K, N, mapping_strategy, loop_order, tiling_policy="max_fit"):
        # 映射策略：N 并行 / K 并行 / Hybrid 枚举宏分配
        if mapping_strategy == 'N_Parallel':
            n_tile = self.n_macros * self.macro_spec.block_n_dim
            k_tile = self.macro_spec.block_k_dim
        elif mapping_strategy == 'K_Parallel':
            n_tile = self.macro_spec.block_n_dim
            k_tile = self.n_macros * self.macro_spec.block_k_dim
        elif mapping_strategy == 'Hybrid':
            bn = self.macro_spec.block_n_dim
            bk = self.macro_spec.block_k_dim
            best = None  # (score, macros_n, macros_k, n_tile, k_tile)
            for macros_n in range(1, self.n_macros):
                macros_k = self.n_macros - macros_n
                n_tile_cand = macros_n * bn
                k_tile_cand = macros_k * bk
                n_eff = min(n_tile_cand, N)
                k_eff = min(k_tile_cand, K)
                tiles_n = math.ceil(N / n_eff) if n_eff > 0 else float('inf')
                tiles_k = math.ceil(K / k_eff) if k_eff > 0 else float('inf')
                w_footprint = k_eff * n_eff
                score = tiles_n + tiles_k + 0.001 * w_footprint
                cand = (score, macros_n, macros_k, n_eff, k_eff)
                if best is None or cand < best:
                    best = cand
            _, macros_n, macros_k, n_tile, k_tile = best
        else:
            raise ValueError(f"不支持的映射策略: {mapping_strategy}")

        # buffer 约束决定 m_tile 可行上界；启用双缓冲的 buffer 等效容量减半
        effective_in_buf = self.size_input_buffer / (2 if self.double_buffer_input else 1)
        effective_out_buf = self.size_output_buffer / (2 if self.double_buffer_output else 1)

        m_tile_from_input = effective_in_buf // (k_tile * math.floor(self.storage_bytes)) if k_tile > 0 else float('inf')
        m_tile_from_output = effective_out_buf // (n_tile * math.floor(self.psum_bytes)) if n_tile > 0 else float('inf')

        base_m_tile = min(m_tile_from_input, m_tile_from_output)
        m_tile = self._select_m_tile(M, base_m_tile, m_tile_from_input, m_tile_from_output, tiling_policy, k_tile, n_tile)
        return m_tile, k_tile, n_tile

    def simulate(self, M: int, K: int, N: int, mapping_strategy: str, loop_order: str, tiling_policy: str = "max_fit") -> dict:
        m_tile, k_tile, n_tile = self._derive_tile_sizes(M, K, N, mapping_strategy, loop_order, tiling_policy)

        num_m_chunks = math.ceil(M / m_tile)
        num_k_chunks = math.ceil(K / k_tile)
        num_n_chunks = math.ceil(N / n_tile)

        latencies = {"load_X": 0, "load_W": 0, "pre_process": 0, "compute_macro": 0, "accum_vector_unit": 0, "local_readout": 0, "post_dequant": 0}
        energies_pj = {"load_X": 0, "load_W": 0, "pre_process": 0, "compute_macro": 0, "accum_vector_unit": 0, "read_X_from_buffer": 0, "read_psums": 0, "local_readout": 0, "post_dequant": 0}
        pipeline_tile_stages = []

        loop_map = {'M': range(num_m_chunks), 'K': range(num_k_chunks), 'N': range(num_n_chunks)}
        outer, mid, inner = loop_order
        main_loop_iterator = itertools.product(loop_map[outer], loop_map[mid], loop_map[inner])

        last_loaded = {'X': (-1, -1), 'W': (-1, -1)}
        tile_counter = 0  # 用于区分首块与后续块

        for loop_indices in main_loop_iterator:
            idx_map = {outer: loop_indices[0], mid: loop_indices[1], inner: loop_indices[2]}
            m, k, n = idx_map['M'], idx_map['K'], idx_map['N']
            tile_stage = {"load": 0, "pre": 0, "compute": 0, "post": 0}

            # --- 1. 权重加载 ---
            if last_loaded['W'] != (k, n):
                bytes_to_load_w = k_tile * n_tile * self.storage_bytes
                latencies["load_W"] += self.t_load_block_per_macro
                energies_pj["load_W"] += bytes_to_load_w * self.energy_params["sram_write_power"]
                last_loaded['W'] = (k, n)
                load_w_cycles = self.t_load_block_per_macro
                tile_stage["load"] += load_w_cycles

            # --- 2. 输入加载 ---
            if last_loaded['X'] != (m, k):
                bytes_to_load_x = m_tile * k_tile * self.storage_bytes
                load_x_cycles = math.ceil(bytes_to_load_x / self.bw_input_buffer_bytes_per_cycle)
                latencies["load_X"] += load_x_cycles
                energies_pj["load_X"] += bytes_to_load_x * self.energy_params["sram_write_power"]
                last_loaded['X'] = (m, k)
                tile_stage["load"] += load_x_cycles

            # --- 3. Pre-Processing / Alignment ---
            num_inputs_in_tile = m_tile * k_tile
            pre_align_cycles = 0
            if self.overheads['pre_align_latency_cycles_per_input'] > 0:
                pre_align_cycles += math.ceil(num_inputs_in_tile * self.overheads['pre_align_latency_cycles_per_input'] / self.macro_spec.block_k_dim)

            if self.data_format in ('BF16', 'MXINT8'):
                align_parallel = self.overheads['pre_align_shift_parallel_inputs'] or self.macro_spec.block_k_dim
                if self.data_format == 'BF16':
                    exp_ops = num_inputs_in_tile
                    exp_bits = self.overheads['bf16_exp_bits']
                else:
                    exp_ops = math.ceil(num_inputs_in_tile / self.overheads['mx_group_size'])
                    exp_bits = self.overheads['mx_scale_bits']

                exp_cycles = math.ceil(exp_ops * self.overheads['pre_align_exp_compare_cycles'] / align_parallel)
                exp_energy = exp_ops * exp_bits * self.overheads['pre_align_energy_pj_per_bit']

                shift_cycles = math.ceil((num_inputs_in_tile * self.overheads['pre_align_shift_avg_bits']) / align_parallel)
                shift_energy = num_inputs_in_tile * self.overheads['pre_align_shift_avg_bits'] * self.overheads['pre_align_shift_energy_pj_per_bit']

                pre_align_cycles += exp_cycles + shift_cycles
                energies_pj["pre_process"] += exp_energy + shift_energy
            else:
                total_bits_aligned = num_inputs_in_tile * (self.storage_bytes * 8)
                energies_pj["pre_process"] += total_bits_aligned * self.overheads['pre_align_energy_pj_per_bit']

            latencies["pre_process"] += pre_align_cycles
            tile_stage["pre"] += pre_align_cycles

            # --- 4. Macro 计算 ---
            num_loc_switches_per_block = self.macro_spec.n_loc_per_subarray
            local_readout_cycles = num_loc_switches_per_block * self.t_local_readout
            latencies["local_readout"] += local_readout_cycles
            energies_pj["local_readout"] += num_loc_switches_per_block * self.bytes_per_local_readout * self.energy_params["cim_local_read_power"]
            tile_stage["compute"] += local_readout_cycles

            total_compute_cycles_for_block = num_loc_switches_per_block * (m_tile * self.compute_bits)
            latencies["compute_macro"] += total_compute_cycles_for_block
            tile_stage["compute"] += total_compute_cycles_for_block

            ops_compute = m_tile * k_tile * n_tile
            energies_pj["compute_macro"] += ops_compute * self.energy_params["tpu_flop_power"]

            bytes_read_x = m_tile * k_tile * self.storage_bytes
            energies_pj["read_X_from_buffer"] += bytes_read_x * self.energy_params["sram_read_power"]

            # --- 5. Vector Unit 累加 ---
            if k > 0:
                ops_to_accum = m_tile * n_tile
                effective_ops_per_cycle = self.ops_per_cycle_vector_unit * self.overheads['post_process_throughput_scale']
                acc_cycles = math.ceil(ops_to_accum / effective_ops_per_cycle)
                latencies["accum_vector_unit"] += acc_cycles
                energies_pj["accum_vector_unit"] += ops_to_accum * self.energy_params["vect_flop_power"] * self.overheads["post_process_energy_scale"]

                bytes_psums = m_tile * n_tile * self.psum_bytes
                energies_pj["read_psums"] += bytes_psums * self.energy_params["sram_read_power"]
                energies_pj["load_X"] += bytes_psums * self.energy_params["sram_write_power"]
                tile_stage["post"] += acc_cycles

                if self.data_format in ('BF16', 'MXINT8'):
                    outputs_in_tile = m_tile * n_tile
                    post_dequant_cycles = math.ceil(outputs_in_tile * self.overheads['post_dequant_cycles_per_output'])
                    latencies["post_dequant"] += post_dequant_cycles
                    tile_stage["post"] += post_dequant_cycles

                    bits_to_handle = outputs_in_tile * self.overheads['psum_bitwidth']
                    energies_pj["post_dequant"] += bits_to_handle * self.overheads['post_dequant_energy_pj_per_bit']

            # 双缓冲：首块后 load 与后续阶段取 max（能耗不变）
            if (self.double_buffer_input or self.double_buffer_weight) and tile_counter > 0:
                load_cycles = tile_stage["load"]
                rest_cycles = tile_stage["pre"] + tile_stage["compute"] + tile_stage["post"]
                overlapped = max(load_cycles, rest_cycles)
                tile_stage["load"] = 0
                tile_stage["pre"] = overlapped
                tile_stage["compute"] = 0
                tile_stage["post"] = 0

            pipeline_tile_stages.append(tile_stage)
            tile_counter += 1

        total_latency = sum(latencies.values())
        total_energy_pj = sum(energies_pj.values())
        total_ops = M * K * N

        pipeline_latency_cycles = self._simulate_pipeline(pipeline_tile_stages)

        if self.clock_freq_ghz > 0 and pipeline_latency_cycles > 0:
            total_time_s = pipeline_latency_cycles / (self.clock_freq_ghz * 1e9)
            throughput_gops = total_ops / total_time_s / 1e9
        else:
            throughput_gops = 0.0

        results = {
            "Format": self.data_format,
            "latencies": latencies,
            "energies_pj": energies_pj,
            "total_latency_cycles_no_pipeline": total_latency,
            "pipeline_latency_cycles": pipeline_latency_cycles,
            "total_energy_pj": total_energy_pj,
            "throughput_gops": throughput_gops,
            "tile_sizes": (m_tile, k_tile, n_tile),
            "energy_efficiency_tops_per_watt": (total_ops / 1e12) / (total_energy_pj / 1e12) if total_energy_pj > 0 else 0
        }

        self._report_results(results)
        return results

    def _simulate_pipeline(self, tile_stage_list):
        """
        四级流水线调度：load -> pre -> compute -> post。
        每个stage一次只能处理一个tile，返回全部tile完成的总周期。
        """
        if not tile_stage_list:
            return 0

        stage_names = ["load", "pre", "compute", "post"]
        stage_free_time = {s: 0 for s in stage_names}
        pipeline_end = 0

        for tile_stage in tile_stage_list:
            s1_start = stage_free_time["load"]
            s1_end = s1_start + tile_stage["load"]

            s2_start = max(s1_end, stage_free_time["pre"])
            s2_end = s2_start + tile_stage["pre"]

            s3_start = max(s2_end, stage_free_time["compute"])
            s3_end = s3_start + tile_stage["compute"]

            s4_start = max(s3_end, stage_free_time["post"])
            s4_end = s4_start + tile_stage["post"]

            stage_free_time["load"] = s1_end
            stage_free_time["pre"] = s2_end
            stage_free_time["compute"] = s3_end
            stage_free_time["post"] = s4_end
            pipeline_end = s4_end

        return pipeline_end

    def _report_results(self, results: dict):
        print(f"=== 仿真结果 [{results['Format']}] ===")
        print(f"Latency Breakdown (Cycles):")
        for k, v in results['latencies'].items():
            ratio = v / results['total_latency_cycles_no_pipeline'] * 100
            print(f"  {k:<20}: {v:10,.0f} ({ratio:5.1f}%)")
        m_tile, k_tile, n_tile = results["tile_sizes"]
        print(f"  Tile Sizes (M,K,N): ({m_tile},{k_tile},{n_tile})")
        print(f"  Total (no pipeline): {results['total_latency_cycles_no_pipeline']:,.0f}")
        print(f"  Total (pipelined)  : {results['pipeline_latency_cycles']:,.0f}")
        print(f"Energy Breakdown (pJ):")
        for k, v in results['energies_pj'].items():
            ratio = v / results['total_energy_pj'] * 100
            print(f"  {k:<20}: {v:10,.2f} ({ratio:5.1f}%)")
        print(f"Performance:")
        print(f"  Throughput: {results['throughput_gops']:.2f} GOPS")
        print(f"  Efficiency: {results['energy_efficiency_tops_per_watt']:.2f} TOPS/W")
        print("========================================\n")


def sweep_buffer_and_tiling(M, K, N, buffer_size_pairs_kb, mapping_strategies, loop_orders, core_ctor_kwargs, tiling_policies=None):
    """
    buffer/tile/策略 批量探索：
    - buffer_size_pairs_kb: [(input_kb, output_kb), ...]
    - mapping_strategies: ['N_Parallel', 'K_Parallel', 'Hybrid', ...]
    - loop_orders: ['KMN', 'NKM', ...]
    - tiling_policies: ['max_fit', 'divisible', 'min_tail', ...]
    """
    summary = []
    if tiling_policies is None:
        tiling_policies = ["max_fit"]
    for in_kb, out_kb in buffer_size_pairs_kb:
        for mapping in mapping_strategies:
            for order in loop_orders:
                for tiling_policy in tiling_policies:
                    core = CIMCoreSimulator(
                        size_input_buffer_bytes=in_kb * 1024,
                        size_output_buffer_bytes=out_kb * 1024,
                        **core_ctor_kwargs
                    )
                    res = core.simulate(M, K, N, mapping_strategy=mapping, loop_order=order, tiling_policy=tiling_policy)
                    m_t, k_t, n_t = res["tile_sizes"]
                    summary.append({
                        "input_kb": in_kb,
                        "output_kb": out_kb,
                        "mapping": mapping,
                        "loop_order": order,
                        "tiling_policy": tiling_policy,
                        "tile": f"{m_t}x{k_t}x{n_t}",
                        "lat_pipelined": res["pipeline_latency_cycles"],
                        "throughput_gops": res["throughput_gops"],
                        "energy_pj": res["total_energy_pj"],
                        "ee_tops_per_w": res["energy_efficiency_tops_per_watt"]
                    })
    return summary


# --- 3. 使用示例 ---
if __name__ == "__main__":

    # 基础能量参数 (22nm)
    base_energy_params = {
        "tpu_flop_power": 0.0823,
        "vect_flop_power": 0.170,
        "sram_write_power": 1.568,
        "sram_read_power": 1.568,
        "cim_local_read_power": 0.436
    }

    macro_spec = CIMMacroSpec()

    # 任务规模
    M, K, N = 2048, 1024, 2048

    # --- Case 1: INT8 Baseline ---
    sim_int8 = CIMCoreSimulator(
        n_macros=4, macro_spec=macro_spec,
        size_input_buffer_bytes=128*1024, size_output_buffer_bytes=128*1024,
        bw_input_buffer_gbps=64, throughput_vector_unit_gops=128,
        energy_params=base_energy_params,
        data_format='INT8',
        overhead_params={
            "pre_align_latency_cycles_per_input": 0,
            "pre_align_energy_pj_per_bit": 0,
            "post_process_energy_scale": 1.0,
            "post_process_throughput_scale": 1.0
        }
    )
    sim_int8.simulate(M, K, N, 'N_Parallel', 'KMN')

    # --- Case 2: BF16 ---
    sim_bf16 = CIMCoreSimulator(
        n_macros=4, macro_spec=macro_spec,
        size_input_buffer_bytes=128*1024, size_output_buffer_bytes=128*1024,
        bw_input_buffer_gbps=64, throughput_vector_unit_gops=128,
        energy_params=base_energy_params,
        data_format='BF16',
        overhead_params={
            "pre_align_latency_cycles_per_input": 0.5,
            "pre_align_energy_pj_per_bit": 0.05,
            "post_process_energy_scale": 1.8,
            "post_process_throughput_scale": 0.8
        }
    )
    sim_bf16.simulate(M, K, N, 'N_Parallel', 'KMN')

    # --- Case 3: MXINT8 ---
    sim_mx = CIMCoreSimulator(
        n_macros=4, macro_spec=macro_spec,
        size_input_buffer_bytes=128*1024, size_output_buffer_bytes=128*1024,
        bw_input_buffer_gbps=64, throughput_vector_unit_gops=128,
        energy_params=base_energy_params,
        data_format='MXINT8',
        overhead_params={
            "pre_align_latency_cycles_per_input": 0.2,
            "pre_align_energy_pj_per_bit": 0.03,
            "post_process_energy_scale": 1.3,
            "post_process_throughput_scale": 0.9
        }
    )
    sim_mx.simulate(M, K, N, 'N_Parallel', 'KMN')

    # --- Buffer/Tiling 探索示例（可按需启用） ---
    # buffer_pairs = [(64, 64), (128, 128), (256, 256)]  # 单位 KB
    # mapping_strategies = ['N_Parallel', 'K_Parallel', 'Hybrid']
    # loop_orders = ['KMN', 'NKM']
    # summary = sweep_buffer_and_tiling(
    #     M=1024, K=1024, N=1024,
    #     buffer_size_pairs_kb=buffer_pairs,
    #     mapping_strategies=mapping_strategies,
    #     loop_orders=loop_orders,
    #     tiling_policies=["max_fit", "divisible", "divisible_max", "min_tail", "aligned_down", "balanced_io", "auto"],
    #     core_ctor_kwargs=dict(
    #         n_macros=4,
    #         macro_spec=macro_spec,
    #         bw_input_buffer_gbps=64,
    #         throughput_vector_unit_gops=128,
    #         energy_params=base_energy_params,
    #         data_format='INT8',
    #         overhead_params={
    #             "pre_align_latency_cycles_per_input": 0,
    #             "pre_align_energy_pj_per_bit": 0,
    #             "post_process_energy_scale": 1.0,
    #             "post_process_throughput_scale": 1.0
    #         }
    #     )
    # )
    # print("=== Buffer/Tiling Sweep Summary ===")
    # for item in summary:
    #     print(item)
