import math
import itertools

# --- 1. 硬件规格定义 (保持不变，代表物理架构) ---
class CIMMacroSpec:
    def __init__(self, n_banks_internal=32, n_subarrays_per_bank=32, n_loc_per_subarray=4, n_cols_per_bank=8, p_input=8):
        self.n_banks_internal = n_banks_internal
        self.n_subarrays_per_bank = n_subarrays_per_bank
        self.n_loc_per_subarray = n_loc_per_subarray
        self.block_k_dim = n_subarrays_per_bank * n_loc_per_subarray
        self.block_n_dim = n_banks_internal
        self.n_cols_per_bank = n_cols_per_bank
        # p_input: 物理单元支持的输入位宽，通常为8
        self.p_input = p_input 

# --- 2. 核心仿真器 (大幅增强，支持多数据格式) ---
class CIMCoreSimulator:
    """
    支持 INT8 / MXINT8 / BF16 的 CIM 仿真器
    包含 Pre-alignment 和 Post-processing 的开销建模
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
                 # --- 新增参数 ---
                 data_format: str = 'INT8', # 'INT8', 'BF16', 'MXINT8'
                 overhead_params: dict = None):
        
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

        # --- A. 设置默认开销参数 (如果用户未提供) ---
        # 这些系数基于 ISSCC Paper 的分析
        default_overheads = {
            # 预处理/对齐带来的每bit能耗 (Digital Logic Power)
            "pre_align_energy_pj_per_bit": 0.0, 
            # 预处理带来的额外延迟 (CIM Array计算前的 Pipeline Bubble 或 Setup time)
            "pre_align_latency_cycles_per_input": 0, 
            
            # 后处理(Vector Unit)的能耗放大系数 (相对于INT累加)
            # BF16通常需要Shift+Add，能耗较高
            "post_process_energy_scale": 1.0,
            # 后处理的吞吐率折损系数 (例如浮点加法比INT加法慢)
            # 1.0 表示全速，0.5 表示速度减半
            "post_process_throughput_scale": 1.0
        }
        if overhead_params:
            default_overheads.update(overhead_params)
        self.overheads = default_overheads

        # --- B. 根据数据格式推导 存储位宽 vs 计算位宽 ---
        if data_format == 'INT8':
            self.storage_bytes = 1.0    # 8-bit 存储
            self.compute_bits = 8       # 8-bit 计算
        elif data_format == 'BF16':
            self.storage_bytes = 2.0    # 16-bit 存储 (带宽瓶颈!)
            self.compute_bits = 8       # 核心只算 8-bit Mantissa (参考ReCIM/M2-CIM)
        elif data_format == 'MXINT8':
            # Microscaling: 8-bit 尾数 + 8-bit 共享指数(每32个元素共享)
            # 平均 bytes = 1 + 1/32 ~= 1.03 (或者按块对齐算 1.125)
            self.storage_bytes = 1.03125 
            self.compute_bits = 8       # 核心只算 8-bit 整数
        else:
            raise ValueError(f"Unsupported format: {data_format}")

        # --- C. 基础参数计算 ---
        self.bw_input_buffer_bytes_per_cycle = (self.bw_input_buffer_gbps * 1e9) / (self.clock_freq_ghz * 1e9)
        self.ops_per_cycle_vector_unit = (self.throughput_vector_unit_gops * 1e9) / (self.clock_freq_ghz * 1e9)
        self.t_load_block_per_macro = self.macro_spec.block_k_dim
        self.t_local_readout = 1 
        
        # 核心计算延迟：取决于 compute_bits (例如BF16也是8个周期，而非16)
        self.t_block_compute_per_m = self.macro_spec.n_loc_per_subarray * \
                                     (self.t_local_readout + self.compute_bits)

        self.bytes_per_local_readout = (self.macro_spec.n_subarrays_per_bank * self.macro_spec.block_n_dim * self.macro_spec.n_cols_per_bank) / 8
        
        print(f"  Storage Precision: {self.storage_bytes:.4f} Bytes/item (影响带宽)")
        print(f"  Compute Precision: {self.compute_bits} Bits (影响阵列计算延迟)")
        print(f"  Pre-Align Overhead: {self.overheads['pre_align_latency_cycles_per_input']} Cycles/Input, {self.overheads['pre_align_energy_pj_per_bit']} pJ/bit")
        print(f"  Post-Process Scale: Energy x{self.overheads['post_process_energy_scale']}, Throughput x{self.overheads['post_process_throughput_scale']}")
        print("---------------------------------------------------\n")

    def _derive_tile_sizes(self, M, K, N, mapping_strategy, loop_order):
        # 逻辑不变，但使用 self.storage_bytes 来计算容量
        if mapping_strategy == 'N_Parallel':
            n_tile = self.n_macros * self.macro_spec.block_n_dim
            k_tile = self.macro_spec.block_k_dim
        elif mapping_strategy == 'K_Parallel':
            n_tile = self.macro_spec.block_n_dim
            k_tile = self.n_macros * self.macro_spec.block_k_dim
        else:
            raise ValueError(f"不支持的映射策略: {mapping_strategy}")

        # 这里的 datatype_bytes 变成了 self.storage_bytes (支持浮点/分数)
        m_tile_from_input = self.size_input_buffer // (k_tile * math.floor(self.storage_bytes)) if k_tile > 0 else float('inf')
        m_tile_from_output = self.size_output_buffer // (n_tile * math.floor(self.storage_bytes)) if n_tile > 0 else float('inf')
        
        m_tile = min(m_tile_from_input, m_tile_from_output)
        m_tile = max(1, int(m_tile))
        return m_tile, k_tile, n_tile

    def simulate(self, M: int, K: int, N: int, mapping_strategy: str, loop_order: str) -> dict:
        m_tile, k_tile, n_tile = self._derive_tile_sizes(M, K, N, mapping_strategy, loop_order)
        
        num_m_chunks = math.ceil(M / m_tile)
        num_k_chunks = math.ceil(K / k_tile)
        num_n_chunks = math.ceil(N / n_tile)

        # 初始化计数器 (新增 pre_process)
        latencies = {"load_X": 0, "load_W": 0, "pre_process": 0, "compute_macro": 0, "accum_vector_unit": 0, "local_readout": 0}
        energies_pj = {"load_X": 0, "load_W": 0, "pre_process": 0, "compute_macro": 0, "accum_vector_unit": 0, "read_X_from_buffer": 0, "read_psums": 0, "local_readout": 0}
        
        loop_map = {'M': range(num_m_chunks), 'K': range(num_k_chunks), 'N': range(num_n_chunks)}
        outer, mid, inner = loop_order
        main_loop_iterator = itertools.product(loop_map[outer], loop_map[mid], loop_map[inner])
        
        last_loaded = {'X': (-1, -1), 'W': (-1, -1)}

        for loop_indices in main_loop_iterator:
            idx_map = {outer: loop_indices[0], mid: loop_indices[1], inner: loop_indices[2]}
            m, k, n = idx_map['M'], idx_map['K'], idx_map['N']

            # --- 1. 权重加载 (受 storage_bytes 影响) ---
            if last_loaded['W'] != (k, n):
                # 数据量 = 元素个数 * 每个元素的存储字节数
                bytes_to_load_w = k_tile * n_tile * self.storage_bytes
                latencies["load_W"] += self.t_load_block_per_macro
                energies_pj["load_W"] += bytes_to_load_w * self.energy_params["sram_write_power"]
                last_loaded['W'] = (k, n)

            # --- 2. 输入加载 (受 storage_bytes 影响) ---
            if last_loaded['X'] != (m, k):
                bytes_to_load_x = m_tile * k_tile * self.storage_bytes
                latencies["load_X"] += math.ceil(bytes_to_load_x / self.bw_input_buffer_bytes_per_cycle)
                energies_pj["load_X"] += bytes_to_load_x * self.energy_params["sram_write_power"]
                last_loaded['X'] = (m, k)

            # --- 3. Pre-Processing / Alignment (新增模块) ---
            # 任何数据在进入CIM阵列前，如果是浮点/MX，需要经过对齐逻辑
            # 这是一个纯数字逻辑开销
            num_inputs_in_tile = m_tile * k_tile
            
            # 延迟：假设硬件有一定的并行度，或者流水线化。
            # 这里简化模型：每批数据有一个固定的 Setup Latency overhead
            if self.overheads['pre_align_latency_cycles_per_input'] > 0:
                 # 假设完全流水线化，只增加流水线填充时间，或者假设 throughput 受限
                 # 这里采用吞吐量模型：处理这么多数据需要多少周期
                 latencies["pre_process"] += math.ceil(num_inputs_in_tile * self.overheads['pre_align_latency_cycles_per_input'] / self.macro_spec.block_k_dim) # 假设K维度并行处理

            # 能耗：每个bit都需要经过数字逻辑 (Shifter/Comparator)
            total_bits_aligned = num_inputs_in_tile * (self.storage_bytes * 8)
            energies_pj["pre_process"] += total_bits_aligned * self.overheads['pre_align_energy_pj_per_bit']

            # --- 4. Macro 计算 (受 compute_bits 影响) ---
            # Local Readout
            num_loc_switches_per_block = self.macro_spec.n_loc_per_subarray
            latencies["local_readout"] += num_loc_switches_per_block * self.t_local_readout
            energies_pj["local_readout"] += num_loc_switches_per_block * self.bytes_per_local_readout * self.energy_params["cim_local_read_power"]

            # 核心计算: 使用 self.compute_bits (BF16和INT8都是8，所以这里延迟一样)
            total_compute_cycles_for_block = num_loc_switches_per_block * (m_tile * self.compute_bits)
            latencies["compute_macro"] += total_compute_cycles_for_block

            # 计算能量
            ops_compute = m_tile * k_tile * n_tile
            energies_pj["compute_macro"] += ops_compute * self.energy_params["tpu_flop_power"]
            
            # 读SRAM能量 (受 storage_bytes 影响)
            bytes_read_x = m_tile * k_tile * self.storage_bytes
            energies_pj["read_X_from_buffer"] += bytes_read_x * self.energy_params["sram_read_power"]
            
            # --- 5. Vector Unit 累加 (受 Post-Process Scale 影响) ---
            if k > 0:
                ops_to_accum = m_tile * n_tile
                
                # 应用吞吐率折损 (Throughput Scale)
                # 如果是浮点加法，ops_per_cycle 会变小 (即 latency 变大)
                effective_ops_per_cycle = self.ops_per_cycle_vector_unit * self.overheads['post_process_throughput_scale']
                latencies["accum_vector_unit"] += math.ceil(ops_to_accum / effective_ops_per_cycle)
                
                # 应用能耗放大 (Energy Scale)
                energies_pj["accum_vector_unit"] += ops_to_accum * self.energy_params["vect_flop_power"] * self.overheads['post_process_energy_scale']
                
                # 读写 buffer (PSUM 通常保持高精度，比如FP32或INT32，这里简化假设PSUM位宽也是 storage_bytes，
                # 或者你可以单独定义 psum_bytes。为了对比公平，暂假设PSUM位宽与输入输出一致或按比例)
                bytes_psums = m_tile * n_tile * self.storage_bytes # 简化
                energies_pj["read_psums"] += bytes_psums * self.energy_params["sram_read_power"]
                energies_pj["load_X"] += bytes_psums * self.energy_params["sram_write_power"] # Write back

        total_latency = sum(latencies.values())
        total_energy_pj = sum(energies_pj.values())
        total_ops = M * K * N 
        
        if self.clock_freq_ghz > 0 and total_latency > 0:
            total_time_s = total_latency / (self.clock_freq_ghz * 1e9)
            throughput_gops = total_ops / total_time_s / 1e9
        else:
            throughput_gops = 0.0
            
        results = {
            "Format": self.data_format,
            "latencies": latencies,
            "energies_pj": energies_pj,
            "total_latency_cycles": total_latency,
            "total_energy_pj": total_energy_pj,
            "throughput_gops": throughput_gops,
            "energy_efficiency_tops_per_watt": (total_ops / 1e12) / (total_energy_pj / 1e12) if total_energy_pj > 0 else 0
        }
        
        self._report_results(results)
        return results

    def _report_results(self, results: dict):
        print(f"=== 仿真结果 [{results['Format']}] ===")
        print(f"Latency Breakdown (Cycles):")
        for k, v in results['latencies'].items():
            ratio = v / results['total_latency_cycles'] * 100
            print(f"  {k:<20}: {v:10,.0f} ({ratio:5.1f}%)")
        print(f"Energy Breakdown (pJ):")
        for k, v in results['energies_pj'].items():
            ratio = v / results['total_energy_pj'] * 100
            print(f"  {k:<20}: {v:10,.2f} ({ratio:5.1f}%)")
        print(f"Performance:")
        print(f"  Throughput: {results['throughput_gops']:.2f} GOPS")
        print(f"  Efficiency: {results['energy_efficiency_tops_per_watt']:.2f} TOPS/W")
        print("========================================\n")


# --- 3. 使用示例：对比 INT8, BF16, MXINT8 ---
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
    # 纯整数，无对齐开销，后处理简单
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

    # --- Case 2: BF16 (Digital CIM Approach) ---
    # 特征：
    # 1. Storage = 2.0 Bytes (带宽压力大)
    # 2. Pre-Align: 需要比较指数，移位尾数。有一定的数字逻辑功耗和延迟。
    # 3. Post-Process: 浮点累加，功耗是INT加法的 1.5~2倍，吞吐可能受限(0.8x)。
    sim_bf16 = CIMCoreSimulator(
        n_macros=4, macro_spec=macro_spec,
        size_input_buffer_bytes=128*1024, size_output_buffer_bytes=128*1024,
        bw_input_buffer_gbps=64, throughput_vector_unit_gops=128,
        energy_params=base_energy_params,
        data_format='BF16',
        overhead_params={
            "pre_align_latency_cycles_per_input": 0.5, # 假设流水线化，平均每输入增加0.5周期
            "pre_align_energy_pj_per_bit": 0.05,       # 移位器功耗估算
            "post_process_energy_scale": 1.8,          # 浮点累加更费电
            "post_process_throughput_scale": 0.8       # 浮点累加稍微慢一点
        }
    )
    sim_bf16.simulate(M, K, N, 'N_Parallel', 'KMN')

    # --- Case 3: MXINT8 (Microscaling, ISSCC 2025 style) ---
    # 特征：
    # 1. Storage ~= 1.03 Bytes (带宽优势巨大，接近INT8)
    # 2. Pre-Align: 主要是处理共享指数(Shared Scale)，比BF16逐个对齐要简单一点，或者类似。
    # 3. Post-Process: 需要 De-scaling，比纯INT复杂，但比全浮点简单。
    sim_mx = CIMCoreSimulator(
        n_macros=4, macro_spec=macro_spec,
        size_input_buffer_bytes=128*1024, size_output_buffer_bytes=128*1024,
        bw_input_buffer_gbps=64, throughput_vector_unit_gops=128,
        energy_params=base_energy_params,
        data_format='MXINT8',
        overhead_params={
            "pre_align_latency_cycles_per_input": 0.2, # 块级对齐，均摊开销小
            "pre_align_energy_pj_per_bit": 0.03,       # 逻辑更简单
            "post_process_energy_scale": 1.3,          # 需要 scaling 操作
            "post_process_throughput_scale": 0.9
        }
    )
    sim_mx.simulate(M, K, N, 'N_Parallel', 'KMN')