import math
import itertools

# 假设之前的 CIMMacroSimulator 类存在，我们这里只需要它的部分属性
# 为了让代码独立运行，我们定义一个简单的结构来存储Macro的规格
class CIMMacroSpec:
    def __init__(self, n_banks_internal=32, n_subarrays_per_bank=32, n_loc_per_subarray=4, n_cols_per_bank=8, p_input=8):
        self.n_banks_internal = n_banks_internal
        self.n_subarrays_per_bank = n_subarrays_per_bank
        self.n_loc_per_subarray = n_loc_per_subarray
        self.block_k_dim = n_subarrays_per_bank * n_loc_per_subarray
        self.block_n_dim = n_banks_internal
        self.n_cols_per_bank = n_cols_per_bank
        self.p_input = p_input

class CIMCoreSimulator:
    """
    一个用于模拟CIM Core进行矩阵乘法的仿真器，包含延迟和能量模型。
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
                 datatype_bytes: int = 1):
        """
        初始化CIM Core的硬件参数和能量模型。
        """
        print("--- 初始化CIM Core硬件配置 (含能量模型) ---")
        self.n_macros = n_macros
        self.macro_spec = macro_spec
        self.size_input_buffer = size_input_buffer_bytes
        self.size_output_buffer = size_output_buffer_bytes
        self.bw_input_buffer_gbps = bw_input_buffer_gbps
        self.throughput_vector_unit_gops = throughput_vector_unit_gops
        self.clock_freq_ghz = clock_freq_ghz
        self.datatype_bytes = datatype_bytes
        self.energy_params = energy_params

        # 转换为每个时钟周期为单位
        self.bw_input_buffer_bytes_per_cycle = (self.bw_input_buffer_gbps * 1e9) / (self.clock_freq_ghz * 1e9)
        self.ops_per_cycle_vector_unit = (self.throughput_vector_unit_gops * 1e9) / (self.clock_freq_ghz * 1e9)
        self.t_load_block_per_macro = self.macro_spec.block_k_dim
        self.t_local_readout = 1 # local readout的延迟，单位：周期
        
        # 处理一个完整的 K_tile 块(128)所需的计算延迟
        # = 4次loc切换 * (1周期readout + 8周期计算)
        self.t_block_compute_per_m = self.macro_spec.n_loc_per_subarray * \
                                     (self.t_local_readout + self.macro_spec.p_input)

        # local readout操作一次的数据量 (bytes)
        self.bytes_per_local_readout = (self.macro_spec.n_subarrays_per_bank * self.macro_spec.block_n_dim * self.macro_spec.n_cols_per_bank) / 8
        
        print(f"推导出的单个Macro处理一个K_tile块的计算延迟: {self.t_block_compute_per_m} 周期")
        print(f"单次Local Readout延迟: {self.t_local_readout} 周期")
        print(f"单次Local Readout数据量: {self.bytes_per_local_readout} Bytes")
        print("--------------------------\n")

        print(f"Macro数量: {self.n_macros}")
        print(f"单个Macro规格 (K_dim x N_dim): {self.macro_spec.block_k_dim} x {self.macro_spec.block_n_dim}")
        print("能量参数 (pJ/op or pJ/byte):")
        for key, value in self.energy_params.items():
            print(f"  {key}: {value}")
        print("--------------------------\n")
        
    def _derive_tile_sizes(self, M, K, N, mapping_strategy, loop_order):
        if mapping_strategy == 'N_Parallel':
            n_tile = self.n_macros * self.macro_spec.block_n_dim
            k_tile = self.macro_spec.block_k_dim
        elif mapping_strategy == 'K_Parallel':
            n_tile = self.macro_spec.block_n_dim
            k_tile = self.n_macros * self.macro_spec.block_k_dim
        else:
            raise ValueError(f"不支持的映射策略: {mapping_strategy}")

        m_tile_from_input = self.size_input_buffer // (k_tile * self.datatype_bytes) if k_tile > 0 else float('inf')
        m_tile_from_output = self.size_output_buffer // (n_tile * self.datatype_bytes) if n_tile > 0 else float('inf')
        
        m_tile = min(m_tile_from_input, m_tile_from_output)
        m_tile = max(1, int(m_tile))
        
        return m_tile, k_tile, n_tile

    def simulate(self, M: int, K: int, N: int, mapping_strategy: str, loop_order: str) -> dict:
        print(f"--- 开始仿真: Y({M},{N}) = X({M},{K}) * W({K},{N}) ---")
        print(f"策略: 映射={mapping_strategy}, 循环顺序={loop_order}")

        m_tile, k_tile, n_tile = self._derive_tile_sizes(M, K, N, mapping_strategy, loop_order)
        print(f"推导的Tile尺寸 (M_tile x K_tile x N_tile): {m_tile} x {k_tile} x {n_tile}")

        num_m_chunks = math.ceil(M / m_tile)
        num_k_chunks = math.ceil(K / k_tile)
        num_n_chunks = math.ceil(N / n_tile)

        # 初始化延迟和能量计数器
        latencies = {"load_X": 0, "load_W": 0, "compute_macro": 0, "accum_vector_unit": 0, "local_readout": 0}
        energies_pj = {"load_X": 0, "load_W": 0, "compute_macro": 0, "accum_vector_unit": 0, "read_X_from_buffer": 0, "read_psums": 0, "local_readout": 0}
        
        loop_map = {'M': range(num_m_chunks), 'K': range(num_k_chunks), 'N': range(num_n_chunks)}
        outer, mid, inner = loop_order
        main_loop_iterator = itertools.product(loop_map[outer], loop_map[mid], loop_map[inner])
        
        last_loaded = {'X': (-1, -1), 'W': (-1, -1)}

        for loop_indices in main_loop_iterator:
            idx_map = {outer: loop_indices[0], mid: loop_indices[1], inner: loop_indices[2]}
            m, k, n = idx_map['M'], idx_map['K'], idx_map['N']

            # A. 权重加载 (逻辑不变)
            if last_loaded['W'] != (k, n):
                bytes_to_load_w = k_tile * n_tile * self.datatype_bytes
                latencies["load_W"] += self.t_load_block_per_macro
                energies_pj["load_W"] += bytes_to_load_w * self.energy_params["sram_write_power"]
                last_loaded['W'] = (k, n)

            # B. 输入加载 (逻辑不变)
            if last_loaded['X'] != (m, k):
                bytes_to_load_x = m_tile * k_tile * self.datatype_bytes
                latencies["load_X"] += math.ceil(bytes_to_load_x / self.bw_input_buffer_bytes_per_cycle)
                energies_pj["load_X"] += bytes_to_load_x * self.energy_params["sram_write_power"]
                last_loaded['X'] = (m, k)

            # ##################################################
            # ### C. Macro计算 (已修正为正确的、高效的数据流) ###
            # ##################################################
            
            # 1. Local Readout 延迟和能量
            # 对于当前计算块，我们只需要进行 N_loc_per_subarray 次 local readout
            num_loc_switches_per_block = self.macro_spec.n_loc_per_subarray
            latencies["local_readout"] += num_loc_switches_per_block * self.t_local_readout
            energies_pj["local_readout"] += num_loc_switches_per_block * self.bytes_per_local_readout * self.energy_params["cim_local_read_power"]

            # 2. 核心计算延迟
            # 每次 local readout 后，都需要 M_tile * P_input 个周期来完成计算
            total_compute_cycles_for_block = num_loc_switches_per_block * (m_tile * self.macro_spec.p_input)
            latencies["compute_macro"] += total_compute_cycles_for_block

            # 3. 计算相关的能量
            # 能量C1: 计算操作本身的能量 (ops数不变)
            ops_compute = m_tile * k_tile * n_tile
            energies_pj["compute_macro"] += ops_compute * self.energy_params["tpu_flop_power"]
            
            # 能量C2: 从Input Buffer读取输入的能量
            bytes_read_x = m_tile * k_tile * self.datatype_bytes
            energies_pj["read_X_from_buffer"] += bytes_read_x * self.energy_params["sram_read_power"]
            
            # D. Vector Unit累加 (逻辑不变)
            if k > 0:
                ops_to_accum = m_tile * n_tile
                latencies["accum_vector_unit"] += math.ceil(ops_to_accum / self.ops_per_cycle_vector_unit)
                energies_pj["accum_vector_unit"] += ops_to_accum * self.energy_params["vect_flop_power"]
                bytes_psums = m_tile * n_tile * self.datatype_bytes
                energies_pj["read_psums"] += bytes_psums * self.energy_params["sram_read_power"]
                energies_pj["load_X"] += bytes_psums * self.energy_params["sram_write_power"]

        total_latency = sum(latencies.values())
        total_energy_pj = sum(energies_pj.values())
        total_ops = M * K * N # MAC计为1个OP
        
        if self.clock_freq_ghz > 0 and total_latency > 0:
            total_time_s = total_latency / (self.clock_freq_ghz * 1e9)
            throughput_gops = total_ops / total_time_s / 1e9
        else:
            total_time_s = float('inf')
            throughput_gops = 0.0
            
        results = {
            "M": M, "K": K, "N": N,
            "mapping_strategy": mapping_strategy,
            "loop_order": loop_order,
            "tile_sizes": f"{m_tile}x{k_tile}x{n_tile}",
            "latencies": latencies,
            "total_latency_cycles": total_latency,
            "total_ops": total_ops,
            "throughput_gops": throughput_gops,
            "latencies_cycles": latencies,
            "energies_pj": energies_pj,
            "total_latency_cycles": total_latency,
            "total_energy_pj": total_energy_pj,
            "energy_efficiency_tops_per_watt": (total_ops / 1e12) / (total_energy_pj / 1e12) if total_energy_pj > 0 else 0
        }

        self._report_results(results)
        return results

    def _report_results(self, results: dict):
        """以可读格式打印仿真结果。"""
        print("\n--- 仿真结果 ---")
        print(f"策略: {results['mapping_strategy']} / {results['loop_order']}")
        print(f"Tile尺寸 (MxKxN): {results['tile_sizes']}")
        print(f"延迟分析 (单位: 时钟周期):")
        for key, value in results['latencies'].items():
            print(f"  {key:<25}: {value:,.0f}")
        print(f"  {'Total Latency':<25}: {results['total_latency_cycles']:,.0f}")
        print("\n性能分析:")
        print(f"  总操作数 (GOP): {results['total_ops'] / 1e9:.4f}")
        print(f"  等效吞吐率 (GOP/s): {results['throughput_gops']:.2f}")
        print("\n能量分析 (单位: pJ):")
        for key, value in results['energies_pj'].items():
            print(f"  {key:<25}: {value:,.2f}")
        print(f"  {'Total Energy':<25}: {results['total_energy_pj']:,.2f}")
        print(f"\n能效 (Energy Efficiency): {results['energy_efficiency_tops_per_watt']:.2f} TOPs/W")
        print("---------------------\n")


if __name__ == "__main__":

    energy_params_22nm = {
        "tpu_flop_power": 0.0823, # pJ/op
        "vect_flop_power": 0.170, # pJ/op
        "sram_write_power": 1.568, # pJ/byte
        "sram_read_power": 1.568, # pJ/byte
        "cim_local_read_power": 0.436 # pJ/byte
    }
    # --- 定义硬件 ---
    # 定义单个Macro的规格
    macro_spec = CIMMacroSpec(n_banks_internal=64, n_subarrays_per_bank=64, n_loc_per_subarray=4, n_cols_per_bank=8, p_input=8)
    
    # 定义CIM Core的规格
    core = CIMCoreSimulator(
        n_macros=4,
        macro_spec=macro_spec,
        size_input_buffer_bytes=1024 * 32,  # 128 KB
        size_output_buffer_bytes=1024 * 32, # 128 KB
        bw_input_buffer_gbps=64,
        throughput_vector_unit_gops=128,
        energy_params=energy_params_22nm, # 注入能量模型
        clock_freq_ghz=1.0
    )

    # --- 定义任务 ---
    M, K, N = 4096, 1152, 4608

    # --- 设计空间探索 ---
    # 策略1: 'N_Parallel'映射，'NKM'循环顺序 (理论上权重复用好)
    core.simulate(M, K, N, mapping_strategy='N_Parallel', loop_order='NKM')

    # 策略2: 'N_Parallel'映射，'KMN'循环顺序 (理论上输入复用好)
    core.simulate(M, K, N, mapping_strategy='N_Parallel', loop_order='KMN')
    
    # 策略3: 切换为'K_Parallel'映射，观察累加单元的压力
    core.simulate(M, K, N, mapping_strategy='K_Parallel', loop_order='KMN')