import math
import itertools
import pandas as pd

# --- 1. 硬件规格定义 ---
class CIMMacroSpec:
    def __init__(self, n_banks_internal=64, n_subarrays_per_bank=64, n_loc_per_subarray=4, n_cols_per_bank=8, p_input=8):
        self.n_banks_internal = n_banks_internal
        self.n_subarrays_per_bank = n_subarrays_per_bank
        self.n_loc_per_subarray = n_loc_per_subarray
        self.block_k_dim = n_subarrays_per_bank * n_loc_per_subarray
        self.block_n_dim = n_banks_internal
        self.n_cols_per_bank = n_cols_per_bank
        # p_input: 物理单元支持的输入位宽，通常为8
        self.p_input = p_input 

# --- 2. 核心仿真器 ---
class CIMCoreSimulator:
    """
    支持 INT8 / MXINT8 / BF16 的 CIM 仿真器
    包含 Pre-alignment 和 Post-processing 的开销建模
    """

    def __init__(self,
                 n_macros: int,
                 macro_spec: CIMMacroSpec,
                 size_l1_input_buffer_bytes: int, 
                 size_l1_output_buffer_bytes: int,
                 size_l0_input_buffer_bytes: int,
                 size_l0_output_buffer_bytes: int,
                 bw_noc_to_l1_gbps: float,
                 bw_l1_to_l0_gbps: float,
                 throughput_vector_unit_gops: float,
                 energy_params: dict,
                 clock_freq_ghz: float = 1.0,
                 data_format: str = 'INT8',
                 custom_overheads: dict = None):
        
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

        # --- 新增：为所有数据格式定义详细属性 ---
        # compute_bits: 物理硬件位宽 (INT8)
        # effective_compute_bits: 有效计算位宽 (影响延迟)
        # mantissa_bits: 尾数位数 (用于计算活动因子)
        format_specs = {
            # Format: {'storage_bytes', 'mantissa_bits', 'guard_bits'}
            'INT8':     {'storage': 1.0,       'mantissa': 8, 'guard': 0},
            'BF16':     {'storage': 2.0,       'mantissa': 8, 'guard': 7}, # 8位尾数 + 5保护位 = 13位有效计算
            'MXINT8':   {'storage': 1.03125,   'mantissa': 8, 'guard': 2}, # 共享指数限制了动态范围
            'FP8':      {'storage': 1.0,       'mantissa': 4, 'guard': 5}, # 4位尾数 + 4保护位 = 8位有效计算
            'MXFP8_M4': {'storage': 1.03125, 'mantissa': 4, 'guard': 5}  # 共享指数限制了动态范围
        }
        
        spec = format_specs.get(data_format)
        if not spec:
            raise ValueError(f"Unsupported format: {data_format}")

        self.storage_bytes = spec['storage']
        self.compute_bits = 8 # 硬件物理位宽固定为 8
        self.effective_compute_bits = spec['mantissa'] + spec['guard']

        base_activity_factor = 0.3
        if self.effective_compute_bits <= self.compute_bits:
            data_activity_ratio = self.effective_compute_bits / self.compute_bits
            self.activity_factor = base_activity_factor + (1 - base_activity_factor) * data_activity_ratio
        else:
            # e.g., BF16 需要13位，远超8位，每次8位操作都是满负荷
            self.activity_factor = 1.0
        
        # --- 统一的、分阶段的开销模型 ---
        self.overheads = {
            "pre_latency_cycles_per_input": 0.0,
            "pre_energy_pj_per_bit": 0.0,
            "macro_energy_scale": 1.0,
            "post_throughput_scale": 1.0,
            "post_energy_scale": 1.0,
        }

        # --- 根据数据格式，加载精细化的开销 ---
        if data_format == 'BF16':
            self.overheads.update({
                "pre_latency_cycles_per_input": 0.3, "pre_energy_pj_per_bit": 0.05,
                "post_throughput_scale": 0.9, "post_energy_scale": 1.4,
            })
        elif data_format == 'MXINT8':
            self.overheads.update({
                "pre_latency_cycles_per_input": 0.2, "pre_energy_pj_per_bit": 0.03,
                "macro_energy_scale": 1.1, "post_throughput_scale": 0.95, "post_energy_scale": 1.2,
            })
        elif data_format == 'FP8':
            # 开销类似BF16，但因位宽短而稍低
            self.overheads.update({
                "pre_latency_cycles_per_input": 0.25, "pre_energy_pj_per_bit": 0.04,
                "post_throughput_scale": 0.9, "post_energy_scale": 1.3,
            })
        elif data_format == 'MXFP8_M4':
            # 开销类似MXINT8，但因位宽短而稍低
            self.overheads.update({
                "pre_latency_cycles_per_input": 0.15, "pre_energy_pj_per_bit": 0.025,
                "macro_energy_scale": 1.05, "post_throughput_scale": 0.95, "post_energy_scale": 1.15,
            })
        
        if custom_overheads:
            self.overheads.update(custom_overheads)

        # 基础参数计算
        self.ops_per_cycle_vector_unit = (self.throughput_vector_unit_gops * 1e9) / (self.clock_freq_ghz * 1e9)
        self.t_load_block_per_macro = self.macro_spec.block_k_dim
        self.t_local_readout = 1 
        self.bytes_per_local_readout = (self.macro_spec.n_subarrays_per_bank * self.macro_spec.block_n_dim * self.macro_spec.n_cols_per_bank) / 8

        # 打印仿真器配置
        print(f"--- Memory Hierarchy Setup ---")
        print(f"  L1 Buffer (Global): {self.size_l1_input/1024:.1f} KB")
        print(f"--- Data Format Specifics ({data_format}) ---")
        print(f"  Storage Precision: {self.storage_bytes:.4f} Bytes/item")
        print(f"  Mantissa/Guard Bits: {spec['mantissa']} / {spec['guard']}")
        print(f"  Effective Compute Bits (Latency): {self.effective_compute_bits} Bits")
        print(f"  CIM Energy Activity Factor: {self.activity_factor:.2f}")
        print("---------------------------------------------------\n")

    def _derive_tile_sizes(self, M, K, N, partition_strategy, loop_order):
        """
        计算两级 Tile 大小：
        1. L0 Tile (Sub Tile): 受限于单个 Macro 的 L0 容量。
        2. L1 Tile (Super Tile): 受限于 Core 的 L1 容量。
        """
        # --- A. 解析并行策略 ---
        if isinstance(partition_strategy, str):
            if partition_strategy == 'N_Parallel':
                pk, pn = 1, self.n_macros
            elif partition_strategy == 'K_Parallel':
                pk, pn = self.n_macros, 1
            else:
                raise ValueError(f"Unknown strategy: {partition_strategy}")
        else:
            pk, pn = partition_strategy

        # --- B. 基础物理维度 (Hardware Atomic Dimensions) ---
        atomic_k = pk * self.macro_spec.block_k_dim
        atomic_n = pn * self.macro_spec.block_n_dim
        bytes_per_item = self.storage_bytes

        # --- C. 计算 L0 Tile (微观切分) ---
        k_per_macro = self.macro_spec.block_k_dim
        # 注意：这里可能产生浮点数，所以必须转 int
        m_limit_l0_input = int(self.size_l0_input // (k_per_macro * bytes_per_item)) if k_per_macro > 0 else M
        
        n_per_macro = self.macro_spec.block_n_dim
        m_limit_l0_output = int(self.size_l0_output // (n_per_macro * bytes_per_item)) if n_per_macro > 0 else M
        
        m_tile_l0 = min(m_limit_l0_input, m_limit_l0_output)
        m_tile_l0 = max(1, int(m_tile_l0)) 

        # L0 Tile 的逻辑尺寸 (系统视角)
        l0_dims = {
            'm': int(m_tile_l0),   # 强制转 int
            'k': int(atomic_k),    # 强制转 int
            'n': int(atomic_n)     # 强制转 int
        }

        # --- D. 计算 L1 Tile (宏观切分) ---
        k_tile_l1 = atomic_k
        n_tile_l1 = atomic_n
        
        # M_L1 * K_L1 * Bytes <= L1_Size
        # 这里的 bytes_per_item 是 float，会导致结果为 float (例如 64.0)
        m_limit_l1 = self.size_l1_input // (k_tile_l1 * bytes_per_item)
        
        # 对齐: L1 Tile 最好是 L0 Tile 的整数倍
        # !!! 修复点：最后加上 int() 强转 !!!
        m_tile_l1 = int((m_limit_l1 // m_tile_l0) * m_tile_l0)
        
        # 边界: 至少要能装下一个 L0 Tile
        if m_tile_l1 < m_tile_l0:
            m_tile_l1 = int(m_tile_l0)
            
        l1_dims = {
            'm': int(m_tile_l1),  # 确保是 int，修复 range() 报错
            'k': int(k_tile_l1),
            'n': int(n_tile_l1)
        }
        
        return l0_dims, l1_dims, (pk, pn)

    def simulate(self, M: int, K: int, N: int, partition_strategy, loop_order: str) -> dict:
        l0_dims, l1_dims, (pk, pn) = self._derive_tile_sizes(M, K, N, partition_strategy, loop_order)
        
        t_avail = {"NoC": 0, "Bus_L1_L0": 0, "Pre_Proc": 0, "Macro": 0, "Post_Proc": 0}
        t_l1_space_available = 0 
        
        energies_pj = {k: 0 for k in [
            "dram_read", "dram_write", "load_X_NoC", "load_W", 
            "read_L1", "write_L0", "pre_process", 
            "read_L0", "local_readout", "compute_macro", 
            "accum_vector_unit", "read_psums", "write_psums"
        ]}

        bw_noc_bpc = (self.bw_noc_to_l1_gbps * 1e9) / (self.clock_freq_ghz * 1e9) / 8
        bw_bus_bpc = (self.bw_l1_to_l0_gbps * 1e9) / (self.clock_freq_ghz * 1e9) / 8
        
        current_l1_input_tag = None
        current_l1_weight_tag = None

        step_m_l1, step_k_l1, step_n_l1 = l1_dims['m'], l1_dims['k'], l1_dims['n']
        step_m_l0, step_k_l0, step_n_l0 = l0_dims['m'], l0_dims['k'], l0_dims['n']

        ranges_l1 = {'M': range(0, M, step_m_l1), 'K': range(0, K, step_k_l1), 'N': range(0, N, step_n_l1)}
        outer_iter = itertools.product(ranges_l1[loop_order[0]], ranges_l1[loop_order[1]], ranges_l1[loop_order[2]])

        for l1_indices in outer_iter:
            idx_map_l1 = {loop_order[0]: l1_indices[0], loop_order[1]: l1_indices[1], loop_order[2]: l1_indices[2]}
            m_l1_start, k_l1_start, n_l1_start = idx_map_l1['M'], idx_map_l1['K'], idx_map_l1['N']
            
            m_l1_actual = min(step_m_l1, M - m_l1_start)
            k_l1_actual = min(step_k_l1, K - k_l1_start)
            n_l1_actual = min(step_n_l1, N - n_l1_start)

            # LEVEL 1: DRAM -> L1 Traffic (With Backpressure)
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

            # LEVEL 2: L1 -> L0 Traffic & Compute (Sub Tiles)
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
                cycles_pre = math.ceil(num_inputs_to_pre_proc * self.overheads['pre_latency_cycles_per_input'] / parallelism)
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
                
                base_readout_energy = num_switches * self.bytes_per_local_readout * self.energy_params["cim_local_read_power"] * self.n_macros
                energies_pj["local_readout"] += base_readout_energy * self.activity_factor * num_chunks

                base_compute_energy = (m_tile * k_tile * n_tile) * self.energy_params["tpu_flop_power"]
                ops_equivalent = base_compute_energy / self.energy_params["tpu_flop_power"] * (self.effective_compute_bits / 8.0)
                energy_compute = ops_equivalent * self.energy_params["tpu_flop_power"]
                energies_pj["compute_macro"] += energy_compute * self.overheads['macro_energy_scale'] * self.activity_factor
                
                # =========================================================
                # === [Action 4: Post-processing with Corrected PSUM Logic] ===
                # =========================================================
                ops_mac_output = m_tile * n_tile
                reduction_penalty = 1.0 if pk == 1 else 1.5
                
                # 检查这是否是K维度的第一次计算
                is_first_k_chunk = (k_l1_start + dk == 0)

                # 1. PSUM Read (如果不是第一个K chunk)
                bytes_psum = ops_mac_output * 4 # PSUMs usually 32-bit
                if not is_first_k_chunk:
                    energies_pj["read_psums"] += bytes_psum * self.energy_params["l0_sram_read_power"]
                
                # 2. Accumulation in Vector Unit (延迟和能耗)
                ops_to_accumulate = ops_mac_output
                
                eff_ops_per_cycle = self.ops_per_cycle_vector_unit * self.overheads['post_throughput_scale'] / reduction_penalty
                dur_post = math.ceil(ops_to_accumulate / eff_ops_per_cycle) if eff_ops_per_cycle > 0 else 0
                start_post = max(t_avail["Macro"], t_avail["Post_Proc"])
                t_avail["Post_Proc"] = start_post + dur_post
                
                base_accum_energy = ops_to_accumulate * self.energy_params["vect_flop_power"]
                energies_pj["accum_vector_unit"] += base_accum_energy * self.overheads['post_energy_scale'] * reduction_penalty
                
                # 3. PSUM Write (每次K迭代后都写回)
                energies_pj["write_psums"] += bytes_psum * self.energy_params["l0_sram_write_power"]

            # === [关键更新] Update Space Availability ===
            t_l1_space_available = t_avail["Macro"]

        # === 循环结束: 最终结果写回 (DRAM Write) ===
        output_bytes = 4 # 最终输出总是高精度
        total_out_bytes = M * N * output_bytes
        energies_pj["dram_write"] += total_out_bytes * self.energy_params["dram_write_power"]

        # 汇总结果
        total_latency = max(t_avail.values())
        total_energy_pj = sum(energies_pj.values())
        # 注意：这里的 M*K*N 是理论Ops，实际有效Ops可能因数据格式而异，但用这个作为基准是通用做法
        total_ops = M * K * N * 2 # MAC operation is 2 ops (multiply and add)
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
            "energy_efficiency_tops_per_watt": (total_ops/1e12)/(total_energy_pj/1e12) if total_energy_pj > 0 else 0,
            "bottleneck": max(t_avail, key=t_avail.get)
        }

    # 稍微修改报告函数以适应新的 latency 结构
    def _report_results(self, results: dict):
        print(f"=== 仿真结果 [{results.get('Format', 'N/A')}] ===")
        
        # 1. 流水线/延迟详情
        print(f"Pipeline Status (Last Timestamp per Stage):")
        # 兼容处理：检查 latencies 是时间戳字典还是时长字典
        if 'latencies' in results:
            # 找最大值用于算百分比（如果是duration的话）或者直接打印
            for k, v in results['latencies'].items():
                print(f"  {k:<15}: {v:12,.0f} Cycles")
        
        if 'bottleneck' in results:
            print(f"  >> System Bottleneck determined by: {results['bottleneck']}")
        
        # 2. 能耗详情
        print(f"Energy Breakdown (pJ):")
        total_e = results['total_energy_pj']
        for k, v in results['energies_pj'].items():
            # 防止除以0
            ratio = (v / total_e * 100) if total_e > 0 else 0
            print(f"  {k:<20}: {v:12,.2f} ({ratio:5.1f}%)")
            
        # 3. 总体性能
        print(f"Performance:")
        print(f"  Total Latency: {results['total_latency_cycles']:,.0f} Cycles")
        print(f"  Total Energy : {results['total_energy_pj']:,.2f} pJ")
        print(f"  Throughput   : {results['throughput_gops']:.2f} GOPS")
        print(f"  Efficiency   : {results['energy_efficiency_tops_per_watt']:.2f} TOPS/W")
        print("========================================\n")

class CIMDesignExplorer:
    def __init__(self, macro_spec, l0_params, energy_params, workload):
        self.macro_spec = macro_spec
        self.l0_params = l0_params
        self.energy_params = energy_params
        self.workload = workload # (M, K, N)
        
        # 固定带宽参数 (可以根据需要改为变量)
        self.fixed_bw = {
            "noc_to_l1": 64.0, 
            "l1_to_l0": 128.0
        }

    def run_dse_hybrid(self, l1_size_candidates, loop_orders, data_format: str):
        # 1. 解包工作负载
        M, K, N = self.workload 
        results_db = []

        # 2. 生成策略 (假设 Total Macros = 4)
        total_macros = 32
        valid_partitions = []
        for i in range(1, total_macros + 1):
            if total_macros % i == 0:
                pk = i
                pn = total_macros // i
                valid_partitions.append((pk, pn))
        
        # 生成搜索空间
        search_space = list(itertools.product(l1_size_candidates, valid_partitions, loop_orders))
        
        print(f"Starting Hybrid DSE for Workload M={M}, K={K}, N={N}...")
        print(f"Total configurations to explore: {len(search_space)}")
        
        count = 0
        failed_count = 0
        
        for l1_size, strategy, order in search_space:
            # 实例化仿真器
            sim = CIMCoreSimulator(
                n_macros=total_macros, 
                macro_spec=self.macro_spec,
                clock_freq_ghz=1.0,
                
                size_l1_input_buffer_bytes=l1_size,
                size_l1_output_buffer_bytes=l1_size,
                
                size_l0_input_buffer_bytes=self.l0_params['input'],
                size_l0_output_buffer_bytes=self.l0_params['output'],
                bw_noc_to_l1_gbps=self.fixed_bw['noc_to_l1'],
                bw_l1_to_l0_gbps=self.fixed_bw['l1_to_l0'],
                
                throughput_vector_unit_gops=128,
                energy_params=self.energy_params,
                data_format=data_format, 
                custom_overheads=None
            )
            
            try:
                # 运行仿真
                res = sim.simulate(M, K, N, strategy, order)
                
                edp = res['total_energy_pj'] * res['total_latency_cycles']
                
                entry = {
                    "Data_Format": data_format,
                    "L1_Size_KB": l1_size / 1024,
                    "Strategy": str(strategy),
                    "Loop_Order": order,
                    "Latency_Cyc": res['total_latency_cycles'],
                    "Energy_pJ": res['total_energy_pj'],
                    "Throughput_GOPS": res['throughput_gops'],
                    "EDP": edp,
                    "Bottleneck": res['bottleneck']
                }
                results_db.append(entry)
                count += 1
                
            except Exception as e:
                failed_count += 1
                # --- 修改点：打印具体的报错信息 ---
                print(f"[ERROR] Config failed: L1={l1_size}, Strategy={strategy}, Order={order}")
                print(f"Error details: {e}")
                import traceback
                traceback.print_exc() # 打印详细的错误堆栈
                print("-" * 30)
                # 如果你想看每一个错误，就把下面这行 break 注释掉；
                # 如果你想一出错就停下来修bug，就保留 break
                break 

        print(f"DSE Completed. Valid: {count}, Failed: {failed_count}")
        return results_db

    def analyze_results(self, results_db):
        # 1. 检查是否为空
        if not results_db:
            print("\n[WARNING] No valid results found! Check the error logs above.")
            return pd.DataFrame() # 返回空表

        try:
            df = pd.DataFrame(results_db)
            print(f"\n=== DSE Summary (Total Configs: {len(df)}) ===")
            
            # 1. 最佳 EDP (平衡)
            # 确保列名存在
            if "EDP" in df.columns:
                best_edp = df.sort_values(by="EDP").iloc[0]
                print(f"\n[Best EDP Config] (Balanced)")
                print(best_edp.to_string())
            
            # 2. 最佳 吞吐
            if "Throughput_GOPS" in df.columns:
                best_perf = df.sort_values(by="Throughput_GOPS", ascending=False).iloc[0]
                print(f"\n[Max Throughput Config] (Performance)")
                print(best_perf[['L1_Size_KB', 'Strategy', 'Loop_Order', 'Throughput_GOPS', 'Bottleneck']].to_string())
            
            # 3. 最佳 能效
            if "Energy_pJ" in df.columns:
                best_energy = df.sort_values(by="Energy_pJ").iloc[0]
                print(f"\n[Min Energy Config] (Efficiency)")
                print(best_energy[['L1_Size_KB', 'Strategy', 'Loop_Order', 'Energy_pJ', 'Bottleneck']].to_string())

            return df
            
        except ImportError:
            print("Pandas not found, utilizing basic sort.")
            sorted_res = sorted(results_db, key=lambda x: x.get('EDP', float('inf')))
            if sorted_res:
                print("Best EDP:", sorted_res[0])
            return sorted_res

    def reproduce_specific_config(self, config_row):
        import ast

        print("\n" + "="*80)
        print(f"REPRODUCING DETAILED REPORT FOR BEST CONFIGURATION")
        print("="*80)

        # 1. 参数还原
        data_format = config_row['Data_Format']
        l1_size_bytes = int(config_row['L1_Size_KB'] * 1024)
        strat_raw = config_row['Strategy']
        try:
            strategy = ast.literal_eval(strat_raw)
        except (ValueError, SyntaxError):
            strategy = strat_raw
        loop_order = config_row['Loop_Order']
        
        print(f"Target Config: L1={config_row['L1_Size_KB']} KB | Strategy={strategy} | Order={loop_order}")
        
        # 2. 重新实例化仿真器
        # 务必确保这里的 n_macros 和你 DSE 时用的一致
        total_macros = 32
        
        sim = CIMCoreSimulator(
            n_macros=total_macros, 
            macro_spec=self.macro_spec,
            clock_freq_ghz=1.0,
            
            size_l1_input_buffer_bytes=l1_size_bytes,
            size_l1_output_buffer_bytes=l1_size_bytes,
            
            size_l0_input_buffer_bytes=self.l0_params['input'],
            size_l0_output_buffer_bytes=self.l0_params['output'],
            bw_noc_to_l1_gbps=self.fixed_bw['noc_to_l1'],
            bw_l1_to_l0_gbps=self.fixed_bw['l1_to_l0'],
            
            throughput_vector_unit_gops=128,
            energy_params=self.energy_params,
            data_format=data_format, 
            custom_overheads=None
        )
        
        # 3. 运行仿真
        M, K, N = self.workload
        res = sim.simulate(M, K, N, strategy, loop_order)
        
        # 4. [关键] 显式调用报告打印函数
        # 即使 simulate 内部不再打印，这里也会强制打印
        sim._report_results(res)


# --- 3. 使用示例 ---
if __name__ == "__main__":
    
    # 基础能量参数 (22nm)
    extended_energy_params = {
        "tpu_flop_power": 0.0823, 
        "vect_flop_power": 0.170, 
        "sram_write_power": 1.568, 
        "sram_read_power": 1.568, 
        "l1_sram_write_power": 2.5,     # High energy to write from NoC
        "l1_sram_read_power": 2.0,      # High energy to drive bus to L0
        
        "l0_sram_write_power": 0.5,     # Low energy (local wire)
        "l0_sram_read_power": 0.5,       # Low energy (feed CIM)
        "cim_local_read_power": 0.436,
        "dram_read_power": 32,
        "dram_write_power": 32, 
    }

    macro_spec = CIMMacroSpec(
        n_banks_internal=64, 
        n_subarrays_per_bank=64, 
        n_loc_per_subarray=4, 
        n_cols_per_bank=8
    )

    # 1. 设置固定参数
    # 使用之前定义的 MXINT8 参数
    mx_overheads = {
        "pre_align_latency_cycles_per_input": 0.2,
        "pre_align_energy_pj_per_bit": 0.03,
        "post_energy_scale": 1.3,
        "post_throughput_scale": 0.9
    }

    # 定义搜索空间
    # L1 大小: 64KB 到 1MB
    l1_candidates = [64*1024, 128*1024, 256*1024, 512*1024, 1024*1024]

    # 映射策略
    # strategies = ['N_Parallel', 'K_Parallel']

    # 循环顺序: 6种排列
    loop_orders = [''.join(p) for p in itertools.permutations('MKN')]
    # 例如: ['MKN', 'MNK', 'KMN', 'KNM', 'NMK', 'NKM']

    # 1. 初始化 DSE 引擎 (不再需要 overheads)
    dse = CIMDesignExplorer(
        macro_spec=macro_spec,
        l0_params={'input': 4096, 'output': 4096},
        energy_params=extended_energy_params,
        workload=(1024, 4096, 1024)
    )

    # 2. 定义要测试的数据格式列表
    formats_to_test = ['INT8', 'BF16', 'MXINT8', 'FP8', 'MXFP8_M4']
    # formats_to_test = ['INT8']
    all_results = []

    # 3. 循环运行 DSE
    for fmt in formats_to_test:
        results = dse.run_dse_hybrid(l1_candidates, loop_orders, data_format=fmt)
        all_results.extend(results)

    # 4. 对所有格式的全部结果进行分析
    if all_results:
        df_all = dse.analyze_results(all_results)
        
        # 找到全局最佳配置并复现
        best_config_overall = df_all.sort_values(by="EDP").iloc[0]
        dse.reproduce_specific_config(best_config_overall)
        
        # (可选) 分别打印每种格式的最佳配置
        for fmt in formats_to_test:
            df_fmt = df_all[df_all['Data_Format'] == fmt]
            if not df_fmt.empty:
                best_config_fmt = df_fmt.sort_values(by="EDP").iloc[0]
                print(f"\n--- Best EDP config for {fmt} ---")
                print(best_config_fmt)
    else:
        print("\n!!! 错误：所有数据格式的仿真均未产生有效结果。请检查上方的 [ERROR] 信息进行修复 !!!")