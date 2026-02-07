"""
Cube Core Simulator

支持 INT8 / INT16 / FP16 等数据格式的 Cube 仿真器
包含内存空间管理和累加逻辑
"""
import math
import itertools
from typing import Dict, Any, Union, Tuple, Optional

from ..hardware.cube import CubeSpec
from ..hardware.dataformat import FORMAT_SPECS, get_cube_format_overheads
from .cube_tiling import CubeTilingEngine


class CubeCoreSimulator:
    """
    Cube Core 仿真器
    
    仿真Cube计算核心的性能和能耗，支持：
    - 输入数据格式 (INT8/INT16/FP16)
    - L0级存储层次 (L0A/L0B/L0C)
    - 内存空间管理和累加逻辑
    """
    
    def __init__(
        self,
        macro_spec: CubeSpec,
        size_l0_A_input_buffer_bytes: int,
        size_l0_B_input_buffer_bytes: int,
        size_l0_C_output_buffer_bytes: int,
        bw_l0_to_cube_gbps: float,
        energy_params: Dict[str, float],
        compute_latency: int,
        size_l1_buffer_bytes: int,
        bw_l1_to_l0_gbps: float,
        clock_freq_ghz: float = 1.0,
        data_format: str = 'INT8',
        custom_overheads: Optional[Dict[str, float]] = None,
        enable_dram_write: bool = False,
        verbose: bool = True
    ):
        """
        初始化Cube Core仿真器
        
        Args:
            macro_spec: Cube规格
            size_l0_A_input_buffer_bytes: L0_A输入Buffer大小（字节）
            size_l0_B_input_buffer_bytes: L0_B输入Buffer大小（字节）
            size_l0_C_output_buffer_bytes: L0_C输出Buffer大小（字节）
            bw_l0_to_cube_gbps: L0到Cube带宽（Gbps）
            energy_params: 能量参数字典
            clock_freq_ghz: 时钟频率（GHz），默认1.0
            data_format: 数据格式，默认'INT8'
            custom_overheads: 自定义开销参数
            compute_latency: Cube计算延迟（周期），默认1
            size_l1_buffer_bytes: L1 Buffer大小（字节），默认1MB
            bw_l1_to_l0_gbps: L1到L0带宽（Gbps），默认128.0
            enable_dram_write: 是否计算DRAM写回功耗，默认False（结果只存储在L0/L1缓存中）
            verbose: 是否打印详细信息
        """
        if verbose:
            print(f"--- 初始化Cube Core (Format: {data_format}) ---")
        
        self.macro_spec = macro_spec
        self.size_l0_A_input = size_l0_A_input_buffer_bytes
        self.size_l0_B_input = size_l0_B_input_buffer_bytes
        self.size_l0_C_output = size_l0_C_output_buffer_bytes
        self.size_l1_buffer = size_l1_buffer_bytes
        self.bw_l0_to_cube_gbps = bw_l0_to_cube_gbps
        self.bw_l1_to_l0_gbps = bw_l1_to_l0_gbps
        self.energy_params = energy_params
        self.clock_freq_ghz = clock_freq_ghz
        self.data_format = data_format
        self.compute_latency = compute_latency
        self.enable_dram_write = enable_dram_write
        self.verbose = verbose
        
        # --- 获取数据格式规格 ---
        # Cube只支持INT8、INT16、FP16三种输入数据格式
        supported_formats = {"INT8", "INT16", "FP16"}
        if data_format not in supported_formats:
            raise ValueError(
                f"Unsupported format for Cube: {data_format}. "
                f"Supported formats: {supported_formats}"
            )
        if data_format not in FORMAT_SPECS:
            raise ValueError(f"Format {data_format} not defined in FORMAT_SPECS")
        
        format_spec = FORMAT_SPECS[data_format]
        self.storage_bytes = format_spec.storage_bytes
        self.compute_bits = 8  # 硬件物理位宽固定为8
        self.effective_compute_bits = format_spec.effective_compute_bits
        self.activity_factor = format_spec.get_activity_factor(self.compute_bits)
        
        # --- 获取格式相关开销 ---
        self.overheads = get_cube_format_overheads(data_format, custom_overheads)
        
        # --- 初始化Tiling引擎 ---
        self.tiling_engine = CubeTilingEngine(
            cube_spec=macro_spec,
            size_l0_A=size_l0_A_input_buffer_bytes,
            size_l0_B=size_l0_B_input_buffer_bytes,
            size_l0_C=size_l0_C_output_buffer_bytes,
            storage_bytes=self.storage_bytes
        )
        
        # --- 打印配置信息 ---
        if verbose:
            self._print_config(format_spec)
    
    def _print_config(self, format_spec) -> None:
        """打印仿真器配置信息"""
        print(f"--- Memory Hierarchy Setup ---")
        print(f"  L1 Buffer: {self.size_l1_buffer/1024:.1f} KB")
        print(f"  L0_A Buffer: {self.size_l0_A_input/1024:.1f} KB")
        print(f"  L0_B Buffer: {self.size_l0_B_input/1024:.1f} KB")
        print(f"  L0_C Buffer: {self.size_l0_C_output/1024:.1f} KB")
        print(f"  L1->L0 Bandwidth: {self.bw_l1_to_l0_gbps:.1f} Gbps")
        print(f"--- Data Format Specifics ({self.data_format}) ---")
        print(f"  Storage Precision: {self.storage_bytes:.4f} Bytes/item")
        print(f"  Mantissa/Guard Bits: {format_spec.mantissa_bits} / {format_spec.guard_bits}")
        print(f"  Effective Compute Bits: {self.effective_compute_bits} Bits")
        print(f"  Cube Energy Activity Factor: {self.activity_factor:.2f}")
        print(f"  Compute Latency: {self.compute_latency} cycles")
        print("---------------------------------------------------\n")
    
    def _derive_tile_sizes(
        self,
        M: int,
        N: int,
        K: int,
        partition_strategy: Union[str, Tuple[int, int]],
        loop_order: str
    ) -> Tuple[Dict[str, int], Dict[str, int], Tuple[int, int]]:
        """
        计算Tile大小（委托给TilingEngine）
        """
        return self.tiling_engine.derive_tile_sizes(M, N, K, partition_strategy)
    
    def simulate(
        self,
        M: int,
        N: int,
        K: int,
        partition_strategy: Union[str, Tuple[int, int]],
        loop_order: str
    ) -> Dict[str, Any]:
        """
        执行仿真
        
        Args:
            M: 矩阵M维度
            N: 矩阵N维度
            K: 矩阵K维度
            partition_strategy: 并行划分策略（Cube架构不支持并行，此参数仅保留用于接口兼容性）
            loop_order: 循环顺序（如'MKN', 'KMN'等）
            
        Returns:
            仿真结果字典，包含延迟、能耗、吞吐量等指标
        """
        # 注意：Cube架构中L1只对应1个core，L0也只对应这1个core，没有并行策略
        # pm和pn始终为(1, 1)，此处解包仅用于接口兼容性
        l0_dims, l1_dims, (pm, pn) = self._derive_tile_sizes(
            M, N, K, partition_strategy, loop_order
        )
        
        # 初始化流水线时间戳
        t_avail = {
            "L0_A": 0,  # L0_A 可用时间
            "L0_B": 0,  # L0_B 可用时间
            "L0_C": 0,
            "Cube": 0,
            "Accumulate": 0,
        }
        
        t_l0_A_space_available = 0  
        t_l0_B_space_available = 0 
        t_l0_C_space_available = 0
        
        # L0_A空间占用跟踪：记录每个A tile在L0_A中的空间占用情况
        # {(m_start, k_start): t_space_available} 每个A tile的空间可用时间
        l0_A_tile_space = {}
        
        # L0_B空间占用跟踪：记录每个B tile在L0_B中的空间占用情况
        # {(k_start, n_start): t_space_available} 每个B tile的空间可用时间
        l0_B_tile_space = {}
        
        # L0_C空间占用跟踪：记录每个C tile在L0_C中的空间占用情况
        # {(m_start, n_start): (t_space_available, bytes)} 每个C tile的空间可用时间和大小
        l0_C_tile_space = {}
        
        # 初始化能耗计数器
        # 注意：L0_A和L0_B数据已预先加载，不需要load_A_L0和load_B_L0
        energies_pj = {k: 0 for k in [
            "load_A_L1", "load_B_L1",  # 从L1加载到L0
            "read_A_L0", "read_B_L0",  # 从L0读取到Cube
            "cube_compute", "accumulate", "write_C_L0", "read_C_L0", "write_C_L1", "dram_write"
        ]}
        
        # L1写回时间戳跟踪（用于L0_C空间管理）
        t_avail["L1"] = 0
        
        # === 调试统计：记录L0_C写入L1的次数 ===
        l0c_to_l1_stats = {
            'total_evict_count': 0,  # 总写回次数
            'total_evict_bytes': 0,  # 总写回数据量（字节）
            'tile_evict_count': {},  # 每个tile被写回的次数 {(m_start, n_start): count}
            'tile_evict_bytes': {},  # 每个tile被写回的总数据量 {(m_start, n_start): bytes}
        }
        
        # 计算带宽（字节/周期）
        bw_l0_to_cube_bpc = (self.bw_l0_to_cube_gbps * 1e9) / (self.clock_freq_ghz * 1e9) / 8
        bw_l1_to_l0_bpc = (self.bw_l1_to_l0_gbps * 1e9) / (self.clock_freq_ghz * 1e9) / 8
        # L0到L1的写回带宽（假设与L1到L0的读取带宽相同）
        bw_l0_to_l1_bpc = bw_l1_to_l0_bpc
        
        # Cube维度
        cube_m = self.macro_spec.m_dim
        cube_n = self.macro_spec.n_dim
        cube_k = self.macro_spec.k_dim
        
        # Tile步长
        step_m, step_n, step_k = l0_dims['m'], l0_dims['n'], l0_dims['k']
        
        # 生成循环迭代器
        ranges = {
            'M': range(0, M, step_m),
            'N': range(0, N, step_n),
            'K': range(0, K, step_k)
        }
        
        # 调试：打印tile大小和ranges
        if self.verbose:
            print(f"\n[Tile配置]")
            print(f"  step_m={step_m}, step_n={step_n}, step_k={step_k}")
            print(f"  M range: {list(ranges['M'])} (共{len(list(ranges['M']))}个)")
            print(f"  N range: {list(ranges['N'])} (共{len(list(ranges['N']))}个)")
            print(f"  K range: {list(ranges['K'])} (共{len(list(ranges['K']))}个)")
            print(f"  循环顺序: {loop_order}")
            total_tiles = len(list(ranges['M'])) * len(list(ranges['N'])) * len(list(ranges['K']))
            print(f"  总tile数: {total_tiles}")
            total_C_tiles = len(list(ranges['M'])) * len(list(ranges['N']))
            print(f"  总C tile数: {total_C_tiles}")
        
        outer_iter = itertools.product(
            ranges[loop_order[0]],
            ranges[loop_order[1]],
            ranges[loop_order[2]]
        )
        
        # === Accum状态管理 ===
        # accum跟踪当前正在累加的C tile，以及每个C tile的K维度完成情况
        accum_state = {
            'current_C_tile': None,  # (m_start, n_start) 当前正在累加的C tile
            'C_tile_K_progress': {}  # {(m_start, n_start): set(k_start, ...)} 每个C tile已完成的K维度
        }
        
        # === A和B矩阵缓存跟踪 ===
        # 跟踪当前cube计算使用的A和B数据，避免重复读取
        current_A_tag = None  # (m_offset, k_offset) 当前使用的A数据标签
        current_B_tag = None  # (k_offset, n_offset) 当前使用的B数据标签
        
        # === 外层循环: Tile迭代 ===
        # 调试：统计所有C tile
        all_C_tiles = set()
        tile_count = 0
        
        for tile_indices in outer_iter:
            tile_count += 1
            idx_map = {
                loop_order[0]: tile_indices[0],
                loop_order[1]: tile_indices[1],
                loop_order[2]: tile_indices[2]
            }
            m_start = idx_map['M']
            n_start = idx_map['N']
            k_start = idx_map['K']
            
            m_actual = min(step_m, M - m_start)
            n_actual = min(step_n, N - n_start)
            k_actual = min(step_k, K - k_start)
            
            # 当前C tile标签
            C_tile_tag = (m_start, n_start)
            all_C_tiles.add(C_tile_tag)
            
            if self.verbose and tile_count <= 10:  # 只打印前10个tile
                print(f"  [Tile迭代] #{tile_count}: C tile {C_tile_tag}, m_start={m_start}, n_start={n_start}, k_start={k_start}, "
                      f"m_actual={m_actual}, n_actual={n_actual}, k_actual={k_actual}")
            
            # 初始化C tile的K进度跟踪
            if C_tile_tag not in accum_state['C_tile_K_progress']:
                accum_state['C_tile_K_progress'][C_tile_tag] = set()
            
            # 检查是否有之前的K维度记录（在添加当前k_start之前）
            # 这用于判断是否需要从L0_C读取之前的结果
            has_previous_K = len(accum_state['C_tile_K_progress'][C_tile_tag]) > 0
            
            # === 从L1加载数据到L0_A和L0_B ===
            # 计算当前tile需要的数据大小
            bytes_A_tile = m_actual * k_actual * self.storage_bytes
            bytes_B_tile = n_actual * k_actual * self.storage_bytes
            
            # A tile标签：(m_start, k_start)
            A_tile_tag = (m_start, k_start)
            # B tile标签：(k_start, n_start)
            B_tile_tag = (k_start, n_start)
            
            # 检查A tile是否已经在L0_A中（数据还在，未被释放）
            # l0_A_tile_space[A_tile_tag] 记录的是该tile空间何时可用（何时可以释放）
            # 如果当前时间 < 空间释放时间，说明数据还在L0_A中
            current_time = max(t_avail["L0_A"], t_l0_A_space_available)
            A_tile_in_L0 = (A_tile_tag in l0_A_tile_space and 
                           current_time < l0_A_tile_space[A_tile_tag])
            
            if A_tile_in_L0:
                # A tile已在L0_A中，数据可用，不需要从L1加载（缓存命中，功耗为0）
                # 但需要等待L0_A可用（可能正在被其他操作使用）
                start_load_A = max(t_avail["L0_A"], current_time)
                t_avail["L0_A"] = start_load_A
                # 注意：缓存命中时不需要从L1加载，因此不记录load_A_L1功耗
            else:
                # A tile不在L0_A中或已被释放，需要从L1加载
                # 需要等待L0_A全局空间可用
                start_load_A = max(t_avail["L0_A"], t_l0_A_space_available)
                dur_load_A = math.ceil(bytes_A_tile / bw_l1_to_l0_bpc)
                t_avail["L0_A"] = start_load_A + dur_load_A
                energies_pj["load_A_L1"] += bytes_A_tile * self.energy_params.get("l1_sram_read_power", 0.5)
                # 更新该A tile的空间占用时间（加载完成后空间可被释放或复用）
                l0_A_tile_space[A_tile_tag] = t_avail["L0_A"]
            
            # 检查B tile是否已经在L0_B中
            current_time = max(t_avail["L0_B"], t_l0_B_space_available)
            B_tile_in_L0 = (B_tile_tag in l0_B_tile_space and 
                           current_time < l0_B_tile_space[B_tile_tag])
            
            if B_tile_in_L0:
                # B tile已在L0_B中，数据可用，不需要从L1加载（缓存命中，功耗为0）
                start_load_B = max(t_avail["L0_B"], current_time)
                t_avail["L0_B"] = start_load_B
                # 注意：缓存命中时不需要从L1加载，因此不记录load_B_L1功耗
            else:
                # B tile不在L0_B中或已被释放，需要从L1加载
                start_load_B = max(t_avail["L0_B"], t_l0_B_space_available)
                dur_load_B = math.ceil(bytes_B_tile / bw_l1_to_l0_bpc)
                t_avail["L0_B"] = start_load_B + dur_load_B
                energies_pj["load_B_L1"] += bytes_B_tile * self.energy_params.get("l1_sram_read_power", 0.5)
                # 更新该B tile的空间占用时间（加载完成后空间可被释放或复用）
                l0_B_tile_space[B_tile_tag] = t_avail["L0_B"]
            
            # === LEVEL 1: Cube计算 ===
            # 遍历所有cube计算单元
            for m_cube in range(0, m_actual, cube_m):
                for n_cube in range(0, n_actual, cube_n):
                    for k_cube in range(0, k_actual, cube_k):
                        m_cube_actual = min(cube_m, m_actual - m_cube)
                        n_cube_actual = min(cube_n, n_actual - n_cube)
                        k_cube_actual = min(cube_k, k_actual - k_cube)
                        
                        # 当前cube使用的A和B数据标签
                        # A矩阵标签：(m_start + m_cube, k_start + k_cube)
                        # B矩阵标签：(k_start + k_cube, n_start + n_cube)
                        A_tag = (m_start + m_cube, k_start + k_cube)
                        B_tag = (k_start + k_cube, n_start + n_cube)
                        
                        # 判断是否需要读取A和B
                        need_read_A = (current_A_tag != A_tag)
                        need_read_B = (current_B_tag != B_tag)
                        
                        # 从L0读取A和B（单次取出进行计算的矩阵大小）
                        # A: cube_dim_m * cube_dim_k * in_bytes
                        # B: cube_dim_n * cube_dim_k * in_bytes
                        bytes_A_read = m_cube_actual * k_cube_actual * self.storage_bytes if need_read_A else 0
                        bytes_B_read = n_cube_actual * k_cube_actual * self.storage_bytes if need_read_B else 0
                        
                        # 读取需要等待L0_A/L0_B数据可用以及Cube可用
                        start_read = max(t_avail["Cube"], t_avail["L0_A"], t_avail["L0_B"])
                        dur_read_A = math.ceil(bytes_A_read / bw_l0_to_cube_bpc) if need_read_A else 0
                        dur_read_B = math.ceil(bytes_B_read / bw_l0_to_cube_bpc) if need_read_B else 0
                        dur_read = max(dur_read_A, dur_read_B)
                        
                        # 记录读取操作的能耗
                        if need_read_A:
                            energies_pj["read_A_L0"] += bytes_A_read * self.energy_params.get("l0_sram_read_power", 0)
                            # 更新L0_A的最后使用时间（追踪从L0读取到Cube的操作）
                            t_avail["L0_A"] = max(t_avail["L0_A"], start_read + dur_read_A)
                        # 注意：如果不需要读取新数据（数据已在Cube中），则不记录read_A_L0功耗
                        if need_read_B:
                            energies_pj["read_B_L0"] += bytes_B_read * self.energy_params.get("l0_sram_read_power", 0)
                            # 更新L0_B的最后使用时间（追踪从L0读取到Cube的操作）
                            t_avail["L0_B"] = max(t_avail["L0_B"], start_read + dur_read_B)
                        # 注意：如果不需要读取新数据（数据已在Cube中），则不记录read_B_L0功耗
                        
                        # 更新当前使用的A和B标签
                        if need_read_A:
                            current_A_tag = A_tag
                        if need_read_B:
                            current_B_tag = B_tag
                        
                        # Cube计算（经过compute_latency周期后写到accum）
                        start_compute = start_read + dur_read
                        t_avail["Cube"] = start_compute + self.compute_latency
                        
                        # 计算能耗
                        ops_cube = m_cube_actual * n_cube_actual * k_cube_actual
                        base_compute_energy = ops_cube * self.energy_params.get("cube_compute_power", 0)
                        energies_pj["cube_compute"] += (
                            base_compute_energy *
                            self.overheads.get('cube_energy_scale', 1.0)
                        )
                        
            # === LEVEL 4: Accum处理（判断权交由accum） ===
            # 计算整个tile的累加数据大小（不是单个cube单元）
            bytes_C_tile_accum = m_actual * n_actual * 4  # 输出32位，整个tile大小
            
            # Accum判断：是否是同一个C矩阵？
            # 注意：使用C_tile_K_progress来判断是否是同一个C tile的连续K迭代
            # 而不是使用accum_state['current_C_tile']，因为循环顺序可能导致C tile切换
            is_first_K_chunk = (k_start == 0)
            
            # 判断是否需要从L0_C读取：
            # 1. 不是第一次K迭代（k_start != 0）
            # 2. 该C tile已经有之前的K迭代记录（说明之前已经写入过L0_C）
            # 注意：has_previous_K是在循环开始时检查的，此时还没有添加当前k_start
            if not is_first_K_chunk and has_previous_K:
                # 同一个C矩阵的后续K迭代，需要累加：需要从L0_C读取之前的结果进行累加
                # 检查该C tile是否已经在L0_C中
                if C_tile_tag in l0_C_tile_space:
                    # 对于同一个C tile的连续K迭代，数据应该一直在L0_C中
                    # 除非被写回L1（但写回L1会从l0_C_tile_space中删除）
                    tile_info = l0_C_tile_space[C_tile_tag]
                    if isinstance(tile_info, tuple):
                        write_complete_time, tile_bytes = tile_info
                    else:
                        write_complete_time = tile_info
                        tile_bytes = m_actual * n_actual * 4  # 兼容旧格式
                    
                    # 需要等待上一次写入L0_C完成
                    # 然后从L0_C读取整个tile进行累加
                    start_accum = max(t_avail["Cube"], t_avail["Accumulate"], write_complete_time)
                    
                    # 从L0_C读取整个tile
                    bytes_C_read = bytes_C_tile_accum
                    energies_pj["read_C_L0"] += bytes_C_read * self.energy_params.get("l0_sram_read_power", 0)
                    
                    # 更新L0_C的最后使用时间（追踪从L0_C读取的操作）
                    t_avail["L0_C"] = max(t_avail["L0_C"], start_accum)
                else:
                    # C tile不在L0_C中（可能已被写回L1）
                    # 需要从L1读取（这里简化处理，假设数据在L1中）
                    start_accum = max(t_avail["Cube"], t_avail["Accumulate"], t_avail["L1"])
                    if self.verbose:
                        print(f"  [WARNING] K迭代 {k_start}: C tile {C_tile_tag} 不在L0_C中（可能已写回L1），从L1读取")
            else:
                # 新的C tile或第一次K迭代，直接使用Cube结果（初始化accum），不需要从L0_C读取
                start_accum = max(t_avail["Cube"], t_avail["Accumulate"])
                accum_state['current_C_tile'] = C_tile_tag
                # 注意：第一次K迭代时不需要从L0_C读取，因此不记录read_C_L0功耗
            
            # 累加操作
            dur_accum = 1  # 累加延迟
            t_avail["Accumulate"] = start_accum + dur_accum
            
            # 累加操作是对整个tile进行的，计算所有cube单元的总数据量
            total_cube_ops = 0
            for m_cube in range(0, m_actual, cube_m):
                for n_cube in range(0, n_actual, cube_n):
                    m_cube_actual = min(cube_m, m_actual - m_cube)
                    n_cube_actual = min(cube_n, n_actual - n_cube)
                    total_cube_ops += m_cube_actual * n_cube_actual
            
            base_accum_energy = total_cube_ops * 4  # 每个元素4字节
            # 累加操作是对32位结果进行的，不依赖于输入数据格式
            energies_pj["accumulate"] += (
                base_accum_energy *
                self.energy_params.get("accmu_flop_power", 0)
            )
            
            # === 每次K迭代完成后，将累加结果写入L0_C ===
            # 这样后续K迭代可以从L0_C读取进行累加
            bytes_C_tile = m_actual * n_actual * 4
            
            # === L0_C空间管理：检查空间是否足够 ===
            # 计算当前L0_C中已占用的空间（只计算当前时间仍占用的tile）
            current_time = max(t_avail["Accumulate"], t_l0_C_space_available)
            occupied_space = 0
            tiles_to_evict = []  # 需要写回L1的已完成C tile列表
            
            # 遍历所有在L0_C中的C tile，计算当前仍占用的空间
            # 注意：对于已完成的C tile，数据会一直存在，直到被写回L1或覆盖
            # space_release_time 表示写入完成时间，但数据会一直占用空间直到被释放
            for tile_tag, tile_info in l0_C_tile_space.items():
                if isinstance(tile_info, tuple):
                    write_complete_time, tile_bytes = tile_info
                else:
                    write_complete_time = tile_info
                    tile_bytes = bytes_C_tile  # 兼容旧格式
                
                # 如果写入已完成（current_time >= write_complete_time），数据仍在L0_C中
                # 只有当数据被写回L1或覆盖时，才会从l0_C_tile_space中删除
                # 因此，只要tile在l0_C_tile_space中，就认为它仍在占用空间
                if current_time >= write_complete_time:
                    # 写入已完成，数据仍在L0_C中（除非被写回L1）
                    # 关键：如果当前tile已经在L0_C中，可以复用空间，不需要额外空间
                    # 因此，当前tile不应该计入占用空间（因为可以覆盖写入）
                    if tile_tag != C_tile_tag:
                        occupied_space += tile_bytes
                        # 关键：只有已经完全完成所有K维度计算的C tile才能被写回L1
                        # 在计算同一个大的C tiling矩阵时，正在计算中的C tile不应该被写回L1
                        # 检查该tile是否已经完成所有K维度
                        expected_K_chunks = set(range(0, K, step_k))
                        completed_K_chunks = accum_state['C_tile_K_progress'].get(tile_tag, set())
                        is_tile_complete = (expected_K_chunks.issubset(completed_K_chunks))
                        
                        # 只有完全完成的tile才能被写回L1
                        if is_tile_complete:
                            tiles_to_evict.append((tile_tag, tile_bytes, write_complete_time))
                    # else: 当前tile可以复用空间，不计入占用空间
                else:
                    # 写入还未完成，但空间已被分配，仍占用空间
                    # 同样，如果是当前tile，可以复用，不计入占用空间
                    if tile_tag != C_tile_tag:
                        occupied_space += tile_bytes
            
            # 检查是否有足够的空间
            available_space = self.size_l0_C_output - occupied_space
            needs_eviction = (available_space < bytes_C_tile)
            
            # === 调试：记录空间不足但无tile可写回的情况 ===
            if needs_eviction and not tiles_to_evict and self.verbose:
                # 详细分析L0_C中的tile状态
                tile_status = []
                for tag, info in l0_C_tile_space.items():
                    if isinstance(info, tuple):
                        wct, tbytes = info
                    else:
                        wct = info
                        tbytes = bytes_C_tile
                    completed_K = accum_state['C_tile_K_progress'].get(tag, set())
                    expected_K = set(range(0, K, step_k))
                    is_complete = (expected_K.issubset(completed_K))
                    tile_status.append(f"{tag}: {tbytes/1024:.1f}KB, complete={is_complete}, K={sorted(completed_K)}")
                
                print(f"  [L0_C空间管理警告] C tile {C_tile_tag} 需要空间 {bytes_C_tile/1024:.1f} KB, "
                      f"可用空间 {available_space/1024:.1f} KB, 已占用 {occupied_space/1024:.1f} KB, "
                      f"但无已完成的tile可写回。")
                print(f"    L0_C中的tile状态: {tile_status}")
            
            # 如果空间不足，先将已完成的C tile写回L1
            if needs_eviction and tiles_to_evict:
                # 按空间释放时间排序，优先写回最早完成的tile
                tiles_to_evict.sort(key=lambda x: x[2])
                
                # 写回tile直到有足够空间
                total_evicted_bytes = 0
                for evict_tile_tag, evict_bytes, evict_release_time in tiles_to_evict:
                    if available_space + total_evicted_bytes >= bytes_C_tile:
                        break
                    
                    # 计算写回L1的时间
                    # 注意：evict_release_time 是写入完成时间，但数据仍在L0_C中
                    # 需要等待写入完成后才能写回L1
                    # 写回L1的过程包括：从L0_C读取数据 → 写入L1
                    start_evict = max(t_avail["L0_C"], t_avail["L1"], evict_release_time)
                    # 从L0_C读取数据的时间（假设与写入L1的时间相同，因为数据流是连续的）
                    dur_read_from_l0 = math.ceil(evict_bytes / bw_l0_to_l1_bpc)
                    dur_evict = dur_read_from_l0  # 写回L1的总时间
                    t_avail["L1"] = start_evict + dur_evict
                    # 更新L0_C的最后使用时间（追踪从L0_C读取的操作）
                    t_avail["L0_C"] = max(t_avail["L0_C"], start_evict + dur_read_from_l0)
                    
                    # 记录写回L1的功耗
                    # 注意：写回L1需要先从L0_C读取数据，再写入L1
                    # 因此需要记录read_C_L0和write_C_L1两种功耗
                    energies_pj["read_C_L0"] += evict_bytes * self.energy_params.get("l0_sram_read_power", 0)
                    energies_pj["write_C_L1"] += evict_bytes * self.energy_params.get("l1_sram_write_power", 
                                                                                      self.energy_params.get("l1_sram_read_power", 0.5))
                    
                    # === 调试统计：记录写回L1的次数和数据量 ===
                    l0c_to_l1_stats['total_evict_count'] += 1
                    l0c_to_l1_stats['total_evict_bytes'] += evict_bytes
                    if evict_tile_tag not in l0c_to_l1_stats['tile_evict_count']:
                        l0c_to_l1_stats['tile_evict_count'][evict_tile_tag] = 0
                        l0c_to_l1_stats['tile_evict_bytes'][evict_tile_tag] = 0
                    l0c_to_l1_stats['tile_evict_count'][evict_tile_tag] += 1
                    l0c_to_l1_stats['tile_evict_bytes'][evict_tile_tag] += evict_bytes
                    
                    # 从L0_C空间跟踪中移除（空间已释放）
                    del l0_C_tile_space[evict_tile_tag]
                    total_evicted_bytes += evict_bytes
                    
                    if self.verbose:
                        print(f"  [L0_C空间管理] C tile {evict_tile_tag} 写回L1，释放 {evict_bytes} 字节 ({evict_bytes/1024:.1f} KB) "
                              f"[累计: {l0c_to_l1_stats['tile_evict_count'][evict_tile_tag]}次, "
                              f"{l0c_to_l1_stats['tile_evict_bytes'][evict_tile_tag]/1024:.1f} KB]")
            
            # 检查该C tile是否已经在L0_C中占用空间
            if C_tile_tag in l0_C_tile_space:
                # 相同的C tile，复用/覆盖同一块空间（更新累加结果）
                tile_info = l0_C_tile_space[C_tile_tag]
                if isinstance(tile_info, tuple):
                    tile_space_available, _ = tile_info
                else:
                    tile_space_available = tile_info
                start_write_output = max(t_avail["Accumulate"], tile_space_available, t_avail["L1"])
            else:
                # 不同的C tile，分配新的L0_C空间
                # 需要等待L0_C全局空间可用和L1写回完成
                start_write_output = max(t_avail["Accumulate"], t_l0_C_space_available, t_avail["L1"])
                # 记录该C tile占用的空间（包含大小信息）
                l0_C_tile_space[C_tile_tag] = (start_write_output, bytes_C_tile)
            
            dur_write_output = math.ceil(bytes_C_tile / bw_l0_to_cube_bpc)
            t_avail["L0_C"] = start_write_output + dur_write_output
            # 记录写回L0_C的功耗
            energies_pj["write_C_L0"] += bytes_C_tile * self.energy_params.get("l0_sram_write_power", 0)
            
            # 更新该C tile的空间占用时间（写入完成后空间可被释放或复用）
            # 保存为元组：(写入完成时间, tile大小)
            # 注意：对于同一个C tile，写入完成时间会被更新，表示数据的最新写入时间
            l0_C_tile_space[C_tile_tag] = (t_avail["L0_C"], bytes_C_tile)
            
            # 记录当前K维度已完成（在写入L0_C之后）
            # 这样下次K迭代时，has_previous_K会为True，可以触发read_C_L0
            accum_state['C_tile_K_progress'][C_tile_tag].add(k_start)
            
            # === Accum判断：是否完成当前C tile的所有K维度？ ===
            # 计算该C tile应该有多少个K chunk
            expected_K_chunks = set(range(0, K, step_k))
            completed_K_chunks = accum_state['C_tile_K_progress'][C_tile_tag]
            is_C_tile_complete = (expected_K_chunks.issubset(completed_K_chunks))
            
            if is_C_tile_complete:
                # 已经算完一个完整的C tile，后续不需要再累加
                # 注意：数据已经在每次K迭代后写入L0_C（见上面的代码），这里只需要重置状态
                # 重置accum状态，准备下一个C tile
                accum_state['current_C_tile'] = None
            
            # === [关键更新] Update Space Availability ===
            # L0_A的全局空间可用性：取所有A tile中最早释放的空间时间
            if l0_A_tile_space:
                t_l0_A_space_available = min(l0_A_tile_space.values())
            else:
                t_l0_A_space_available = t_avail["L0_A"]
            
            # L0_B的全局空间可用性：取所有B tile中最早释放的空间时间
            if l0_B_tile_space:
                t_l0_B_space_available = min(l0_B_tile_space.values())
            else:
                t_l0_B_space_available = t_avail["L0_B"]
            
            # L0_C的全局空间可用性：取所有C tile中最早释放的空间时间
            if l0_C_tile_space:
                # 处理新的元组格式和旧的单值格式
                release_times = []
                for tile_info in l0_C_tile_space.values():
                    if isinstance(tile_info, tuple):
                        release_times.append(tile_info[0])
                    else:
                        release_times.append(tile_info)
                t_l0_C_space_available = min(release_times)
            else:
                t_l0_C_space_available = t_avail["L0_C"]
        
        # === [修复] 循环结束后，将L0_C中所有剩余的已完成C tile写回L1 ===
        # 在循环过程中，只有当L0_C空间不足时才会写回C tile
        # 如果空间足够，已完成的C tile会一直留在L0_C中
        # 因此需要在循环结束后，将所有剩余的已完成C tile写回L1
        expected_K_chunks = set(range(0, K, step_k))
        remaining_tiles_to_evict = []
        for tile_tag, tile_info in l0_C_tile_space.items():
            if isinstance(tile_info, tuple):
                write_complete_time, tile_bytes = tile_info
            else:
                write_complete_time = tile_info
                # 需要从accum_state中获取tile大小，或者使用默认值
                # 这里使用一个合理的默认值，或者从已完成的tile中推断
                # 由于循环已结束，我们可以使用step_m和step_n来计算
                tile_bytes = step_m * step_n * 4  # 每个C tile的大小：M*N*4字节
            
            # 检查该tile是否已经完成所有K维度
            completed_K_chunks = accum_state['C_tile_K_progress'].get(tile_tag, set())
            is_tile_complete = (expected_K_chunks.issubset(completed_K_chunks))
            
            if is_tile_complete:
                remaining_tiles_to_evict.append((tile_tag, tile_bytes, write_complete_time))
        
        # 按空间释放时间排序，写回所有剩余的已完成C tile
        if remaining_tiles_to_evict:
            remaining_tiles_to_evict.sort(key=lambda x: x[2])
            for evict_tile_tag, evict_bytes, evict_release_time in remaining_tiles_to_evict:
                # 计算写回L1的时间
                start_evict = max(t_avail["L0_C"], t_avail["L1"], evict_release_time)
                dur_read_from_l0 = math.ceil(evict_bytes / bw_l0_to_l1_bpc)
                dur_evict = dur_read_from_l0
                t_avail["L1"] = start_evict + dur_evict
                t_avail["L0_C"] = max(t_avail["L0_C"], start_evict + dur_read_from_l0)
                
                # 记录写回L1的功耗
                energies_pj["read_C_L0"] += evict_bytes * self.energy_params.get("l0_sram_read_power", 0)
                energies_pj["write_C_L1"] += evict_bytes * self.energy_params.get("l1_sram_write_power", 
                                                                                  self.energy_params.get("l1_sram_read_power", 0.5))
                
                # === 调试统计：记录写回L1的次数和数据量 ===
                l0c_to_l1_stats['total_evict_count'] += 1
                l0c_to_l1_stats['total_evict_bytes'] += evict_bytes
                if evict_tile_tag not in l0c_to_l1_stats['tile_evict_count']:
                    l0c_to_l1_stats['tile_evict_count'][evict_tile_tag] = 0
                    l0c_to_l1_stats['tile_evict_bytes'][evict_tile_tag] = 0
                l0c_to_l1_stats['tile_evict_count'][evict_tile_tag] += 1
                l0c_to_l1_stats['tile_evict_bytes'][evict_tile_tag] += evict_bytes
                
                # 从L0_C空间跟踪中移除（空间已释放）
                del l0_C_tile_space[evict_tile_tag]
                
                if self.verbose:
                    print(f"  [L0_C空间管理] C tile {evict_tile_tag} 写回L1（循环结束），释放 {evict_bytes} 字节 ({evict_bytes/1024:.1f} KB) "
                          f"[累计: {l0c_to_l1_stats['tile_evict_count'][evict_tile_tag]}次, "
                          f"{l0c_to_l1_stats['tile_evict_bytes'][evict_tile_tag]/1024:.1f} KB]")
        
        # === 循环结束: 最终结果写回 (DRAM Write) ===
        # 注意：只有在结果确实需要写回DRAM时才计算此功耗
        # 如果结果只存储在L0/L1缓存中，则不需要计算dram_write功耗
        if self.enable_dram_write:
            output_bytes = 4  # 最终输出总是32位
            total_out_bytes = M * N * output_bytes
            energies_pj["dram_write"] = total_out_bytes * self.energy_params.get("dram_write_power", 0)
        else:
            energies_pj["dram_write"] = 0
        
        # 调试：打印所有C tile统计
        if self.verbose:
            print(f"\n[C Tile统计]")
            print(f"  处理的C tile总数: {len(all_C_tiles)}")
            print(f"  所有C tile: {sorted(all_C_tiles)}")
            print(f"  总tile迭代次数: {tile_count}")
        
        # === 调试统计：输出L0_C写入L1的统计信息 ===
        if self.verbose:
            print(f"\n[L0_C → L1 写回统计]")
            print(f"  总写回次数: {l0c_to_l1_stats['total_evict_count']}")
            print(f"  总写回数据量: {l0c_to_l1_stats['total_evict_bytes']:,} 字节 "
                  f"({l0c_to_l1_stats['total_evict_bytes']/1024:.1f} KB, "
                  f"{l0c_to_l1_stats['total_evict_bytes']/(1024*1024):.2f} MB)")
            print(f"  理论最小写回数据量: {M * N * 4:,} 字节 ({M * N * 4/1024:.1f} KB)")
            if l0c_to_l1_stats['total_evict_bytes'] > 0:
                ratio = l0c_to_l1_stats['total_evict_bytes'] / (M * N * 4)
                print(f"  实际/理论比例: {ratio:.2f}x")
            
            if l0c_to_l1_stats['tile_evict_count']:
                print(f"\n  每个C tile的写回详情:")
                for tile_tag in sorted(l0c_to_l1_stats['tile_evict_count'].keys()):
                    count = l0c_to_l1_stats['tile_evict_count'][tile_tag]
                    bytes_written = l0c_to_l1_stats['tile_evict_bytes'][tile_tag]
                    print(f"    C tile {tile_tag}: {count}次, {bytes_written:,} 字节 "
                          f"({bytes_written/1024:.1f} KB)")
            else:
                print(f"  没有C tile被写回L1（L0_C空间足够）")
        
        # === 汇总结果 ===
        total_latency = max(t_avail.values())
        total_energy_pj = sum(energies_pj.values())
        total_ops = M * N * K * 2  # MAC operation is 2 ops (multiply and add)
        throughput = (total_ops / total_latency * self.clock_freq_ghz) if total_latency > 0 else 0
        
        return {
            "Format": self.data_format,
            "L0_Tile_M": l0_dims['m'],
            "L0_Tile_N": l0_dims['n'],
            "L0_Tile_K": l0_dims['k'],
            "latencies": t_avail,
            "energies_pj": energies_pj,
            "total_latency_cycles": total_latency,
            "total_energy_pj": total_energy_pj,
            "throughput_gops": throughput,
            "energy_efficiency_tops_per_watt": (
                (total_ops / 1e12) / (total_energy_pj / 1e12) if total_energy_pj > 0 else 0
            ),
            "bottleneck": max(t_avail, key=t_avail.get),
            # 调试统计：L0_C写入L1的统计信息
            "l0c_to_l1_stats": {
                "total_evict_count": l0c_to_l1_stats['total_evict_count'],
                "total_evict_bytes": l0c_to_l1_stats['total_evict_bytes'],
                "tile_evict_count": l0c_to_l1_stats['tile_evict_count'],
                "tile_evict_bytes": l0c_to_l1_stats['tile_evict_bytes'],
                "theoretical_min_bytes": M * N * 4,
                "actual_to_theoretical_ratio": (
                    l0c_to_l1_stats['total_evict_bytes'] / (M * N * 4) 
                    if (M * N * 4) > 0 else 0
                )
            }
        }
