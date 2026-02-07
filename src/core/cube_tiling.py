"""
Cube Tiling Engine

计算Cube的Tiling策略引擎
"""
from typing import Tuple, Union, Dict

from ..hardware.cube import CubeSpec


class CubeTilingEngine:
    """
    Cube Tiling策略引擎
    
    根据Cube硬件规格和L0 Buffer容量计算最优的Tile大小
    
    注意：Cube架构中L1只对应1个core，L0也只对应这1个core，
    因此不存在并行策略（partition_strategy），所有计算都是单核心执行。
    """
    
    def __init__(
        self,
        cube_spec: CubeSpec,
        size_l0_A: int,
        size_l0_B: int,
        size_l0_C: int,
        storage_bytes: float
    ):
        """
        初始化Cube Tiling引擎
        
        Args:
            cube_spec: Cube规格
            size_l0_A: L0 A输入Buffer大小（字节）
            size_l0_B: L0 B输入Buffer大小（字节）
            size_l0_C: L0 C输出Buffer大小（字节）
            storage_bytes: 每元素存储字节数
        """
        self.cube_spec = cube_spec
        self.size_l0_A = size_l0_A
        self.size_l0_B = size_l0_B
        self.size_l0_C = size_l0_C
        self.storage_bytes = storage_bytes
    
    def derive_tile_sizes(
        self,
        M: int,
        N: int,
        K: int,
        partition_strategy: Union[str, Tuple[int, int]]
    ) -> Tuple[Dict[str, int], Dict[str, int], Tuple[int, int]]:
        """
        计算Tile大小
        
        Args:
            M: 矩阵M维度
            N: 矩阵N维度
            K: 矩阵K维度
            partition_strategy: 并行划分策略（Cube架构不支持并行，此参数仅保留用于接口兼容性，会被忽略）
            
        Returns:
            (l0_dims, l1_dims, (pm, pn)) 元组:
            - l0_dims: L0 Tile维度字典 {'m', 'n', 'k'}
            - l1_dims: L1 Tile维度字典 {'m', 'n', 'k'} (与L0相同，Cube只有一级存储)
            - (pm, pn): 并行度，始终为(1, 1)，因为Cube架构L1只对应1个core，L0也只对应这1个core，没有并行策略
        """
        # 注意：Cube架构中L1只对应1个core，L0也只对应这1个core
        # 因此不存在并行策略，partition_strategy参数在此架构中无意义
        # Cube的基础维度
        cube_m = self.cube_spec.m_dim
        cube_n = self.cube_spec.n_dim
        cube_k = self.cube_spec.k_dim
        
        # 计算基于Buffer限制的Tile大小
        # A矩阵: M * K * storage_bytes <= size_l0_A
        m_tile_A = int(self.size_l0_A // (cube_k * self.storage_bytes))
        m_tile_A = max(1, min(m_tile_A, M))
        
        # B矩阵: N * K * storage_bytes <= size_l0_B
        n_tile_B = int(self.size_l0_B // (cube_k * self.storage_bytes))
        n_tile_B = max(1, min(n_tile_B, N))
        
        # C矩阵: M * N * 4 (输出通常是32位) <= size_l0_C
        # 正确的计算：m_tile_C * n_tile_C * 4 <= size_l0_C
        # 假设m_tile_C = n_tile_C（正方形tile），则 m_tile_C^2 * 4 <= size_l0_C
        # m_tile_C <= sqrt(size_l0_C / 4)
        import math
        max_tile_dim_C = int(math.sqrt(self.size_l0_C / 4))
        # 对齐到cube维度
        max_tile_dim_C = (max_tile_dim_C // cube_m) * cube_m if max_tile_dim_C >= cube_m else cube_m
        m_tile_C = min(max_tile_dim_C, M)
        n_tile_C = min(max_tile_dim_C, N)
        
        # 取最小值以确保所有Buffer都能容纳
        m_tile = min(m_tile_A, m_tile_C, M)
        n_tile = min(n_tile_B, n_tile_C, N)
        k_tile = cube_k  # K维度使用Cube的K维度
        
        # 对齐到Cube维度
        m_tile = (m_tile // cube_m) * cube_m if m_tile >= cube_m else cube_m
        n_tile = (n_tile // cube_n) * cube_n if n_tile >= cube_n else cube_n
        k_tile = (k_tile // cube_k) * cube_k if k_tile >= cube_k else cube_k
        
        l0_dims = {
            'm': max(1, m_tile),
            'n': max(1, n_tile),
            'k': max(1, k_tile)
        }
        
        l1_dims = l0_dims.copy()
        
        # 返回并行度 (1, 1)，因为Cube架构不支持并行
        return l0_dims, l1_dims, (1, 1)
