"""
Tiling Engine

计算两级Tile大小（L0/L1）的策略引擎
"""
from typing import Tuple, Union, Dict

from ..hardware.macro import CIMMacroSpec


class TilingEngine:
    """
    Tiling策略引擎

    根据硬件规格和存储层次计算最优的Tile大小，支持两级Tiling：
    - L0 Tile (Sub Tile): 受限于单个Macro的L0 Buffer容量
    - L1 Tile (Super Tile): 受限于Core的L1 Buffer容量
    """

    def __init__(
        self,
        macro_spec: CIMMacroSpec,
        n_macros: int,
        size_l0_input: int,
        size_l0_output: int,
        size_l1_input: int,
        storage_bytes: float
    ):
        """
        初始化Tiling引擎

        Args:
            macro_spec: CIM Macro规格
            n_macros: Macro数量
            size_l0_input: L0输入Buffer大小（字节）
            size_l0_output: L0输出Buffer大小（字节）
            size_l1_input: L1输入Buffer大小（字节）
            storage_bytes: 每元素存储字节数
        """
        self.macro_spec = macro_spec
        self.n_macros = n_macros
        self.size_l0_input = size_l0_input
        self.size_l0_output = size_l0_output
        self.size_l1_input = size_l1_input
        self.storage_bytes = storage_bytes

    def parse_partition_strategy(
        self,
        partition_strategy: Union[str, Tuple[int, int]]
    ) -> Tuple[int, int]:
        """
        解析并行划分策略

        Args:
            partition_strategy: 策略名称（'N_Parallel'/'K_Parallel'）或(pk, pn)元组

        Returns:
            (pk, pn) 元组，表示K和N维度的并行度

        Raises:
            ValueError: 未知的策略名称
        """
        if isinstance(partition_strategy, str):
            if partition_strategy == 'N_Parallel':
                return 1, self.n_macros
            elif partition_strategy == 'K_Parallel':
                return self.n_macros, 1
            else:
                raise ValueError(f"Unknown strategy: {partition_strategy}")
        return partition_strategy

    def derive_tile_sizes(
        self,
        M: int,
        K: int,
        N: int,
        partition_strategy: Union[str, Tuple[int, int]]
    ) -> Tuple[Dict[str, int], Dict[str, int], Tuple[int, int]]:
        """
        计算两级Tile大小

        Args:
            M: 矩阵M维度
            K: 矩阵K维度
            N: 矩阵N维度
            partition_strategy: 并行划分策略

        Returns:
            (l0_dims, l1_dims, (pk, pn)) 元组:
            - l0_dims: L0 Tile维度字典 {'m', 'k', 'n'}
            - l1_dims: L1 Tile维度字典 {'m', 'k', 'n'}
            - (pk, pn): 实际使用的并行度
        """
        # --- A. 解析并行策略 ---
        pk, pn = self.parse_partition_strategy(partition_strategy)

        # --- B. 基础物理维度 (Hardware Atomic Dimensions) ---
        atomic_k = pk * self.macro_spec.block_k_dim
        atomic_n = pn * self.macro_spec.block_n_dim
        bytes_per_item = self.storage_bytes

        # --- C. 计算 L0 Tile (微观切分) ---
        k_per_macro = self.macro_spec.block_k_dim

        # L0输入Buffer限制
        m_limit_l0_input = int(self.size_l0_input // (k_per_macro * bytes_per_item)) \
            if k_per_macro > 0 else M

        # L0输出Buffer限制
        n_per_macro = self.macro_spec.block_n_dim
        m_limit_l0_output = int(self.size_l0_output // (n_per_macro * bytes_per_item)) \
            if n_per_macro > 0 else M

        # 取两者最小值
        m_tile_l0 = min(m_limit_l0_input, m_limit_l0_output)
        m_tile_l0 = max(1, int(m_tile_l0))

        # L0 Tile 的逻辑尺寸 (系统视角)
        l0_dims = {
            'm': int(m_tile_l0),
            'k': int(atomic_k),
            'n': int(atomic_n)
        }

        # --- D. 计算 L1 Tile (宏观切分) ---
        k_tile_l1 = atomic_k
        n_tile_l1 = atomic_n

        # M_L1 * K_L1 * Bytes <= L1_Size
        m_limit_l1 = self.size_l1_input // (k_tile_l1 * bytes_per_item)

        # 对齐: L1 Tile 最好是 L0 Tile 的整数倍
        m_tile_l1 = int((m_limit_l1 // m_tile_l0) * m_tile_l0)

        # 边界: 至少要能装下一个 L0 Tile
        if m_tile_l1 < m_tile_l0:
            m_tile_l1 = int(m_tile_l0)

        l1_dims = {
            'm': int(m_tile_l1),
            'k': int(k_tile_l1),
            'n': int(n_tile_l1)
        }

        return l0_dims, l1_dims, (pk, pn)
