"""
CIM Macro Specification

定义单个CIM Macro的物理规格参数
"""


class CIMMacroSpec:
    """
    CIM Macro 硬件规格类

    定义单个CIM Macro的物理结构参数，包括：
    - Bank数量和结构
    - Sub-array配置
    - Column配置
    - 输入位宽

    Attributes:
        n_banks_internal: 内部Bank数量 (对应N维度)
        n_subarrays_per_bank: 每个Bank的Sub-array数量
        n_loc_per_subarray: 每个Sub-array的本地读出开关数量
        n_cols_per_bank: 每个Bank的Column数量
        p_input: 物理单元支持的输入位宽
        block_k_dim: K维度的块大小 (= n_subarrays_per_bank * n_loc_per_subarray)
        block_n_dim: N维度的块大小 (= n_banks_internal)
    """

    def __init__(
        self,
        n_banks_internal: int = 64,
        n_subarrays_per_bank: int = 64,
        n_loc_per_subarray: int = 4,
        n_cols_per_bank: int = 8,
        p_input: int = 8
    ):
        """
        初始化CIM Macro规格

        Args:
            n_banks_internal: 内部Bank数量，默认64
            n_subarrays_per_bank: 每Bank的Sub-array数量，默认64
            n_loc_per_subarray: 每Sub-array的本地读出开关数，默认4
            n_cols_per_bank: 每Bank的Column数量，默认8
            p_input: 输入位宽，默认8
        """
        self.n_banks_internal = n_banks_internal
        self.n_subarrays_per_bank = n_subarrays_per_bank
        self.n_loc_per_subarray = n_loc_per_subarray
        self.n_cols_per_bank = n_cols_per_bank
        self.p_input = p_input

        # 计算派生参数
        self.block_k_dim = n_subarrays_per_bank * n_loc_per_subarray
        self.block_n_dim = n_banks_internal

    @classmethod
    def from_dict(cls, config: dict) -> "CIMMacroSpec":
        """
        从字典创建CIMMacroSpec实例

        Args:
            config: 包含macro配置的字典

        Returns:
            CIMMacroSpec实例
        """
        return cls(
            n_banks_internal=config.get("n_banks_internal", 64),
            n_subarrays_per_bank=config.get("n_subarrays_per_bank", 64),
            n_loc_per_subarray=config.get("n_loc_per_subarray", 4),
            n_cols_per_bank=config.get("n_cols_per_bank", 8),
            p_input=config.get("p_input", 8)
        )

    def to_dict(self) -> dict:
        """
        将规格导出为字典

        Returns:
            包含所有规格参数的字典
        """
        return {
            "n_banks_internal": self.n_banks_internal,
            "n_subarrays_per_bank": self.n_subarrays_per_bank,
            "n_loc_per_subarray": self.n_loc_per_subarray,
            "n_cols_per_bank": self.n_cols_per_bank,
            "p_input": self.p_input,
            "block_k_dim": self.block_k_dim,
            "block_n_dim": self.block_n_dim,
        }

    def __repr__(self) -> str:
        return (
            f"CIMMacroSpec(banks={self.n_banks_internal}, "
            f"subarrays={self.n_subarrays_per_bank}, "
            f"locs={self.n_loc_per_subarray}, "
            f"cols={self.n_cols_per_bank}, "
            f"block_k={self.block_k_dim}, block_n={self.block_n_dim})"
        )
