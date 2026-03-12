"""
TSP-CIM Tiling Engine (V2)

单级 Tiling 策略，含 MEM 切片分配建模。
所有数据在片上 SRAM（220MB）中，无 DRAM 层级。
根据 CIM Macro 计算能力 + MEM 切片物理位置进行切分。
"""
import math
from typing import Dict

from ..hardware.tsp_cim import TSPCIMSpec


class TSPCIMTilingEngine:
    """
    TSP-CIM 单级 Tiling 引擎（含 MEM 分配）

    切分 GEMM (M×K × K×N) 为 CIM Macro 可处理的 tile，
    同时计算每种数据需要的 MEM 切片数量和位置。
    """

    def __init__(self, tsp_spec: TSPCIMSpec, storage_bytes: float = 1.0):
        self.tsp_spec = tsp_spec
        self.storage_bytes = storage_bytes

    def derive_tile_sizes(
        self,
        M: int,
        K: int,
        N: int,
    ) -> Dict:
        """
        计算 tile 大小和 MEM 分配

        Returns:
            {
                'm': tile_m,
                'k': tile_k,
                'n': tile_n,
                'mem_alloc': {
                    'act_slices': int,       # activation 使用的 MEM 切片数
                    'act_start': int,        # activation 起始切片
                    'wt_slices': int,        # weight 使用的 MEM 切片数
                    'wt_start': int,         # weight 起始切片
                    'result_slices': int,    # result 使用的 MEM 切片数
                    'result_start': int,     # result 起始切片
                    'streams_for_act': int,  # 分配给 activation 的 stream 数
                    'streams_for_wt': int,   # 分配给 weight 的 stream 数
                },
            }
        """
        spec = self.tsp_spec

        # K tile: 单 macro 的 block_k
        tile_k = min(spec.chip_block_k, K)  # 256

        # N tile: 所有 superlane 并行, 每 superlane chip_block_n
        tile_n = min(spec.n_superlanes * spec.chip_block_n, N)

        # M tile: 受 stream 并发度约束
        # 每 stream 320 bytes, activation row = K * storage_bytes
        # 一行 activation = tile_k * storage_bytes bytes
        # 可用 stream 数 = n_streams_per_direction (32)
        # 每批可读 32 个 stream × 320 bytes = 10,240 bytes
        # M tile 设为 total_lanes (320) 或更小
        tile_m = min(M, spec.total_lanes)

        # ---- MEM 切片分配 ----
        slice_cap = spec.mem_slice_capacity_bytes

        # Activation: tile_m × tile_k × storage_bytes
        act_tile_bytes = tile_m * tile_k * self.storage_bytes
        act_slices = max(1, math.ceil(act_tile_bytes / slice_cap))

        # Weight: tile_k × tile_n × storage_bytes
        wt_tile_bytes = tile_k * tile_n * self.storage_bytes
        wt_slices = max(1, math.ceil(wt_tile_bytes / slice_cap))

        # Result: tile_m × tile_n × 4 (PSUM int32)
        result_tile_bytes = tile_m * tile_n * 4
        result_slices = max(1, math.ceil(result_tile_bytes / slice_cap))

        # Stream 分配策略:
        # outward streams (→CIM): activation + weight
        # inward streams (←VXM): results
        # 32 streams per direction, 分配给 act 和 wt
        total_outward_streams = spec.n_streams_per_direction
        # weight 需要的 stream 数 (同时送到 CIM 编程)
        wt_vectors_per_batch = math.ceil(wt_tile_bytes / spec.stream_width_bytes)
        act_vectors_per_batch = math.ceil(act_tile_bytes / spec.stream_width_bytes)

        # 简单分配：weight 和 activation 时分复用同批 stream
        # MXM 原始设计用所有 32 stream 送 weight (IW), 也用 stream 送 activation
        # CIM: weight 编程和 activation 读取不同时进行（流水线不同阶段）
        streams_for_act = total_outward_streams
        streams_for_wt = total_outward_streams

        return {
            'm': tile_m,
            'k': tile_k,
            'n': tile_n,
            'mem_alloc': {
                'act_slices': act_slices,
                'act_start': spec.act_mem_start_slice,
                'wt_slices': wt_slices,
                'wt_start': spec.wt_mem_start_slice,
                'result_slices': result_slices,
                'result_start': spec.act_mem_start_slice,  # results 也放近 VXM
                'streams_for_act': streams_for_act,
                'streams_for_wt': streams_for_wt,
            },
        }
