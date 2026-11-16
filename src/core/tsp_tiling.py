"""
TSP Tiling Engine
"""
import math
from typing import Dict

from ..hardware.tsp import TSPSpec


class TSPTilingEngine:
    """Derive paper-aligned TSP tiles and MEM placement."""

    def __init__(self, tsp_spec: TSPSpec, storage_bytes: float = 1.0):
        self.tsp_spec = tsp_spec
        self.storage_bytes = storage_bytes

    def derive_tile_sizes(
        self,
        M: int,
        K: int,
        N: int,
        data_format: str,
    ) -> Dict[str, object]:
        spec = self.tsp_spec

        tile_m = min(M, spec.total_lanes)
        tile_k = min(K, spec.mxm_plane_dim)
        tile_n = min(N, spec.tile_n_capacity_per_hemisphere(data_format))

        slice_cap = spec.mem_slice_capacity_bytes
        act_tile_bytes = tile_m * tile_k * self.storage_bytes
        wt_tile_bytes = tile_k * tile_n * self.storage_bytes
        result_tile_bytes = tile_m * tile_n * 4

        act_slices = max(1, math.ceil(act_tile_bytes / slice_cap))
        wt_slices = max(1, math.ceil(wt_tile_bytes / slice_cap))
        result_slices = max(1, math.ceil(result_tile_bytes / slice_cap))

        return {
            "m": tile_m,
            "k": tile_k,
            "n": tile_n,
            "topology": spec.summarize_topology(data_format, tile_k, tile_n),
            "mem_alloc": {
                "act_slices": act_slices,
                "act_start": spec.act_mem_start_slice,
                "wt_slices": wt_slices,
                "wt_start": spec.wt_mem_start_slice,
                "result_slices": result_slices,
                "result_start": spec.act_mem_start_slice,
                "streams_for_act": spec.n_streams_per_direction,
                "streams_for_wt": spec.n_streams_per_direction,
                "streams_for_result": spec.n_streams_per_direction,
            },
        }
