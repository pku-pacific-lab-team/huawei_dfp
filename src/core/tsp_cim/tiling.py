"""Single Cluster task-domain tiling, without MEM allocation or chip scheduling."""
# K/N 来自固定物理域；tile_m 仅用于观察行分组如何影响权重复用。
from ..cim.macro_model import positive_integer


class TSPCIMTilingEngine:
    def __init__(self, tsp_spec, storage_bytes=1):
        if type(storage_bytes) is not int or storage_bytes != 1:
            raise ValueError("TSP-CIM count tiling supports INT8 only")
        self.tsp_spec = tsp_spec

    def derive_tile_sizes(self, M, K, N, tile_m=None):
        for name, value in (("M", M), ("K", K), ("N", N)):
            positive_integer(value, name)
        if tile_m is not None:
            positive_integer(tile_m, "tile_m")
        spec = self.tsp_spec
        # Stable physical K/N slot strides even for tails. M is a count grouping,
        # not a claim that these activations fit in a hardware buffer.
        # 尾块仍使用完整 domain 的槽位跨度，有效 K/N 由下层逐 Macro 截取。
        return {"m": M if tile_m is None else min(M, tile_m),
                "k": spec.plane.domain_k, "n": spec.cluster.n,
                "status": "single_cluster_domain; logical row grouping; no MEM capacity claim"}
