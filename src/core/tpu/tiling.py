"""Simple logical blocks, without hardware padding or MXU placement."""
# 分开检查整个任务的 HBM 占用，以及当前 A/B/C 小块的 VMEM 同时驻留量。
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TilePlan:
    m: int
    k: int
    n: int
    peak_working_bytes: int
    hbm_footprint_bytes: int

    def to_dict(self):
        return asdict(self)


def make_plan(workload, vmem_bytes, hbm_bytes, tile=(128, 128, 128)):
    if len(tile) != 3 or any(type(x) is not int or x <= 0 for x in tile):
        raise ValueError("tile must contain three positive integers in M K N order")
    if any(type(x) is not int or x <= 0 for x in (vmem_bytes, hbm_bytes)):
        raise ValueError("usable VMEM/HBM capacities must be positive integers")
    M, K, N = workload.shape
    # A、B 为 INT8（1 byte），C 为 INT32（4 byte）；不额外分配一份完整 PSUM。
    footprint = M * K + K * N + 4 * M * N
    if footprint > hbm_bytes:
        raise ValueError("A/B/C logical footprint exceeds usable HBM capacity")
    if vmem_bytes < 6:
        # 最小 1×1 乘法也要同时放下 A、B、C，共 1+1+4 byte。
        raise ValueError("VMEM cannot hold the minimum INT8/INT32 logical block (6 bytes)")
    dims = [min(limit, size) for limit, size in zip(tile, workload.shape)]

    def size():
        m, k, n = dims
        return m * k + k * n + 4 * m * n

    while size() > vmem_bytes:
        # 每次缩短当前最长的一维，直到能放下；这个回退规则不搜索性能最优块。
        # Deterministic capacity fallback, not an optimization or hardware tile shape.
        index = max(range(3), key=lambda i: dims[i])
        dims[index] = max(1, dims[index] // 2)
    return TilePlan(*dims, size(), footprint)
