"""Bounded single-buffer tiling; the output stays in L0C across all K panels."""
# m/k/n 是 L0 计算块；l1_k 是 L1 搬入的 K 面板长度，两者不可混用。
from dataclasses import asdict, dataclass


def align_up(value, alignment):
    return ((value + alignment - 1) // alignment) * alignment


@dataclass(frozen=True)
class TilePlan:
    m: int
    k: int
    n: int
    l1_k: int
    ub_chunk_bytes: int

    def to_dict(self):
        return asdict(self)


def make_plan(workload, capacities, tile=(64, 128, 64), l1_k=256):
    if len(tile) != 3 or any(type(v) is not int or v <= 0 or v % a for v, a in zip(tile, (16, 32, 16))):
        raise ValueError("tile must be positive aligned M K N (multiples of 16 32 16)")
    if type(l1_k) is not int or l1_k <= 0 or l1_k % 32:
        raise ValueError("l1_k must be a positive multiple of 32")
    if any(capacities[x] < minimum for x, minimum in {"L1": 1024, "L0A": 512, "L0B": 512, "L0C": 1024, "UB": 1024}.items()):
        raise ValueError("Capacity cannot hold the minimum INT8/INT32 fractals")
    m, k, n = (min(v, align_up(dim, a)) for v, dim, a in zip(tile, workload.shape, (16, 32, 16)))
    k1 = min(l1_k, align_up(workload.K, 32))
    k = min(k, k1)

    def shrink(v, step):
        # 缩块后仍保持 INT8 分形对齐，并且不低于最小分形尺寸。
        return max(step, (v // 2 // step) * step)

    while True:
        # 依次满足输出、两路输入、L1 面板容量，返回可行块，不做性能搜索。
        if 4 * m * n > capacities["L0C"]:
            if m >= n:
                m = shrink(m, 16)
            else:
                n = shrink(n, 16)
        elif m * k > capacities["L0A"]:
            if k > 32:
                k = shrink(k, 32)
            else:
                m = shrink(m, 16)
        elif k * n > capacities["L0B"]:
            if k > 32:
                k = shrink(k, 32)
            else:
                n = shrink(n, 16)
        elif (m + n) * k1 > capacities["L1"]:
            if k1 > 32:
                k1 = shrink(k1, 32)
                k = min(k, k1)
            elif m >= n:
                m = shrink(m, 16)
            else:
                n = shrink(n, 16)
        else:
            break
    # UB 可以分段承接最终 C，所以只需给出不超容量、按 1024 byte 对齐的导出块。
    return TilePlan(m, k, n, k1, min(4 * m * n, capacities["UB"] // 1024 * 1024))
