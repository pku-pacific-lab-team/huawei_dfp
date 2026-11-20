"""Single-side resident layout and coarse tiling, without bank/stream scheduling."""
# 当前模型要求整个任务的补齐 A/B/C 放在本侧所选 MEM slices 中，不能只检查一个 tile。
import math


def align_up(value, alignment):
    return (value + alignment - 1) // alignment * alignment


class TSPTilingEngine:
    def __init__(self, tsp_spec):
        self.spec = tsp_spec

    def derive_tile_sizes(self, M, K, N, data_format="INT8", *, tile=(320, 320, 640),
                          mem_slices=None, streams=None, reserved_bytes=0):
        spec = self.spec
        if data_format != "INT8":
            raise ValueError("Reference execution supports INT8 only")
        if any(type(d) is not int or d <= 0 for d in (M, K, N)):
            raise ValueError("M, K, N must be positive integers")
        if len(tile) != 3 or any(type(d) is not int or d <= 0 for d in tile):
            raise ValueError("tile must contain three positive integers (M,K,N)")
        slices = tuple(range(spec.n_mem_slices_per_hemisphere)) if mem_slices is None else tuple(mem_slices)
        if not slices or any(type(s) is not int or not 0 <= s < spec.n_mem_slices_per_hemisphere for s in slices) or len(set(slices)) != len(slices):
            raise ValueError("MEM slices must be distinct valid single-hemisphere indices")
        streams = spec.n_streams_per_direction if streams is None else streams
        if type(streams) is not int or not 1 <= streams <= spec.n_streams_per_direction:
            raise ValueError("stream allocation exceeds directional inventory")
        if type(reserved_bytes) is not int or reserved_bytes < 0:
            raise ValueError("reserved_bytes must be a nonnegative explicit policy value")
        block = spec.supercell_dim
        if tile[1] < block or tile[2] < block:
            raise ValueError("K/N tile bounds must fit at least one supercell")
        k = min(spec.mxm_plane_dim, tile[1] // block * block)
        # K 受单 plane 高度限制，N 可由本侧多个 plane 拼接；M 只作为行分组，不补齐。
        n = min(spec.tile_n_capacity_per_hemisphere(data_format), tile[2] // block * block)
        m = min(M, tile[0])
        kp, np = align_up(K, block), align_up(N, block)
        # C_PSUM 同时表示中间部分和与最终输出，不再为它们分别申请完整矩阵空间。
        sizes = {"A": M * kp, "B": kp * np, "C_PSUM": 4 * M * np, "reserved": reserved_bytes}
        ranges, offset = {}, 0
        for name, size in sizes.items():
            # 每个逻辑区均匀条带化到所选 slices，逐区向上取整后再安排不重叠的偏移。
            per_slice = math.ceil(size / len(slices))
            ranges[name] = {"bytes": size, "per_slice_bytes": per_slice,
                            "start_per_slice": offset, "end_per_slice": offset + per_slice}
            offset += per_slice
        if offset > spec.mem_slice_capacity_bytes:
            # 超出本侧容量就拒绝任务；没有隐含的片外搬入、远端半片借用或换页机制。
            raise ValueError(f"unsupported_capacity: resident A/B/C require {offset * len(slices)} bytes "
                             f"on {len(slices)} local slices; capacity={spec.mem_slice_capacity_bytes * len(slices)}; "
                             "off-chip/remote-side supply is not modeled")
        return {"m": m, "k": k, "n": n, "padded_K": kp, "padded_N": np,
                "mem_slices": list(slices), "streams_per_direction": streams,
                "memory": {"scope": "single_hem", "capacity_bytes": len(slices) * spec.mem_slice_capacity_bytes,
                           "resident_bytes": offset * len(slices), "ranges": ranges,
                           "layout": "disjoint uniformly striped logical packed ranges; not an ISA memory ABI",
                           "tile_capacity_includes_banks": True},
                "tail_policy": "pad K/N to supercell; keep M exact; no extra full C PSUM allocation"}
