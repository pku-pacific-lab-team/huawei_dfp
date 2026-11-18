"""Cold CMEM input panels with capacity-aware buffering for one TensorCore.

Each outer M/N tile holds full-K A and B panels. Inner output blocks finish all
K blocks in VMEM before exporting C. This deliberately avoids partial-sum spills
and does not retain data across outer tiles. Timing schedules outer prefetches.
"""
from fractions import Fraction
from math import ceil


def ceildiv(value, divisor):
    return (value + divisor - 1) // divisor


def make_cmem_plan(spec, workload, inner, mode, groups, buffers=2):
    if not isinstance(mode, str) or mode not in ("bypass", "tiled"):
        raise ValueError("cmem_mode must be bypass or tiled")
    if (not isinstance(groups, (list, tuple)) or len(groups) != 2
            or any(type(n) is not int or n <= 0 for n in groups)):
        raise ValueError("cmem_groups_mn must contain two positive integer block counts")
    if type(buffers) is not int or buffers not in (1, 2):
        raise ValueError("cmem_buffers must be 1 or 2")
    v = spec.values
    capacity = v["cmem_usable_bytes"]
    share = None if capacity is None else capacity // v["tensorcores_per_chip"]
    result = {
        "scope": "single_tensorcore", "requested_mode": mode, "mode": mode,
        "reason": "explicit_bypass" if mode == "bypass" else "full_k_input_panels_fit",
        "chip_usable_bytes": capacity, "usable_bytes": share,
        "allocated_bytes": 0, "buffer_slots": 0, "requested_buffer_slots": buffers,
        "single_buffer_bytes": 0,
        "requested_groups_mn": list(groups), "selected_groups_mn": [0, 0],
        "outer_tile_mkn": [0, workload.K, 0], "outer_tiles": 0,
        "initial_state": "cold", "placement": "full_k_A_B_panels",
        "capacity_policy": "equal_partition_floor_no_borrow",
        "bandwidth_policy": "equal_partition_no_borrow_independent_read_write",
        "output_policy": "VMEM_to_HBM_bypass",
        "outer_schedule": "prefetch_with_shared_HBM; release_after_last_input_read; serial_inner_output_blocks",
        "buffer_policy": "preserve_panel_shape; use_second_slot_only_if_capacity_and_next_tile_allow",
        "fallback_policy": "shrink_groups_then_bypass_if_one_panel_pair_cannot_fit",
    }
    if mode == "bypass":
        return result
    if share is None:
        raise ValueError("CMEM counting parameters missing: cmem_usable_bytes")
    M, K, N = workload.shape
    selected = [min(groups[0], ceildiv(M, inner.m)), min(groups[1], ceildiv(N, inner.n))]
    while True:
        m, n = min(M, selected[0] * inner.m), min(N, selected[1] * inner.n)
        size = K * (m + n)  # INT8 inputs only; C remains in VMEM.
        if size <= share:
            count = ceildiv(M, m) * ceildiv(N, n)
            slots = min(buffers, share // size, count)
            result.update(allocated_bytes=size * slots, single_buffer_bytes=size, buffer_slots=slots,
                          selected_groups_mn=selected, outer_tile_mkn=[m, K, n], outer_tiles=count)
            if selected != [min(groups[0], ceildiv(M, inner.m)), min(groups[1], ceildiv(N, inner.n))]:
                result["reason"] = "groups_reduced_to_fit_capacity"
            return result
        candidates = [i for i in range(2) if selected[i] > 1]
        if not candidates:
            result.update(mode="bypass", reason="insufficient_capacity_for_one_full_k_panel_pair")
            return result
        axis = max(candidates, key=lambda i: (m, n)[i])
        selected[axis] = (selected[axis] + 1) // 2


def input_traffic(workload, inner, cmem):
    M, K, N = workload.shape
    a = ceildiv(N, inner.n) * M * K
    b = ceildiv(M, inner.m) * K * N
    if cmem["mode"] == "tiled":
        m, _, n = cmem["outer_tile_mkn"]
        hbm_a = ceildiv(N, n) * M * K
        hbm_b = ceildiv(M, m) * K * N
        fill, read = hbm_a + hbm_b, a + b
    else:
        hbm_a, hbm_b, fill, read = a, b, 0, 0
    return {
        "input_a_transfer_bytes": a, "input_b_transfer_bytes": b,
        "input_transfer_bytes": a + b,  # Source-to-VMEM payload, not HBM traffic.
        "hbm_input_a_read_bytes": hbm_a, "hbm_input_b_read_bytes": hbm_b,
        "hbm_input_read_bytes": hbm_a + hbm_b,
        "cmem_fill_bytes": fill, "cmem_to_vmem_bytes": read,
        "hbm_read_bytes_saved_vs_bypass": a + b - hbm_a - hbm_b,
    }


def services(spec):
    required = ("cmem_read_bandwidth_bytes_per_second", "cmem_write_bandwidth_bytes_per_second")
    missing = [name for name in required if spec.values[name] is None]
    if missing:
        raise ValueError("CMEM timing parameters missing: " + ", ".join(missing))
    v = spec.values
    clock = Fraction(str(v["clock_hz"]))
    shares = {direction: Fraction(str(v[f"cmem_{direction}_bandwidth_bytes_per_second"]))
              / v["tensorcores_per_chip"] / clock for direction in ("read", "write")}
    hbm = Fraction(str(v["hbm_bandwidth_bytes_per_second"])) / v["tensorcores_per_chip"] / clock
    shares["fill"] = min(hbm, shares["write"])
    shares["to_vmem"] = min(shares["read"], Fraction(str(v["vmem_bandwidth_bytes_per_second"])) / clock)
    return shares


def fill_cycles(workload, cmem, bandwidth, startup):
    """At most four outer shapes; no per-panel or per-element event expansion."""
    def sizes(length, tile):
        full, tail = divmod(length, tile)
        return ([(tile, full)] if full else []) + ([(tail, 1)] if tail else [])

    m, _, n = cmem["outer_tile_mkn"]
    return sum((ceil(workload.K * (a + b) / bandwidth) + startup) * ar * br
               for a, ar in sizes(workload.M, m) for b, br in sizes(workload.N, n))
