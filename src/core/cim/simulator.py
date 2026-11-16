"""CIM-TSP aggregation of INT8 Macro actions.

Only Macro actions and required logical K reductions are aggregated here.
An optional tile observer feeds upper-level timing; energy/area stay separate.
"""
from collections import Counter
import itertools

from .macro_model import CIMMacroModel, MacroContexts, positive_integer

# CIM-TSP 通过 count_macro_tiles 聚合 Macro 动作并管理各物理槽位的 context。
# 上层提供物理 K/N 容量与执行范围，本文件不定义四架构通用调度器。

LOOP_ORDERS = tuple("".join(p) for p in itertools.permutations("MKN"))


def count_macro_tiles(spec, M, K, N, tile_m, tile_k, tile_n, n_macros, loop_order, scope, *, on_tile=None):
    for name, value in (("M", M), ("K", K), ("N", N), ("tile_m", tile_m),
                        ("tile_k", tile_k), ("tile_n", tile_n), ("n_macros", n_macros)):
        positive_integer(value, name)
    if loop_order not in LOOP_ORDERS:
        raise ValueError("loop_order must be a permutation of MKN")
    bk, bn = spec.block_k_dim, spec.block_n_dim
    if tile_k % bk or tile_n % bn:
        raise ValueError("physical K/N tile capacities must align to Macro geometry")
    pk, pn = tile_k // bk, tile_n // bn
    # pk/pn 是 K/N 方向的物理 Macro 槽位数，不是当前尾块的有效分片数。
    if pk * pn > n_macros:
        raise ValueError("tile exceeds physical Macro inventory")
    ranges = {"M": range(0, M, tile_m), "K": range(0, K, tile_k), "N": range(0, N, tile_n)}
    model, contexts, cache = CIMMacroModel(spec), {}, {}
    # contexts 随一次 simulate 的遍历保留权重状态；cache 只复用相同形状的计数结果。
    # 二者每次调用都从空开始，循环顺序实验之间不会继承上一轮权重。
    counts = Counter()
    for offsets in itertools.product(*(ranges[axis] for axis in loop_order)):
        pos = dict(zip(loop_order, offsets))
        ms, ks, ns = (pos[axis] for axis in "MKN")
        m, k, n = min(M-ms, tile_m), min(K-ks, tile_k), min(N-ns, tile_n)
        programs_before, switches_before = counts["context_programs"], counts["context_switches"]
        counts["tiles"] += 1
        active_k = (k + bk - 1) // bk
        # 同一输出列的 K 分片需要相加：块内 active_k 份合并需 active_k-1 次。
        # 后续 K tile 再与先前部分和相加一次；N 分片负责不同列，无跨列归约。
        # Required logical adds are outside the Macro; no timing or routing claim.
        counts["cross_macro_k_adds"] += m * n * (active_k - 1)
        counts["cross_tile_k_adds"] += m * n if ks else 0
        for ik in range(active_k):
            for jn in range((n + bn - 1) // bn):
                ka, na = min(bk, k - ik*bk), min(bn, n - jn*bn)
                # 固定使用物理 pn 作为行跨度，尾块不会让同一位置换一个槽位编号。
                slot = ik * pn + jn
                if slot not in contexts:
                    contexts[slot] = MacroContexts(spec)
                # B 只由 K/N 位置及有效形状决定；tag 不含 M，才能跨输入行复用权重。
                tag = (ks + ik*bk, ns + jn*bn, ka, na)
                loaded, switched = contexts[slot].ensure(tag)
                if loaded:
                    counts.update(model.program_context()["counts"])
                    counts["context_programs"] += 1
                if switched:
                    counts.update(model.switch_context()["counts"])
                key = (m, ka, na)
                # 缓存省去重复求公式，但每次调用仍将动作加入 counts，不会少计工作。
                if key not in cache:
                    cache[key] = model.compute(m, ka, na)["counts"]
                counts.update(cache[key])
        # 时间模型观察实际的 context 命中/替换结果，不另写一套权重复用逻辑。
        # 不传观察器时保持原来的纯计数行为，也不保存逐 tile 轨迹占用内存。
        if on_tile is not None:
            on_tile(m=m, k=k, n=n, k_start=ks,
                    programs=counts["context_programs"] - programs_before,
                    switches=counts["context_switches"] - switches_before)
    counts.update(useful_ops=2*M*K*N, executed_padded_ops=2*counts["issued_mac"],
                  logical_output_bytes=4*M*N)
    units = {"useful_mac": "MAC", "issued_mac": "MAC", "launches": "launch",
             "context_programs": "context", "context_switches": "switch",
             "program_wordlines": "wordline", "bit_serial_cycles": "macro_cycle",
             "weight_word_bit_products": "word_bit_product", "local_tree_adds": "add",
             "shift_accumulates": "shift_accumulate"}
    actions = {key: {"count": value, "unit": "byte" if key.endswith("_bytes") else units[key],
                     "scope": "sum_of_macro_actions"}
               for key, value in counts.items() if key in units or key in {
                   "program_bytes", "activation_capture_bytes", "result_bytes"}}
    # 这些周期与 byte 是各 Macro 的局部动作之和，不能直接当作系统时延或 MEM 流量。
    return {
        "schema_version": 2, "model": "cim_macro_count_integration", "mode": "count",
        "reference_id": spec.reference_id,
        "workload": {"operation": "gemm", "M": M, "K": K, "N": N, "data_format": "INT8",
                     "accumulation_dtype": "INT32", "output_dtype": "INT32"},
        "boundary": {"scope": scope, "start": "weight_tags_unloaded",
                     "end": "macro_local_INT32_partials", "numeric_values_simulated": False,
                     "transport_included": False, "cross_macro_reduction_timed": False},
        "policy": {"loop_order": loop_order, "selected_tile_mkn": [tile_m, tile_k, tile_n],
                   "context_replacement": "round_robin_inactive",
                   "context_initial_state": "empty_per_simulate",
                   "task_boundaries": "drained_serialized",
                   "tail_policy": "zero_pad_full_macro; no free gating savings"},
        "counts": dict(counts), "action_counts": actions,
        "timing": {"status": "unavailable_system_model", "total_cycles": None,
                   "chip_cycles": None, "time_ns": None, "throughput_gops": None},
        "energy": {"status": "not_evaluated", "total_pj": None, "tops_per_watt": None},
        "area": {"status": "not_evaluated", "total_mm2": None, "tops_per_mm2": None},
        "hardware_inventory": {"allocated_macros": n_macros, "used_macro_slots": len(contexts),
                               "resident_macro_weight_bytes": n_macros * spec.resident_weight_bytes},
        "hardware_derived": spec.derived(),
        "parameter_provenance": spec.to_dict(),
        "limitations": [
            "Generic Macro aggregation; the caller supplies the upper topology and execution scope.",
            "Summed Macro cycles/bytes are local actions, not elapsed system time or MEM traffic.",
            "Cross-Macro and cross-tile K additions are logical obligations, not executed fabric actions.",
            "No transport overlap, rolling IW schedule, context busy state, power or physical area calibration.",
        ],
    }
