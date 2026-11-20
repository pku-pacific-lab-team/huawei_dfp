"""Aggregate Macro actions and context changes for a hemisphere task tile."""
import itertools

from .macro_model import CIMMacroModel, MacroContexts

LOOP_ORDERS = tuple("".join(p) for p in itertools.permutations("MKN"))


def account_macro_tile(spec, contexts, cache, counts, *, ks, ns, m, k, n, pn):
    """Account one globally tagged tile in a persistent physical domain."""
    bk, bn = spec.block_k_dim, spec.block_n_dim
    model = CIMMacroModel(spec)
    programmed, switched_slots = [], []
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
            old_active = contexts[slot].active
            loaded, switched = contexts[slot].ensure(tag)
            if loaded:
                programmed.append((slot, contexts[slot].active))
                counts.update(model.program_context()["counts"])
                counts["context_programs"] += 1
            if switched:
                switched_slots.append((slot, old_active))
                counts.update(model.switch_context()["counts"])
            key = (m, ka, na)
            # 缓存省去重复求公式，但每次调用仍将动作加入 counts，不会少计工作。
            if key not in cache:
                cache[key] = model.compute(m, ka, na)["counts"]
            counts.update(cache[key])
    return programmed, switched_slots
