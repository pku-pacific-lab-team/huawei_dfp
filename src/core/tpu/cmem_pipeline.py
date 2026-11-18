"""Whole-panel CMEM prefetch around the existing serial-output VMEM pipeline.

HBM fills and exports share one non-preemptive service. CMEM reads and writes
may overlap on different live slots. A slot is reusable after its final read,
not after the associated output export. This is a scheduling proxy, not XLA.
"""
from dataclasses import replace
from functools import lru_cache
from math import ceil

from .cmem import ceildiv
from .pipeline import pipeline_cycles


def cmem_pipeline_cycles(spec, workload, plan, input_slots, cmem, services, *,
                         events=None, accelerate=True):
    """Return makespan; optional small-case trace supports dependency checks.

    K and inner-output repetitions are aggregated. Identical column/row states
    are also advanced in bulk, retaining the last three columns/rows for tails
    and prefetch lookahead. Trace mode deliberately evaluates each output.
    """
    outer_m, _, outer_n = cmem["outer_tile_mkn"]
    rows, columns = ceildiv(workload.M, outer_m), ceildiv(workload.N, outer_n)
    count, slots = rows * columns, cmem["buffer_slots"]
    startup = spec.execution["fixed_latency_cycles"]["transfer_startup"]

    def shape(index):
        row, col = divmod(index, columns)
        return min(outer_m, workload.M-row*outer_m), min(outer_n, workload.N-col*outer_n)

    @lru_cache(None)
    def profile(m, n):
        total, _, detail = pipeline_cycles(spec, replace(workload, M=m, N=n), plan,
            input_slots, input_bpc=services["read"], block_details=True)
        return total, detail[0]

    def sizes(length, block):
        full, tail = divmod(length, block)
        return ([(block, full)] if full else []) + ([(tail, 1)] if tail else [])

    @lru_cache(None)
    def panel(m, n):
        first = profile(min(m, plan.m), min(n, plan.n))
        last = profile((m-1) % plan.m+1, (n-1) % plan.n+1)
        total = sum(profile(a, b)[0]*ar*br for a, ar in sizes(m, plan.m)
                    for b, br in sizes(n, plan.n))
        return ceildiv(m, plan.m)*ceildiv(n, plan.n), total, first, last

    t = hbm_free = 0
    ready, tags = [0]*slots, [-1]*slots

    def record(kind, start, end, outer, **extra):
        if events is not None:
            events.append(dict(kind=kind, start=start, end=end, outer=outer, **extra))

    def fill_duration(index):
        m, n = shape(index)
        return ceil(workload.K*(m+n)/services["fill"])+startup

    def fill(index, earliest):
        nonlocal hbm_free
        slot = index % slots
        start = max(hbm_free, earliest)
        hbm_free = start+fill_duration(index)
        ready[slot], tags[slot] = hbm_free, index
        record("cmem_fill", start, hbm_free, index, slot=slot)

    def output(detail, index, last=False):
        nonlocal t, hbm_free
        body_end = t+detail["before_export"]
        release = t+detail["last_input_ready"]
        record("inner_body", t, body_end, index, last_input_read_end=release)
        future = index+slots
        if last:
            record("cmem_release", release, release, index, slot=index % slots)
        # Opportunistic prefetch may use the tail-compute window, but a whole
        # panel that would delay this export is deferred until after export.
        early = last and future < count and max(hbm_free, release)+fill_duration(future) <= body_end
        if early:
            fill(future, release)
        export_start = max(body_end, hbm_free)
        t = hbm_free = export_start+detail["export_cycles"]
        record("output_export", export_start, t, index)
        if last and future < count and not early:
            fill(future, max(t, release))

    for index in range(slots):
        fill(index, 0)
    index, seen = 0, {}
    while index < count:
        row, col = divmod(index, columns)
        if accelerate and events is None:
            # At row boundaries, skip complete full rows. Within a row, skip
            # only full columns. Leave the tail and its two-slot lookahead
            # explicit: the boundary state already includes future fills.
            phase, limit = None, index
            if col == 0 and row < rows-3:
                phase, limit = ("row",), (rows-3)*columns
            elif col < columns-3:
                phase, limit = ("column", row), (row+1)*columns-3
            if phase is not None:
                state = (index % slots, max(0, hbm_free-t),
                         tuple(max(0, end-t) for end in ready),
                         tuple(tag-index for tag in tags))
                key = phase, state
                if key in seen:
                    old_index, old_t = seen[key]
                    period = index-old_index
                    jumps = (limit-index)//period
                    if jumps:
                        shift, advance = jumps*(t-old_t), jumps*period
                        t += shift
                        hbm_free += shift
                        ready = [end+shift for end in ready]
                        tags = [tag+advance for tag in tags]
                        index += advance
                        continue
                else:
                    seen[key] = index, t
        assert tags[index % slots] == index
        t = max(t, ready[index % slots])
        m, n = shape(index)
        outputs, total, first, last = panel(m, n)
        if events is not None:
            for mi in range(0, m, plan.m):
                for ni in range(0, n, plan.n):
                    _, detail = profile(min(plan.m, m-mi), min(plan.n, n-ni))
                    output(detail, index, mi+plan.m >= m and ni+plan.n >= n)
        else:
            if outputs > 1:
                output(first[1], index)
                # Any earlier fill has drained before the first export. No
                # new fill can start until this panel's last read, so middle
                # outputs have exactly their isolated pipeline duration.
                middle = total-first[0]-last[0]
                t += middle
                hbm_free = t
            output(last[1], index, last=True)
        index += 1
    return t
