"""Buffered K-block pipeline with shared VMEM and HBM/CMEM input service.

Output blocks remain serial. Repeated shapes are evaluated once; steady K-block
states can be advanced without expanding every block into an event object.
The CMEM outer scheduler uses block profiles; final output always uses HBM.
"""
from fractions import Fraction
from math import ceil


def pipeline_cycles(spec, workload, plan, slots, *, input_bpc=None, block_details=False):
    v = spec.values
    clock = Fraction(str(v["clock_hz"]))
    hbm = Fraction(str(v["hbm_bandwidth_bytes_per_second"])) / v["tensorcores_per_chip"] / clock
    vmem = Fraction(str(v["vmem_bandwidth_bytes_per_second"])) / clock
    dma = min(hbm, vmem)
    source = "hbm" if input_bpc is None else "cmem_read"
    input_dma = dma if input_bpc is None else min(input_bpc, vmem)
    rate = Fraction(str(v["int8_macs_per_cell_cycle"]))
    rows, cols, mxus = (v[n] for n in ("mxu_rows", "mxu_columns", "mxus_per_tensorcore"))
    latency = spec.execution["fixed_latency_cycles"]["transfer_startup"]

    def sizes(length, block):
        full, tail = divmod(length, block)
        return ([(block, full)] if full else []) + ([(tail, 1)] if tail else [])

    total = 0
    details = []
    busy_total = dict.fromkeys(("hbm", "vmem", "mxu"), 0)
    if source != "hbm":
        busy_total[source] = 0
    for m, mr in sizes(workload.M, plan.m):
        for n, nr in sizes(workload.N, plan.n):
            blocks = ceil(workload.K / plan.k)
            waves = ceil(ceil(m/rows) * ceil(n/cols) / mxus)
            source_free = vfree = xfree = ready = 0
            busy = dict.fromkeys(busy_total, 0)

            def transfer(size, external=False, earliest=0, input_load=False):
                nonlocal source_free, vfree
                bandwidth = input_dma if input_load else dma if external else vmem
                duration = ceil(size / bandwidth) + latency
                start = max(vfree, earliest, source_free if input_load or (external and source == "hbm") else 0)
                vfree = start + duration
                busy["vmem"] += duration
                if input_load:
                    source_free = vfree
                    busy[source] += duration
                elif external:
                    if source == "hbm":
                        source_free = vfree
                    busy["hbm"] += duration
                return vfree

            def ksize(i):
                return min(plan.k, workload.K - i*plan.k)

            ready = transfer((m+n)*ksize(0), input_load=True)
            i, seen = 0, {}
            while i < blocks:
                # Repetition is valid only for interior full K blocks, each with
                # old C and a full next prefetch. Tail and final export stay explicit.
                if 0 < i < blocks-2:
                    origin = max(source_free, vfree, xfree, ready)
                    key = (source_free-origin, vfree-origin, xfree-origin, ready-origin)
                    if key in seen:
                        old_i, old_origin, old_busy = seen[key]
                        period = i-old_i
                        jumps = (blocks-2-i)//period
                        if jumps:
                            shift = jumps*(origin-old_origin)
                            source_free += shift; vfree += shift; xfree += shift; ready += shift
                            for resource in busy:
                                busy[resource] += jumps*(busy[resource]-old_busy[resource])
                            i += jumps*period
                            continue
                    else:
                        seen[key] = (i, origin, dict(busy))
                k = ksize(i)
                transfer((m+n)*k, earliest=ready)
                if i:
                    transfer(4*m*n)
                compute = waves*(ceil(k/rate)+rows+cols-2)
                xfree = max(xfree, vfree)+compute
                busy["mxu"] += compute
                # A second input slot permits next-K DMA during current compute.
                if slots == 2 and i+1 < blocks:
                    ready = transfer((m+n)*ksize(i+1), input_load=True)
                transfer(4*m*n, earliest=xfree)
                if slots == 1 and i+1 < blocks:
                    ready = transfer((m+n)*ksize(i+1), input_load=True)
                i += 1
            before_export = max(source_free, vfree, xfree)
            last_input_ready = source_free
            export_end = transfer(4*m*n, True)
            if block_details:
                details.append({"m": m, "n": n, "before_export": before_export,
                                "last_input_ready": last_input_ready,
                                "export_cycles": export_end - before_export})
            repeat = mr*nr
            total += max(source_free, vfree, xfree)*repeat
            for resource in busy:
                busy_total[resource] += busy[resource]*repeat
    return (total, busy_total, details) if block_details else (total, busy_total)
