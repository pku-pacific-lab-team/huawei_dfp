"""Single-TensorCore counts and serial/pipelined timing with optional CMEM panels."""
from ..metrics import attach_contract, attach_timing_breakdowns
from .pipeline import pipeline_cycles
from .cmem_pipeline import cmem_pipeline_cycles
from .cmem import make_cmem_plan, input_traffic, services as cmem_services, fill_cycles
# 阅读顺序：make_plan 检查容量，simulate 按选定块大小求和，最后组装报告。
# 这里只统计单个 TensorCore 的逻辑工作；芯片库存不会放大本次任务的计数。
from .tiling import make_plan
from fractions import Fraction
from math import ceil


def _block_sizes(length, block):
    """完整块和尾块最多两种尺寸；重复次数替代逐块事件表。"""
    full, tail = divmod(length, block)
    return ([(block, full)] if full else []) + ([(tail, 1)] if tail else [])


def _serial_timing(spec, workload, plan, counts, *, cmem_read_bpc=None):
    """保持原 MNK 逻辑程序，在每块内部按 MXU 数量分批执行输出小块。"""
    spec.require_timing_inputs()
    v = spec.values
    rows, columns, mxus = (v[k] for k in ("mxu_rows", "mxu_columns", "mxus_per_tensorcore"))
    # 用有理数进行带宽换算/取整，避免浮点舍入凭空增加一个周期。
    clock = Fraction(str(v["clock_hz"]))
    mac_rate = Fraction(str(v["int8_macs_per_cell_cycle"]))
    hbm_share = Fraction(str(v["hbm_bandwidth_bytes_per_second"])) / v["tensorcores_per_chip"]
    hbm_bpc = hbm_share / clock
    # 服务假设显式保存在 JSON；null 不再触发隐藏的带宽回退。
    vmem_bpc = Fraction(str(v["vmem_bandwidth_bytes_per_second"])) / clock
    vmem_basis = "Configured VMEM byte/s divided by clock; assume one shared read/write service."
    # DMA 同时访问 HBM/VMEM，按较慢一端计一次；不重复收取两个端点的时间。
    dma_bpc = min(hbm_bpc, vmem_bpc)
    input_stage = "hbm_to_vmem" if cmem_read_bpc is None else "cmem_to_vmem"
    input_bpc = dma_bpc if cmem_read_bpc is None else min(cmem_read_bpc, vmem_bpc)
    startup = spec.execution["fixed_latency_cycles"]["transfer_startup"]
    fill_drain = rows + columns - 2
    stages = dict.fromkeys((input_stage, "vmem_to_mxu", "psum_read", "matrix_compute",
                            "matrix_fill_drain", "result_to_vmem", "vmem_to_hbm"), 0)
    total_waves = total_mxu_tiles = max_parallel = 0
    M, K, N = workload.shape
    k_blocks = (K + plan.k - 1) // plan.k
    for m, m_repeats in _block_sizes(M, plan.m):
        for n, n_repeats in _block_sizes(N, plan.n):
            outputs = m_repeats * n_repeats
            output_bytes = 4 * m * n
            # C 留在 VMEM 跨 K 累加，只有最后一次导出经过 HBM。
            stages["psum_read"] += outputs * (k_blocks - 1) * (ceil(output_bytes / vmem_bpc) + startup)
            stages["result_to_vmem"] += outputs * k_blocks * (ceil(output_bytes / vmem_bpc) + startup)
            stages["vmem_to_hbm"] += outputs * (ceil(output_bytes / dma_bpc) + startup)
            mxu_tiles = ((m + rows - 1) // rows) * ((n + columns - 1) // columns)
            waves = (mxu_tiles + mxus - 1) // mxus
            max_parallel = max(max_parallel, min(mxus, mxu_tiles))
            for k, k_repeats in _block_sizes(K, plan.k):
                blocks = outputs * k_repeats
                input_bytes = m * k + k * n
                stages[input_stage] += blocks * (ceil(input_bytes / input_bpc) + startup)
                stages["vmem_to_mxu"] += blocks * (ceil(input_bytes / vmem_bpc) + startup)
                # 同一 wave 的 MXU 并行，最后不足四个的 wave 仍要完整完成一次计算。
                stages["matrix_compute"] += blocks * waves * ceil(k / mac_rate)
                stages["matrix_fill_drain"] += blocks * waves * fill_drain
                total_waves += blocks * waves
                total_mxu_tiles += blocks * mxu_tiles

    official = "https://docs.cloud.google.com/tpu/docs/v4"
    architecture = "https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm"
    method = "https://docs.jax.dev/en/latest/pallas/tpu/matmul.html#matrix-multiplication-performance"

    def assumption(value, unit, source, conditions):
        return {"value": value, "unit": unit, "scope": "single_tensorcore", "status": "assumed",
                "source": source, "source_version": "Public references checked 2026-11-16; serial model v1",
                "conditions": conditions}

    cell_rate = spec.provenance()["int8_macs_per_cell_cycle"]
    cell_rate["conditions"] += (" Timing v1 explicitly reuses this value as ideal active-array service; "
                                "effective instruction throughput is not calibrated.")
    parameters = {
        "parallel_mxus": assumption(mxus, "unit", official,
            "At most the configured MXUs in one TensorCore; independent output tiles in each logical block form waves. "
            "No parallelism across logical blocks or across TensorCores."),
        "hbm_share_bytes_per_second": assumption(float(hbm_share), "byte/s", official,
            "Even partition of chip bandwidth between TensorCores; no borrowing from an idle core."),
        "vmem_service_bytes_per_cycle": {**spec.provenance()["vmem_bandwidth_bytes_per_second"],
            "value": float(vmem_bpc), "unit": "byte/cycle", "selection": "configured",
            "conditions": spec.parameters["vmem_bandwidth_bytes_per_second"].conditions + " " + vmem_basis},
        "dma_service_bytes_per_cycle": assumption(float(dma_bpc), "byte/cycle", method,
            "Minimum of allocated HBM and VMEM service; both transfer endpoints are charged once."),
        "fill_drain_cycles_per_wave": assumption(fill_drain, "cycle", architecture,
            "Ideal 2D array proxy: rows + columns - 2; reserve full geometry even for tails. "
            "Not a published TPU v4 instruction latency or an XLA scheduling model."),
        "fixed_dma_cycles": assumption(startup, "cycle", method,
            "Configured fixed transfer allowance in cycles; zero explicitly omits this allowance, not measured zero latency."),
        "int8_macs_per_cell_cycle": cell_rate,
    }
    total = sum(stages.values())
    time_ns = float(total * 10**9 / clock)
    return attach_timing_breakdowns({
        "status": "analytical_estimate", "model": "single_tensorcore_serial_v1",
        "scope": "single_tensorcore", "policy": "serial_logical_blocks_and_stages; parallel_MXUs_within_each_wave",
        "total_cycles": total, "chip_cycles": None, "time_ns": time_ns,
        "throughput_gops": counts["useful_ops"] / time_ns,
        "scheduled_blocks": counts["compute_blocks"],
        "mxu_output_tiles": total_mxu_tiles, "matrix_waves": total_waves,
        "max_parallel_mxus_used": max_parallel,
        "parameters": parameters, "clock": spec.provenance()["clock_hz"],
        "assumptions": [
            "INT8 input and INT32 accumulator/output follow the project convention; no TPU numerical emulation.",
            "Each logical block is serialized. Its output is split into full MXU-sized tiles; tails idle without shortening a wave.",
            "Logical input tiles are read once from VMEM and ideally shared across MXUs; no fanout traffic or persistent weights between output blocks.",
            "Padding is implicit inside the array proxy; no extra padded VMEM allocation or physical action counts are claimed.",
            "Old C seeds matrix accumulation across K; MAC service includes the addition, with explicit VMEM partial-sum read/write stages.",
            "No DMA/compute overlap, bank conflicts, compiler scheduling, register spills, CMEM/SparseCore execution or chip scheduling.",
            "This is an uncalibrated estimate, not a real-hardware upper/lower bound; configured assumed services are not measured hardware bandwidths.",
        ],
    }, service=stages, elapsed=stages)


class TPUSimulator:
    def __init__(self, spec):
        self.spec = spec

    def simulate(self, workload, *, tile=None, mode="count", schedule=None, buffers=None,
                 cmem_mode=None, cmem_groups_mn=None, cmem_buffers=None):
        if mode not in {"count", "timing"}:
            raise ValueError("mode must be count or timing")
        if (workload.data_format, workload.accumulation_dtype, workload.output_dtype) != ("INT8", "INT32", "INT32"):
            raise ValueError("TPU counting supports INT8 inputs / INT32 logical accumulation and output; use --format INT8")
        spec = self.spec
        execution = spec.execution
        tile = tuple(execution["default_tile_mkn"]) if tile is None else tile
        schedule = execution["schedule"] if schedule is None else schedule
        buffers = execution["buffer_slots"] if buffers is None else buffers
        cmem_mode = execution["cmem_mode"] if cmem_mode is None else cmem_mode
        cmem_groups_mn = execution["cmem_groups_mn"] if cmem_groups_mn is None else cmem_groups_mn
        cmem_buffers = execution["cmem_buffer_slots"] if cmem_buffers is None else cmem_buffers
        if schedule not in {"serial", "resource_pipeline"}:
            raise ValueError("Unsupported TPU schedule")
        if type(buffers) is not int or buffers not in (1, 2):
            raise ValueError("buffers must be 1 or 2")
        spec.require_count_capacities()
        plan = make_plan(workload, spec.values["vmem_usable_bytes"], spec.values["hbm_usable_bytes"], tile)
        cmem = make_cmem_plan(spec, workload, plan, cmem_mode, cmem_groups_mn, cmem_buffers)
        if cmem["mode"] == "bypass":
            cmem["outer_schedule"] = "inactive_bypass"
        elif schedule == "serial":
            cmem["outer_schedule"] = "serial_outer_fills_and_inner_stages"
        cmem_active = cmem["mode"] == "tiled"
        traffic = input_traffic(workload, plan, cmem)
        M, K, N = workload.shape
        input_slot_bytes = plan.m*plan.k + plan.k*plan.n
        output_slot_bytes = 4*plan.m*plan.n
        slots = min(buffers, (spec.values["vmem_usable_bytes"]-output_slot_bytes)//input_slot_bytes)
        rows, columns = spec.values["mxu_rows"], spec.values["mxu_columns"]
        issued = None if rows is None or columns is None else (
            2*K*sum(ceil(m/rows)*rows*r for m,r in _block_sizes(M,plan.m))
            *sum(ceil(n/columns)*columns*r for n,r in _block_sizes(N,plan.n)))
        # Exact sums of MNK logical tiles, including tails. No per-tile event lists.
        nm, nk, nn = ((dim + block - 1) // block for dim, block in zip(workload.shape, (plan.m, plan.k, plan.n)))
        # VMEM 不跨输出块保留输入；CMEM 外层驻留可减少对应的 HBM 重读。
        # K 的尾块按实际尺寸计数，因此遍历完所有 K 块后仍恰好覆盖完整输入。
        a_bytes = nn * M * K
        b_bytes = nm * K * N
        c_bytes = 4 * M * N
        # nk 段 K 之间有 nk-1 个交界，每个交界保存并读回一次 INT32 部分和。
        # 最后一段产生最终 C，其写入与导出另计，不能再算一次中间部分和。
        psum_bytes = (nk - 1) * c_bytes
        counts = {
            "useful_macs": M * K * N,
            "useful_ops": 2 * M * K * N,
            "executed_padded_ops": issued,
            "compute_blocks": nm * nk * nn,
            "output_blocks": nm * nn,
            "logical_output_bytes": c_bytes,
            **traffic,
            "psum_read_bytes": psum_bytes,
            "psum_write_bytes": psum_bytes,
        }
        # Endpoint actions, not additional end-to-end transfer charges. Partial writes
        # exclude the last K block; final writes and export reads are counted once.
        # 同一次传输在 HBM 和 VMEM 端各有动作；端点动作与传输摘要不能重复计费。
        actions = {
            "matrix_mac": (M * K * N, "MAC", "logical_arithmetic"),
            "hbm_read_bytes": (traffic["hbm_input_read_bytes"], "byte", "logical_payload"),
            "hbm_write_bytes": (c_bytes, "byte", "logical_payload"),
            "cmem_write_bytes": (traffic["cmem_fill_bytes"], "byte", "logical_payload"),
            "cmem_read_bytes": (traffic["cmem_to_vmem_bytes"], "byte", "logical_payload"),
            "vmem_input_write_bytes": (a_bytes + b_bytes, "byte", "logical_payload"),
            "vmem_input_read_bytes": (a_bytes + b_bytes, "byte", "logical_payload"),
            "vmem_psum_read_bytes": (psum_bytes, "byte", "logical_payload"),
            "vmem_psum_write_bytes": (psum_bytes, "byte", "logical_payload"),
            "vmem_final_write_bytes": (c_bytes, "byte", "logical_payload"),
            "vmem_export_read_bytes": (c_bytes, "byte", "logical_payload"),
        }
        result = {
            "schema_version": 13, "model": "tpu_v4_count", "mode": "count",
            "workload": workload.to_dict(),
            "boundary": {
                "scope": "single_tensorcore", "simulated_tensorcores": 1,
                "hardware_scope": "chip", "counting_scope": "single_tensorcore",
                "chip_execution_included": False,
                "start": "logical_INT8_A_B_in_HBM", "end": "logical_INT32_C_in_HBM",
                "numeric_values_simulated": False, "packing_included": False,
                "host_transfer_included": False,
            },
            "counting_policy": {
                "status": "assumed_logical_program", "loop_order": "outer_MN_inner_MNK" if cmem_active else "MNK", "buffers": slots,
                "requested_tile_mkn": list(tile), "selected": plan.to_dict(),
                "tail_policy": "logical payload; issued array arithmetic pads M/N to MXU geometry; no extra physical traffic claim",
                "cross_output_block_input_reuse": traffic["hbm_read_bytes_saved_vs_bypass"] > 0,
                "input_transfer_boundary": "source_to_VMEM; HBM input reads and CMEM fills are reported separately",
                "partial_sum": "C stays in logical VMEM across K; read/write once per K block, first block has no old-C read",
                "input_access": "one logical input read per block at aggregate matrix service; no MXU fanout model",
                "accumulation": "INT32 project workload convention; native TPU numerical semantics unverified",
                "scratch": "only A/B/C logical storage; runtime, register spills and packing workspace not modeled",
            },
            "counts": counts,
            "action_counts": {name: {"count": count, "unit": unit, "scope": "single_tensorcore", "basis": basis}
                              for name, (count, unit, basis) in actions.items()},
            "memory": {
                "cmem": cmem,
                "vmem": {"scope": "tensorcore", "usable_bytes": spec.values["vmem_usable_bytes"],
                         "peak_logical_bytes": plan.peak_working_bytes,
                         "allocated_bytes": slots*input_slot_bytes+output_slot_bytes,
                         "input_buffer_slots": slots},
                "hbm": {"scope": "chip", "usable_bytes": spec.values["hbm_usable_bytes"],
                        "logical_footprint_bytes": plan.hbm_footprint_bytes},
            },
            "hardware_derived": spec.derived(), "hardware_hierarchy": spec.hierarchy(),
            # 私有 VMEM 的芯片合计只是库存，单核不能使用其他核的私有容量。
            "hardware_inventory": {"per_tensorcore": spec.tensorcore_inventory(), "chip": spec.inventory()},
            "parameter_provenance": spec.provenance(), "missing_parameters": spec.missing_parameters(),
            "timing": {"status": "not_evaluated", "total_cycles": None, "time_ns": None, "throughput_gops": None},
            "energy": {"status": "not_evaluated", "total_pj": None},
            "area": {"status": "not_evaluated", "total_mm2": None},
            "limitations": [
                "Chip hardware hierarchy and inventory; single-TensorCore logical counts only.",
                "Private VMEM totals are inventory, not pooled usable capacity; shared resources are counted once.",
                "Chip peaks are specification arithmetic; no multi-TensorCore scheduling or count scaling.",
                "No hardware padding, physical transactions, numerical simulation or compiler mapping.",
                "CMEM is bypassed: reserved capacity/bandwidth do not affect tiling, reuse or timing. Vector/Nonlinear has a standalone compute service, not scheduled by GEMM.",
                "Action coverage is partial; unknown internal activity is not treated as zero.",
                "Transfer summaries overlap endpoint actions; do not charge both in an energy model.",
            ],
        }
        if mode == "timing":
            spec.require_timing_inputs()
            services = cmem_services(spec) if cmem_active else None
            read_bpc = services["read"] if cmem_active else None
            fills = fill_cycles(workload, cmem, services["fill"], execution["fixed_latency_cycles"]["transfer_startup"]) if cmem_active else 0
            result.update(model="tpu_v4_analytical", mode="timing")
            result["timing"] = _serial_timing(spec, workload, plan, counts, cmem_read_bpc=read_bpc)
            timing = result["timing"]
            service_sum = timing["total_cycles"] + fills
            if schedule == "resource_pipeline":
                cycles, busy = pipeline_cycles(spec, workload, plan, slots, input_bpc=read_bpc)
                if cmem_active:
                    serialized_outer = cycles + fills
                    cycles = cmem_pipeline_cycles(spec, workload, plan, slots, cmem, services)
                    timing["cmem_outer_overlap_saved_cycles"] = serialized_outer-cycles
                    timing["serialized_cmem_outer_cycles"] = serialized_outer
                    busy["hbm"] += fills
                    busy["cmem_write"] = fills
                timing.update(total_cycles=cycles, time_ns=cycles*1e9/spec.values["clock_hz"],
                              resource_busy_cycles=busy,
                              resource_utilization={k:v/cycles for k,v in busy.items()})
                timing["throughput_gops"] = counts["useful_ops"]/timing["time_ns"]
                timing["model"] = "single_tensorcore_buffered_pipeline_v1"
                timing["policy"] = "serial_output_blocks; buffered_K_prefetch; shared_HBM_VMEM; parallel_MXU_waves"
                timing["assumptions"] = [x for x in timing["assumptions"] if "No DMA/compute overlap" not in x and "Each logical block is serialized" not in x]
                timing["assumptions"].append("Two input slots enable next-K prefetch during compute when capacity permits; VMEM transfers share one service. Output blocks remain serialized; no bank conflicts or XLA scheduling.")
            elif cmem_active:
                timing.update(total_cycles=service_sum, time_ns=service_sum*1e9/spec.values["clock_hz"])
                timing["throughput_gops"] = counts["useful_ops"]/timing["time_ns"]
            if cmem_active:
                timing["service_cycles"]["by_stage"]["hbm_to_cmem"] = fills
                timing["model"] = "single_tensorcore_cmem_" + schedule + ("_v2" if schedule == "resource_pipeline" else "_v1")
                outer_policy = "buffered_outer_prefetch; shared_HBM; early_slot_release" if schedule == "resource_pipeline" else "serial_outer_fills"
                timing["policy"] = "cold_full_K_panels; " + outer_policy + "; " + timing["policy"]
                timing["parameters"]["dma_service_bytes_per_cycle"]["conditions"] = "Output VMEM-to-HBM service only; min of allocated HBM and VMEM rates."
                for direction in ("read", "write"):
                    p = spec.provenance()[f"cmem_{direction}_bandwidth_bytes_per_second"]
                    p.update(value=float(services[direction]), unit="byte/cycle", scope="single_tensorcore",
                             conditions=p["conditions"] + " Allocated service = chip byte/s / TensorCore count / clock; no idle-share borrowing.")
                    timing["parameters"][f"cmem_{direction}_service_bytes_per_cycle"] = p
                timing["cmem_transfer_services_bytes_per_cycle"] = {
                    "hbm_to_cmem": float(services["fill"]), "cmem_to_vmem": float(services["to_vmem"])}
                timing["assumptions"] = [a.replace("CMEM/SparseCore execution", "SparseCore execution")
                                         .replace("no fanout traffic or persistent weights between output blocks", "no fanout traffic; input panels are reused within each CMEM outer tile")
                                         for a in timing["assumptions"]]
                timing["assumptions"].append("Cold CMEM holds full-K A/B panels in one or two capacity-checked slots. Each panel is completely filled before its first read. Pipeline scheduling permits next-panel fill alongside inner execution, and releases a slot after its last CMEM read. Fills and output exports share one non-preemptive HBM service; a refill that cannot finish before the current export is ready is deferred until after that export. Serial scheduling disables these overlaps. CMEM read and write services are independent across different live slots; each DMA reserves both endpoints once at their minimum rate. C stays in VMEM across K and exports directly to HBM. No cross-outer retention, partial-fill consumption, physical layout/coalescing, OCI or bank-conflict model; transfer_startup uses the common explicit allowance.")
            timing.update(overlapped_service_cycles=service_sum-timing["total_cycles"], input_buffer_slots=slots)
            service_stages = timing["service_cycles"]["by_stage"]
            attach_timing_breakdowns(timing, service=service_stages,
                                    elapsed=service_stages if schedule == "serial" else None)
            result["boundary"]["timing_resolution"] = schedule + "_with_explicit_MXU_waves"
            result["limitations"][0] = "Chip hardware inventory; execution and timing cover one TensorCore only."
            result["limitations"][3] = "Timing reserves full MXU tiles but actions remain logical; no physical transactions, numerical simulation or XLA mapping."
            result["limitations"][4] = "No detailed weight residency or SparseCore execution. CMEM is bypassed; reserved Vector services do not add GEMM time."
        if cmem_active:
            result["limitations"][4] = "CMEM uses assumed full-K panel placement, capacity-checked buffering and uncalibrated bandwidth proxies (libtpu v4 baseline assigned to read; v4i-derived 2:1 read/write ratio). No partial-fill consumption, cross-output VMEM pipeline or cross-task residency. Vector/Nonlinear is compute-only outside GEMM; SparseCore/SPMEM are retired."
        return attach_contract(result, {**execution, "default_tile_mkn": list(tile),
                                       "buffer_slots": slots, "schedule": schedule,
                                       "cmem_mode": cmem_mode, "cmem_groups_mn": list(cmem_groups_mn),
                                       "cmem_buffer_slots": cmem_buffers, "effective_cmem_buffer_slots": cmem["buffer_slots"],
                                       "effective_cmem_mode": cmem["mode"]})
