"""Single-TensorCore logical counts and optional serial task timing."""
# 阅读顺序：make_plan 检查容量，simulate 按选定块大小求和，最后组装报告。
# 这里只统计单个 TensorCore 的逻辑工作；芯片库存不会放大本次任务的计数。
from .tiling import make_plan
from fractions import Fraction
from math import ceil


def _block_sizes(length, block):
    """完整块和尾块最多两种尺寸；重复次数替代逐块事件表。"""
    full, tail = divmod(length, block)
    return ([(block, full)] if full else []) + ([(tail, 1)] if tail else [])


def _serial_timing(spec, workload, plan, counts):
    """保持原 MNK 逻辑程序，在每块内部按 MXU 数量分批执行输出小块。"""
    spec.require_timing_inputs()
    v = spec.values
    rows, columns, mxus = (v[k] for k in ("mxu_rows", "mxu_columns", "mxus_per_tensorcore"))
    # 用有理数进行带宽换算/取整，避免浮点舍入凭空增加一个周期。
    clock = Fraction(str(v["clock_hz"]))
    mac_rate = Fraction(str(v["int8_macs_per_cell_cycle"]))
    hbm_share = Fraction(str(v["hbm_bandwidth_bytes_per_second"])) / v["tensorcores_per_chip"]
    hbm_bpc = hbm_share / clock
    if v["vmem_bandwidth_bytes_per_second"] is None:
        # 暂按每个 MXU 每拍供给 rows+columns 个 INT8 元素估算共享 VMEM 服务宽度。
        # 这是独立的模型假设，不能写回硬件配置或把 unknown 改成 documented。
        vmem_bpc = Fraction(mxus * (rows + columns))
        vmem_basis = "Proxy: mxus * (mxu_rows + mxu_columns) INT8 bytes/cycle; physical bandwidth remains unknown."
    else:
        vmem_bpc = Fraction(str(v["vmem_bandwidth_bytes_per_second"])) / clock
        vmem_basis = "Configured VMEM byte/s divided by clock; assume one shared read/write service."
    # DMA 同时访问 HBM/VMEM，按较慢一端计一次；不重复收取两个端点的时间。
    dma_bpc = min(hbm_bpc, vmem_bpc)
    fill_drain = rows + columns - 2
    stages = dict.fromkeys(("hbm_to_vmem", "vmem_to_mxu", "psum_read", "matrix_compute",
                            "matrix_fill_drain", "result_to_vmem", "vmem_to_hbm"), 0)
    total_waves = total_mxu_tiles = max_parallel = 0
    M, K, N = workload.shape
    k_blocks = (K + plan.k - 1) // plan.k
    for m, m_repeats in _block_sizes(M, plan.m):
        for n, n_repeats in _block_sizes(N, plan.n):
            outputs = m_repeats * n_repeats
            output_bytes = 4 * m * n
            # C 留在 VMEM 跨 K 累加，只有最后一次导出经过 HBM。
            stages["psum_read"] += outputs * (k_blocks - 1) * ceil(output_bytes / vmem_bpc)
            stages["result_to_vmem"] += outputs * k_blocks * ceil(output_bytes / vmem_bpc)
            stages["vmem_to_hbm"] += outputs * ceil(output_bytes / dma_bpc)
            mxu_tiles = ((m + rows - 1) // rows) * ((n + columns - 1) // columns)
            waves = (mxu_tiles + mxus - 1) // mxus
            max_parallel = max(max_parallel, min(mxus, mxu_tiles))
            for k, k_repeats in _block_sizes(K, plan.k):
                blocks = outputs * k_repeats
                input_bytes = m * k + k * n
                stages["hbm_to_vmem"] += blocks * ceil(input_bytes / dma_bpc)
                stages["vmem_to_mxu"] += blocks * ceil(input_bytes / vmem_bpc)
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
        "vmem_service_bytes_per_cycle": assumption(float(vmem_bpc), "byte/cycle", architecture, vmem_basis),
        "dma_service_bytes_per_cycle": assumption(float(dma_bpc), "byte/cycle", method,
            "Minimum of allocated HBM and VMEM service; both transfer endpoints are charged once."),
        "fill_drain_cycles_per_wave": assumption(fill_drain, "cycle", architecture,
            "Ideal 2D array proxy: rows + columns - 2; reserve full geometry even for tails. "
            "Not a published TPU v4 instruction latency or an XLA scheduling model."),
        "fixed_dma_cycles": assumption(0, "cycle", method,
            "Fixed DMA/dispatch latency is omitted explicitly; payload service only, not measured zero latency."),
        "int8_macs_per_cell_cycle": cell_rate,
    }
    total = sum(stages.values())
    time_ns = float(total * 10**9 / clock)
    return {
        "status": "analytical_estimate", "model": "single_tensorcore_serial_v1",
        "scope": "single_tensorcore", "policy": "serial_logical_blocks_and_stages; parallel_MXUs_within_each_wave",
        "total_cycles": total, "chip_cycles": None, "time_ns": time_ns,
        "throughput_gops": counts["useful_ops"] / time_ns,
        "stages_cycles": stages, "scheduled_blocks": counts["compute_blocks"],
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
            "This is an uncalibrated estimate, not a real-hardware upper/lower bound; internal bandwidth unknowns remain unchanged.",
        ],
    }


class TPUSimulator:
    def __init__(self, spec):
        self.spec = spec

    def simulate(self, workload, *, tile=(128, 128, 128), mode="count"):
        if mode not in {"count", "timing"}:
            raise ValueError("mode must be count or timing")
        if (workload.data_format, workload.accumulation_dtype, workload.output_dtype) != ("INT8", "INT32", "INT32"):
            raise ValueError("TPU counting supports INT8 inputs / INT32 logical accumulation and output; use --format INT8")
        spec = self.spec
        spec.require_count_capacities()
        plan = make_plan(workload, spec.values["vmem_usable_bytes"], spec.values["hbm_usable_bytes"], tile)
        M, K, N = workload.shape
        # Exact sums of MNK logical tiles, including tails. No per-tile event lists.
        nm, nk, nn = ((dim + block - 1) // block for dim, block in zip(workload.shape, (plan.m, plan.k, plan.n)))
        # MNK 顺序下不跨输出块保留输入：A 随 N 块重读，B 随 M 块重读。
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
            "executed_padded_ops": None,
            "compute_blocks": nm * nk * nn,
            "output_blocks": nm * nn,
            "logical_output_bytes": c_bytes,
            "input_a_transfer_bytes": a_bytes,
            "input_b_transfer_bytes": b_bytes,
            "input_transfer_bytes": a_bytes + b_bytes,
            "psum_read_bytes": psum_bytes,
            "psum_write_bytes": psum_bytes,
        }
        # Endpoint actions, not additional end-to-end transfer charges. Partial writes
        # exclude the last K block; final writes and export reads are counted once.
        # 同一次传输在 HBM 和 VMEM 端各有动作；端点动作与传输摘要不能重复计费。
        actions = {
            "matrix_mac": (M * K * N, "MAC", "logical_arithmetic"),
            "hbm_read_bytes": (a_bytes + b_bytes, "byte", "logical_payload"),
            "hbm_write_bytes": (c_bytes, "byte", "logical_payload"),
            "vmem_input_write_bytes": (a_bytes + b_bytes, "byte", "logical_payload"),
            "vmem_input_read_bytes": (a_bytes + b_bytes, "byte", "logical_payload"),
            "vmem_psum_read_bytes": (psum_bytes, "byte", "logical_payload"),
            "vmem_psum_write_bytes": (psum_bytes, "byte", "logical_payload"),
            "vmem_final_write_bytes": (c_bytes, "byte", "logical_payload"),
            "vmem_export_read_bytes": (c_bytes, "byte", "logical_payload"),
        }
        result = {
            "schema_version": 2, "model": "tpu_v4_count", "mode": "count",
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
                "status": "assumed_logical_program", "loop_order": "MNK", "buffers": 1,
                "requested_tile_mkn": list(tile), "selected": plan.to_dict(),
                "tail_policy": "actual_dimensions; physical_padding_not_evaluated",
                "cross_output_block_input_reuse": False,
                "partial_sum": "C stays in logical VMEM across K; read/write once per K block, first block has no old-C read",
                "input_access": "one logical input read per block at aggregate matrix service; no MXU fanout model",
                "accumulation": "INT32 project workload convention; native TPU numerical semantics unverified",
                "scratch": "only A/B/C logical storage; runtime, register spills and packing workspace not modeled",
            },
            "counts": counts,
            "action_counts": {name: {"count": count, "unit": unit, "scope": "single_tensorcore", "basis": basis}
                              for name, (count, unit, basis) in actions.items()},
            "memory": {
                "vmem": {"scope": "tensorcore", "usable_bytes": spec.values["vmem_usable_bytes"],
                         "peak_logical_bytes": plan.peak_working_bytes},
                "hbm": {"scope": "chip", "usable_bytes": spec.values["hbm_usable_bytes"],
                        "logical_footprint_bytes": plan.hbm_footprint_bytes},
            },
            "hardware_derived": spec.derived(), "hardware_hierarchy": spec.hierarchy(),
            # 私有 VMEM 的芯片合计只是库存，单核不能使用其他核的私有容量。
            "hardware_inventory": {"per_tensorcore": spec.tensorcore_inventory(), "chip": spec.inventory()},
            "interconnect_specification": dict(spec.interconnect),
            "parameter_provenance": spec.provenance(), "missing_parameters": spec.missing_parameters(),
            "timing": {"status": "not_evaluated", "total_cycles": None, "time_ns": None, "throughput_gops": None},
            "energy": {"status": "not_evaluated", "total_pj": None},
            "area": {"status": "not_evaluated", "total_mm2": None},
            "limitations": [
                "Chip hardware hierarchy and inventory; single-TensorCore logical counts only.",
                "Private VMEM totals are inventory, not pooled usable capacity; shared resources are counted once.",
                "Chip peaks are specification arithmetic; no multi-TensorCore scheduling or count scaling.",
                "No hardware padding, physical transactions, numerical simulation or compiler mapping.",
                "No MXU placement, weight residency, CMEM execution, VPU/SFU execution or timing.",
                "Action coverage is partial; unknown internal activity is not treated as zero.",
                "Transfer summaries overlap endpoint actions; do not charge both in an energy model.",
            ],
        }
        if mode == "timing":
            result.update(model="tpu_v4_analytical", mode="timing")
            result["timing"] = _serial_timing(spec, workload, plan, counts)
            result["boundary"]["timing_resolution"] = "serial_logical_blocks_with_MXU_waves_and_explicit_assumptions"
            result["limitations"][0] = "Chip hardware inventory; execution and timing cover one TensorCore only."
            result["limitations"][3] = "Timing reserves full MXU tiles but actions remain logical; no physical transactions, numerical simulation or XLA mapping."
            result["limitations"][4] = "No detailed weight residency, CMEM/SparseCore execution or calibrated VPU/SFU timing."
        return result
