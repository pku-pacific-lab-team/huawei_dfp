"""Single Cluster INT8 counts and an explicitly simplified serial timing model."""
# 这里把一个固定 domain 映射到共享 Macro 计数器，再补充上层广播和归约口径。
# 本文件不重复实现 Macro 逐位动作，也不把整片库存当作同时执行的资源。
from ..cim.simulator import LOOP_ORDERS, count_macro_tiles
from ..cim.macro_model import positive_integer
from .tiling import TSPCIMTilingEngine
from ...utils.workload import parse_workloads


_SOURCE_ROOT = ("https://github.com/pku-pacific-lab-team/LPU-HUAWEI-2012/blob/"
                "c641844c1d213d02a7d961d889838cae9b9dc8aa/")
_DESIGN_SOURCE = _SOURCE_ROOT + "docs/legacy/patent-idea-review.md"
_LATENCY_SOURCE = _SOURCE_ROOT + "tools/qwen3_ffn_cim_eval/latency_model.py"


class _SerialTaskTiming:
    """按 tile 累加阶段时间；复用 Macro 计数器的事件，不模拟逐拍流水。"""

    def __init__(self, spec):
        self.spec = spec
        self.stages = dict.fromkeys(("load", "switch", "compute", "drain",
                                     "partial_read", "partial_add", "write"), 0)
        self.tiles = self.load_batches = self.switch_batches = 0

        def parameter(value, unit, status, source, conditions):
            return {"value": value, "unit": unit, "scope": "single_cluster",
                    "status": status, "source": source,
                    "source_version": "commit c641844c1d213d02a7d961d889838cae9b9dc8aa",
                    "conditions": conditions}

        self.parameters = {
            "load_scan_cycles": parameter(spec.derived()["lw_full_context_scan_cycles"]["value"],
                "cycle", "assumed", _DESIGN_SOURCE + "#L430-L439",
                "Reserve one documented full 160-cycle row scan whenever any Macro needs programming; "
                "unused slots idle. Reused weights cost zero load cycles; no separate physical write penalty."),
            "input_ii_cycles": parameter(spec.macro_spec.values["input_ii_cycles"], "cycle", "documented",
                _DESIGN_SOURCE + "#L744-L750", "INT8 Cluster ACC occupancy is rows * II, including tails."),
            "local_pipeline_cycles": parameter(3, "cycle", "assumed", _LATENCY_SOURCE + "#L26-L30",
                "Reuse the external analytical model's local pipeline allowance; not measured Macro first latency."),
            "reduction_hop_cycles": parameter(1, "cycle", "assumed", _LATENCY_SOURCE + "#L200-L207",
                "Drain allowance is II + local_pipeline + ceil(k/16) * hop; includes a capture step. "
                "Core stagger is absorbed into this coarse allowance, not separately scheduled."),
            "switch_cycles": parameter(1, "cycle", "assumed", _DESIGN_SOURCE + "#L649-L669",
                "One serialized Cluster-wide switch stage if any Macro changes active context, "
                "including initial activation; simplifies the documented per-Core rolling switch."),
            "writer_bpc": parameter(spec.values["nim_output_bpc"], "byte/cycle", "assumed",
                _DESIGN_SOURCE + "#L545-L584",
                "Use the existing 256 B/cycle writer target as one dedicated INT32 result/partial-read "
                "service for the selected Cluster. Do not multiply by NIM tiles or use whole-chip bandwidth."),
            "external_adds_per_cycle": parameter(spec.values["nim_output_bpc"] // 4, "add/cycle", "assumed",
                _DESIGN_SOURCE + "#L568",
                "For K segments after the first, assume one INT32 add per writer-width element (64/cycle). "
                "This is a service proxy, not calibrated VXM throughput or added hardware inventory."),
        }

    def account_tile(self, *, m, k, n, k_start, programs, switches):
        p, stages = self.parameters, self.stages
        # 一次扫描只写本次真正缺失的 Macro 权重；空闲槽保留，已有四套 context 继续复用。
        stages["load"] += p["load_scan_cycles"]["value"] if programs else 0
        stages["switch"] += p["switch_cycles"]["value"] if switches else 0
        stages["compute"] += m * p["input_ii_cycles"]["value"]
        slices = (k + self.spec.core.k - 1) // self.spec.core.k
        stages["drain"] += (p["input_ii_cycles"]["value"] + p["local_pipeline_cycles"]["value"]
                            + slices * p["reduction_hop_cycles"]["value"])
        # 单域先完成 K 归约，才写有效 N 列的 INT32 结果，不能写出每个 Macro 的局部和。
        bpc = p["writer_bpc"]["value"]
        write_cycles = (m * n * 4 + bpc - 1) // bpc
        if k_start:
            # 后续 K 段读已有部分和、相加，再共用下方 write 阶段覆盖写回；不重复计最终写出。
            stages["partial_read"] += write_cycles
            adds_per_cycle = p["external_adds_per_cycle"]["value"]
            stages["partial_add"] += (m * n + adds_per_cycle - 1) // adds_per_cycle
        stages["write"] += write_cycles
        self.tiles += 1
        self.load_batches += bool(programs)
        self.switch_batches += bool(switches)

    def report(self, useful_ops):
        total = sum(self.stages.values())
        clock = self.spec.clock_freq_ghz
        time_ns = total / clock if clock is not None else None
        return {
            "status": "analytical_estimate", "model": "single_cluster_serial_v1",
            "scope": "single_cluster", "policy": "serial_stage_sum_no_overlap",
            "total_cycles": total, "chip_cycles": None, "time_ns": time_ns,
            "throughput_gops": useful_ops / time_ns if time_ns is not None else None,
            "stages_cycles": dict(self.stages), "scheduled_tiles": self.tiles,
            "load_batches": self.load_batches, "switch_batches": self.switch_batches,
            "parameters": self.parameters,
            "clock": dict(self.spec.native_reference_spec.parameters["clock_ghz"]),
            "assumptions": [
                "A/B already reside in local MEM; activation delivery is hidden within ACC row service.",
                "Tiles and stages are serialized; no LW/ACC overlap, bank/stream contention or routing simulation.",
                "Full-row LW slots may idle; no claim that a partial Cluster physically requires all 160 cycles.",
                "Writer emits valid INT32 columns; partial sums remain in MEM until their next K segment.",
                "Physical Macro timing remains uncalibrated; this upper-level estimator does not fill or use those null parameters.",
                "No host transfer, packing, memory-capacity scheduling, energy, area or calibrated cross-architecture speedup.",
            ],
        }


class TSPCIMSimulator:
    def __init__(self, tsp_spec, *, data_format="INT8"):
        if data_format != "INT8":
            raise ValueError("TSP-CIM Macro supports INT8 operands and INT32 results only")
        self.tsp_spec = tsp_spec

    def simulate(self, M, K, N, loop_order="KNM", *, mode="count", scope="single_cluster",
                 hemisphere=0, plane=0, cluster=0, tile_m=None):
        if mode not in {"count", "timing"}:
            raise ValueError("mode must be count or timing")
        if scope != "single_cluster":
            raise ValueError("Only single_cluster execution is modeled; chip inventory is not chip execution")
        for name, value in (("M", M), ("K", K), ("N", N)):
            positive_integer(value, name)
        spec = self.tsp_spec
        domain = spec.validate_domain(hemisphere, plane, cluster)
        # 一个 domain 跨所有 SuperLane；同一行的多个 Core/Macro 负责不同 N 列。
        tile = TSPCIMTilingEngine(spec).derive_tile_sizes(M, K, N, tile_m)
        timing = _SerialTaskTiming(spec) if mode == "timing" else None
        result = count_macro_tiles(
            spec.macro_spec, M, K, N, tile["m"], tile["k"], tile["n"],
            spec.plane.domain_macros, loop_order, "single_cluster",
            on_tile=timing.account_tile if timing is not None else None)
        counts = result["counts"]
        n_tiles = (N + tile["n"] - 1) // tile["n"]
        k_tiles = (K + tile["k"] - 1) // tile["k"]
        # 广播入口的一份输入会被多个 Macro 捕获；入口 byte 与捕获 byte 属于不同位置。
        # 下面的加法数是完成 GEMM 必须执行的逻辑工作，不表示归约网络已被调度。
        counts.update(
            logical_activation_broadcast_bytes=M * ((K + spec.core.k - 1) // spec.core.k) * spec.core.k * n_tiles,
            logical_inter_superlane_adds=counts["cross_macro_k_adds"],
            logical_inter_k_tile_adds=counts["cross_tile_k_adds"],
            logical_cluster_partial_bytes=M * N * 4 * k_tiles,
        )
        for name, unit, action_scope in (
            ("logical_activation_broadcast_bytes", "byte", "cluster_domain_input"),
            ("logical_inter_superlane_adds", "add", "same_N_shard_across_SuperLanes"),
            ("logical_inter_k_tile_adds", "add", "outside_cluster_K_segments"),
            ("logical_cluster_partial_bytes", "byte", "valid_cluster_K_segment_results"),
        ):
            result["action_counts"][name] = {"count": counts[name], "unit": unit, "scope": action_scope,
                "basis": "logical requirement only; not physical traffic, switching or scheduled execution"}
        result.update(schema_version=3, model="tsp_cim_cluster_reference", reference_id=spec.reference_id,
                      macro_reference_id=spec.macro_spec.reference_id,
                      native_reference_id=spec.native_reference_spec.reference_id)
        result["boundary"].update(selected_domain=domain, end="macro_local_INT32_partials_and_logical_reduction_obligations",
                                  simulated_cluster_domains=1, chip_execution_included=False,
                                  mem_allocation_included=False, nim_conversion_included=False)
        result["policy"].update(upper_model_status="reviewed_core_cluster_plane_count",
                                row_grouping=tile["status"],
                                physical_slot_order="SuperLane * 20 + Core * 4 + Macro",
                                domain_context_policy="independent Macro tags; calls drained before switching; no rolling IW")
        result["topology"] = spec.hierarchy()
        # 在单域执行报告中附带整片规格，供解释硬件层级；不会据此缩放本次动作数。
        result["hardware_inventory"]["chip"] = spec.inventory()
        result["hardware_inventory"]["allocated_cores"] = spec.plane.superlanes * spec.cluster.cores
        result["hardware_inventory"]["allocated_cluster_domains"] = 1
        result["hardware_derived"] = {"macro": spec.macro_spec.derived(), **spec.derived()}
        result["parameter_provenance"] = spec.provenance()
        result["missing_parameters"] = spec.missing_parameters()
        result["timing"]["status"] = "not_computed"
        if timing is not None:
            result["mode"] = "timing"
            result["timing"] = timing.report(counts["useful_ops"])
            result["boundary"].update(start="packed_INT8_A_B_in_local_MEM", end="logical_INT32_C_in_local_MEM",
                                      transport_included=True, cross_macro_reduction_timed=True,
                                      timing_resolution="serialized_task_stages_with_explicit_assumptions")
            result["policy"]["upper_model_status"] = "simplified_single_cluster_timing"
        # §5.1 按每个 K/N 任务的 M×II 计算 ACC 占用，域内 Macro 是并行工作的。
        # 尾块仍占完整 II；切分 M 行或更换循环顺序不减少总行数，也不在这里加入换权重等待。
        # 占用仍作为独立分项报告；只有 timing 模式会将装载、排空和写回等阶段计入总时间。
        ii = spec.macro_spec.values["input_ii_cycles"]
        result["timing"]["compute_occupancy"] = {
            "value": M * k_tiles * n_tiles * ii, "unit": "cycle", "scope": "single_cluster",
            "status": "documented_schedule", "kn_tiles": k_tiles * n_tiles,
            "input_ii_cycles": ii,
            "formula": "M * ceil(K / domain_k) * ceil(N / domain_n) * input_ii_cycles",
            "source": "https://github.com/pku-pacific-lab-team/LPU-HUAWEI-2012/blob/"
                      "c641844c1d213d02a7d961d889838cae9b9dc8aa/docs/legacy/patent-idea-review.md#L744-L750",
            "source_version": "git-blob acf4798e3d0a5ed57bec576849d6c3ffb4679ab7; section 5.1",
            "conditions": "Sum of INT8 Cluster ACC occupancy only; tails retain a full II. "
                          "Excludes LW/IW waits, first-result/drain latency, reduction and writer service; "
                          "not elapsed time or an additional Macro energy action.",
        }
        result["limitations"] = [
            "One Cluster domain runs all K/N tiles serially; no multi-domain or chip task scheduler.",
            "Four N-sharded Macros per Core; K reduction links equal N shards across SuperLanes only.",
            "Macro cycles and captures are local action sums, not elapsed time or MEM traffic.",
            "Logical reduction/broadcast obligations are not a cycle or route-level implementation.",
            "No MEM placement, stream contention, collector/NIM scheduling, rolling IW or overlapping LW/ACC.",
            "NIM counts and service targets describe hardware; FP/BFP conversion and SFU execution are not modeled.",
            "Known MEM plus Macro SRAM is a subtotal; additional buffers and physical total remain unknown.",
        ]
        return result

    def compare_mxm(self, M, K, N, loop_order="KNM", workload=None):
        # 使用与独立 tsp.py 相同的原生内核。缺少可比系统时间时只返回参考计数。
        from ...utils.tsp_reference import compare_native
        if workload is None:
            workload = parse_workloads({"gemm": {"M": M, "K": K, "N": N}, "data_format": "INT8"})[0]
        if workload.shape != (M, K, N) or workload.data_format != "INT8":
            raise ValueError("comparison workload must match the INT8 CIM task")
        result = compare_native(self.tsp_spec.native_reference_spec, workload, loop_order=loop_order)
        result["reasons"].append("CIM single Cluster domain and native TSP single hemisphere have different execution scopes.")
        return result

    def sweep_m(self, m_values, K, N, loop_order="KNM", **policy):
        return [{"M": m, "result": self.simulate(m, K, N, loop_order, **policy),
                 "comparison": self.compare_mxm(m, K, N, loop_order)} for m in m_values]

    def compare_loop_orders(self, M, K, N, **policy):
        return {order: self.simulate(M, K, N, order, **policy) for order in LOOP_ORDERS}
