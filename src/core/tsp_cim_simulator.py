"""
TSP-CIM Core Simulator (V2) — 精细化东西南北向数据流建模

Groq TSP 架构 + CIM Macro 替换 MXM 的仿真器，含：
- 物理芯片布局建模 (VXM | MEM[0..43] | SXM | CIM per hemisphere)
- Stream register hop 延迟 (1 hop = 1 cycle)
- 20-cycle staggered SIMD 执行
- East/West 半球并行
- MEM 切片分配策略 (activation 近 VXM, weight 近 CIM)
- CIM→VXM chaining 优化
- 可配置循环顺序 (KNM/MKN/KMN)
- MXM baseline 对比
"""
import math
from typing import Dict, Any, Optional, List, Tuple

from ..hardware.tsp_cim import TSPCIMSpec
from ..hardware.dataformat import FORMAT_SPECS, get_format_overheads
from .tsp_cim_tiling import TSPCIMTilingEngine

# 支持的循环顺序
LOOP_ORDERS = {
    'KNM': 'K-outer: 最大化权重复用 (CIM optimal)',
    'KMN': 'K-outer, M-middle: 权重复用 + M优先',
    'MKN': 'M-outer: 原始顺序 (MXM friendly)',
    'NKM': 'N-outer: N维度优先',
}


def _generate_tile_iter(
    loop_order: str,
    M: int, K: int, N: int,
    tile_m: int, tile_k: int, tile_n: int,
) -> List[Tuple[int, int, int, int, int, int]]:
    """根据循环顺序生成 tile 迭代序列
    Returns: [(m_start, k_start, n_start, m_actual, k_actual, n_actual), ...]
    """
    m_tiles = [(ms, min(tile_m, M - ms)) for ms in range(0, M, tile_m)]
    k_tiles = [(ks, min(tile_k, K - ks)) for ks in range(0, K, tile_k)]
    n_tiles = [(ns, min(tile_n, N - ns)) for ns in range(0, N, tile_n)]

    result = []
    if loop_order == 'KNM':
        for ks, ka in k_tiles:
            for ns, na in n_tiles:
                for ms, ma in m_tiles:
                    result.append((ms, ks, ns, ma, ka, na))
    elif loop_order == 'KMN':
        for ks, ka in k_tiles:
            for ms, ma in m_tiles:
                for ns, na in n_tiles:
                    result.append((ms, ks, ns, ma, ka, na))
    elif loop_order == 'MKN':
        for ms, ma in m_tiles:
            for ks, ka in k_tiles:
                for ns, na in n_tiles:
                    result.append((ms, ks, ns, ma, ka, na))
    elif loop_order == 'NKM':
        for ns, na in n_tiles:
            for ks, ka in k_tiles:
                for ms, ma in m_tiles:
                    result.append((ms, ks, ns, ma, ka, na))
    else:
        raise ValueError(f"Unknown loop_order: {loop_order}. Use: {list(LOOP_ORDERS.keys())}")
    return result


class TSPCIMSimulator:
    """
    TSP-CIM 仿真器 V2 — 精细化数据流建模

    物理布局 (每半球，从中心向外):
        VXM | SRF | MEM[0] | ... | MEM[43] | SRF | SXM | SRF | CIM | SRF | C2C

    流水线阶段:
        1. MEM_Read:      从 SRAM 读 activation → stream outward → CIM
        2. Weight_Prog:   从 SRAM 读 weight → stream outward → CIM 编程
        3. CIM_Compute:   CIM 本地读出 + 计算
        4. CIM→VXM Transit: 结果 stream inward → VXM (chaining)
        5. VXM_Post:      累加 / 后处理
        6. MEM_Write:     结果 stream outward → SRAM 写回

    循环顺序:
        KNM (默认): K-outer → 权重编程仅在 K/N tile 变化时触发，M 维度完全复用
        MKN: M-outer → 每轮 M 都需重新编程所有 K tiles（权重复用差）
    """

    def __init__(
        self,
        tsp_spec: TSPCIMSpec,
        energy_params: Dict[str, float],
        data_format: str = 'INT8',
        custom_overheads: Optional[Dict[str, float]] = None,
        verbose: bool = True,
    ):
        self.tsp_spec = tsp_spec
        self.energy_params = energy_params
        self.data_format = data_format
        self.verbose = verbose

        # Data format
        if data_format not in FORMAT_SPECS:
            raise ValueError(f"Unsupported format: {data_format}")
        fmt_spec = FORMAT_SPECS[data_format]
        self.storage_bytes = fmt_spec.storage_bytes
        self.compute_bits = 8
        self.effective_compute_bits = fmt_spec.effective_compute_bits
        self.activity_factor = fmt_spec.get_activity_factor(self.compute_bits)
        self.overheads = get_format_overheads(data_format, custom_overheads)

        # Tiling engine
        self.tiling_engine = TSPCIMTilingEngine(tsp_spec, self.storage_bytes)

        # CIM macro shortcuts
        self.macro = tsp_spec.macro_spec
        self.t_local_readout = 1
        self.bytes_per_local_readout = (
            self.macro.n_subarrays_per_bank *
            self.macro.block_n_dim *
            self.macro.n_cols_per_bank
        ) / 8

    def _energy_param(
        self,
        name: str,
        legacy_name: Optional[str] = None,
        default: float = 0.0,
    ) -> float:
        if name in self.energy_params:
            return self.energy_params[name]
        if legacy_name and legacy_name in self.energy_params:
            return self.energy_params[legacy_name]
        return default

    def _common_fixed_energy(self, wall_clock_cycles: int) -> Dict[str, float]:
        spec = self.tsp_spec
        return {
            "global_sram_standby": (
                wall_clock_cycles *
                spec.global_mem_sram_mib *
                self._energy_param("global_sram_standby_power_per_mib")
            ),
            "local_sram_standby": (
                wall_clock_cycles *
                spec.reserved_local_sram_mib *
                self._energy_param("local_sram_standby_power_per_mib")
            ),
            "sxm_logic_leak": (
                wall_clock_cycles *
                spec.n_hemispheres *
                self._energy_param("sxm_logic_leak_power")
            ),
            "vxm_logic_leak": (
                wall_clock_cycles *
                spec.n_hemispheres *
                self._energy_param("vxm_logic_leak_power")
            ),
            "clock_backbone": (
                wall_clock_cycles *
                self._energy_param("clock_backbone_power")
            ),
            "control_backbone": (
                wall_clock_cycles *
                self._energy_param("control_backbone_power")
            ),
        }

    def _cim_fixed_energy(self, wall_clock_cycles: int) -> Dict[str, float]:
        spec = self.tsp_spec
        return {
            "macro_logic_leak": (
                wall_clock_cycles *
                spec.total_macros_chip *
                self._energy_param("macro_logic_leak_power")
            ),
        }

    def _fabric_fixed_energy(
        self,
        wall_clock_cycles: int,
        spatial_flow: Dict[str, int],
    ) -> Dict[str, float]:
        spec = self.tsp_spec
        active_macros = max(1, int(spatial_flow.get("active_macros", 1)))
        active_superlanes = max(1, int(spatial_flow.get("active_superlanes", 1)))
        active_k_parallel = max(1, int(spatial_flow.get("active_k_parallel_groups", 1)))
        active_n_parallel = max(1, int(spatial_flow.get("active_n_parallel_groups", 1)))
        inactive_factor = self._energy_param(
            "inactive_retention_factor",
            default=0.35,
        )

        active_fabric_domains = max(
            math.ceil(active_macros / 24),
            active_superlanes + math.ceil(active_k_parallel / 4) + math.ceil(active_n_parallel / 4),
        )
        effective_logic_domains = max(
            active_fabric_domains,
            math.ceil(spec.fabric_logic_domains * inactive_factor),
        )
        effective_clock_domains = max(
            active_fabric_domains,
            math.ceil(spec.clock_domains_total * inactive_factor),
        )
        effective_control_domains = max(
            active_fabric_domains,
            math.ceil(spec.control_domains_total * inactive_factor),
        )

        return {
            "fabric_retention": (
                wall_clock_cycles *
                spec.fabric_retention_domains *
                self._energy_param("fabric_retention_power_per_domain")
            ),
            "fabric_logic_leak": (
                wall_clock_cycles *
                effective_logic_domains *
                self._energy_param("fabric_logic_power_per_domain")
            ),
            "clock_domains": (
                wall_clock_cycles *
                effective_clock_domains *
                self._energy_param("clock_domain_power")
            ),
            "control_domains": (
                wall_clock_cycles *
                effective_control_domains *
                self._energy_param("control_domain_power")
            ),
        }

    def simulate(
        self,
        M: int,
        K: int,
        N: int,
        loop_order: str = 'KNM',
    ) -> Dict[str, Any]:
        spec = self.tsp_spec
        tile_info = self.tiling_engine.derive_tile_sizes(M, K, N)
        tile_m = tile_info['m']
        tile_k = tile_info['k']
        tile_n = tile_info['n']
        nominal_macro_topology = tile_info['macro_topology']
        mem_alloc = tile_info['mem_alloc']

        STAGGER = spec.stagger_cycles  # 19

        # ---- Transit delays ----
        hops_act_to_cim = int(spec.avg_hops_act_to_cim(mem_alloc['act_slices']))
        hops_wt_to_cim = int(spec.avg_hops_wt_to_cim(mem_alloc['wt_slices']))
        hops_cim_to_vxm = spec.hops_cim_to_vxm()
        hops_vxm_to_result = int(spec.avg_hops_vxm_to_result_mem(
            mem_alloc['result_slices'], mem_alloc['result_start']))

        # ---- Pipeline timestamps ----
        t_avail = {
            "MEM_Read":      0,
            "Weight_Prog":   0,
            "CIM_Compute":   0,
            "CIM_to_VXM":    0,
            "VXM_Post":      0,
            "MEM_Write":     0,
        }

        # ---- Energy counters (pJ) ----
        energies_pj = {
            "sram_read_act":      0.0,
            "sram_read_weight":   0.0,
            "sram_psum_buffer_read": 0.0,
            "sram_psum_buffer_write": 0.0,
            "weight_prog":        0.0,
            "cim_local_readout":  0.0,
            "cim_compute":        0.0,
            "pre_process":        0.0,
            "vxm_post_proc":      0.0,
            "vxm_operand_io":     0.0,
            "sram_write_result":  0.0,
            "stream_hop_act_external":   0.0,
            "stream_hop_act_internal":   0.0,
            "stream_hop_wt_external":    0.0,
            "stream_hop_wt_internal":    0.0,
            "stream_hop_result_external": 0.0,
            "stream_hop_result_internal": 0.0,
            "sxm_broadcast":     0.0,
            "reduce_collect":    0.0,
            "reduce_buffer":     0.0,
            "global_sram_standby": 0.0,
            "local_sram_standby": 0.0,
            "macro_logic_leak":  0.0,
            "fabric_retention":  0.0,
            "fabric_logic_leak": 0.0,
            "sxm_logic_leak":    0.0,
            "vxm_logic_leak":    0.0,
            "clock_backbone":    0.0,
            "clock_domains":     0.0,
            "control_backbone":  0.0,
            "control_domains":   0.0,
        }

        # ---- Latency detail tracking ----
        latency_detail = {
            "stagger_overhead_total": 0,
            "transit_act_total": 0,
            "macro_fanout_act_total": 0,
            "transit_wt_total": 0,
            "macro_fanout_wt_total": 0,
            "transit_cim_vxm_total": 0,
            "macro_return_total": 0,
            "transit_vxm_mem_total": 0,
            "cim_compute_total": 0,
            "vxm_compute_total": 0,
            "mem_read_issue_total": 0,
            "weight_prog_total": 0,
            "macro_act_dispatch_total": 0,
            "macro_wt_dispatch_total": 0,
            "macro_reduce_occupancy_total": 0,
        }

        # ---- Constants ----
        wprog_bpc = spec.weight_prog_bytes_per_cycle
        stream_width = spec.stream_width_bytes
        n_streams = spec.n_streams_per_direction
        vxm_ops_per_cycle = spec.total_vxm_alus
        total_macros_chip = spec.total_macros_chip
        PSUM_BYTES = 4
        stream_hop_external_pj = self._energy_param(
            "stream_hop_external_power", legacy_name="stream_hop_power"
        )
        stream_hop_internal_pj = self._energy_param(
            "stream_hop_internal_power", legacy_name="stream_hop_power"
        )
        sxm_broadcast_pj = self._energy_param("sxm_broadcast_power")
        reduce_collect_pj = self._energy_param("reduce_collect_power")
        reduce_buffer_rw_pj = self._energy_param("reduce_buffer_rw_power")

        # ---- Tracking ----
        last_weight_tag = None
        psum_k_progress = {}
        weight_prog_count = 0
        observed_spatial_flow = {
            "act_fanout_internal_hops": 0,
            "result_return_internal_hops": 0,
            "active_k_groups": 0,
            "active_n_groups": 0,
            "active_k_parallel_groups": 0,
            "active_n_parallel_groups": 0,
            "serial_k_depth": 0,
            "serial_n_depth": 0,
            "active_superlanes": 0,
            "active_macros": 0,
        }

        # ---- Generate tile iteration order ----
        tile_iter = _generate_tile_iter(
            loop_order, M, K, N, tile_m, tile_k, tile_n
        )

        if self.verbose:
            n_k = math.ceil(K / tile_k)
            n_n = math.ceil(N / tile_n)
            n_m = math.ceil(M / tile_m)
            print(f"  Loop order: {loop_order} ({LOOP_ORDERS[loop_order]})")
            print(f"  Tiles: {n_m}(M) x {n_k}(K) x {n_n}(N) = {len(tile_iter)} total")
            print(f"  Weight prog events (optimal): {n_k * n_n} "
                  f"(worst: {len(tile_iter)})")

        # ---- Main tile loop ----
        for m_start, k_start, n_start, m_actual, k_actual, n_actual in tile_iter:
            macro_usage = spec.describe_macro_usage(k_actual, n_actual, hemisphere=0)
            act_route = spec.estimate_activation_route(
                mem_alloc['act_start'], macro_usage, m_actual * k_actual * self.storage_bytes
            )
            wt_route = spec.estimate_weight_route(
                mem_alloc['wt_start'], macro_usage, k_actual, self.storage_bytes
            )
            result_route = spec.estimate_result_route(
                macro_usage, m_actual, PSUM_BYTES
            )
            observed_spatial_flow["act_fanout_internal_hops"] = max(
                observed_spatial_flow["act_fanout_internal_hops"],
                int(act_route["max_internal_hops"]),
            )
            observed_spatial_flow["result_return_internal_hops"] = max(
                observed_spatial_flow["result_return_internal_hops"],
                int(result_route["max_internal_hops"]),
            )
            observed_spatial_flow["active_k_groups"] = max(
                observed_spatial_flow["active_k_groups"],
                int(macro_usage["active_k_groups"]),
            )
            observed_spatial_flow["active_n_groups"] = max(
                observed_spatial_flow["active_n_groups"],
                int(macro_usage.get("active_n_groups", 0)),
            )
            observed_spatial_flow["active_k_parallel_groups"] = max(
                observed_spatial_flow["active_k_parallel_groups"],
                int(macro_usage.get("active_k_parallel_groups", 0)),
            )
            observed_spatial_flow["active_n_parallel_groups"] = max(
                observed_spatial_flow["active_n_parallel_groups"],
                int(macro_usage.get("active_n_parallel_groups", 0)),
            )
            observed_spatial_flow["serial_k_depth"] = max(
                observed_spatial_flow["serial_k_depth"],
                int(macro_usage.get("serial_k_depth", 1)),
            )
            observed_spatial_flow["serial_n_depth"] = max(
                observed_spatial_flow["serial_n_depth"],
                int(macro_usage.get("serial_n_depth", 1)),
            )
            observed_spatial_flow["active_superlanes"] = max(
                observed_spatial_flow["active_superlanes"],
                int(macro_usage["active_superlanes"]),
            )
            observed_spatial_flow["active_macros"] = max(
                observed_spatial_flow["active_macros"],
                int(macro_usage["active_macros"]),
            )

            act_dispatch_cycles = max(
                1,
                math.ceil(act_route["fabric_injection_bytes"] / spec.stream_bw_per_direction_bytes_per_cycle),
            ) if act_route["fabric_injection_bytes"] > 0 else 0

            weight_tag = (k_start, n_start)
            weight_changed = (weight_tag != last_weight_tag)

            # ==== Stage 1: MEM Read Activation ====
            act_bytes = m_actual * k_actual * self.storage_bytes
            n_act_vectors = math.ceil(act_bytes / stream_width)
            read_issue_cycles = math.ceil(n_act_vectors / n_streams)
            d_func_read = 1
            dur_read = (
                read_issue_cycles + d_func_read + STAGGER +
                int(act_route["max_total_hops"]) +
                act_dispatch_cycles
            )

            start_read = t_avail["MEM_Read"]
            t_avail["MEM_Read"] = start_read + dur_read

            latency_detail["mem_read_issue_total"] += read_issue_cycles + d_func_read
            latency_detail["stagger_overhead_total"] += STAGGER
            latency_detail["transit_act_total"] += int(act_route["external_hops"])
            latency_detail["macro_fanout_act_total"] += int(act_route["max_internal_hops"])
            latency_detail["macro_act_dispatch_total"] += act_dispatch_cycles

            energies_pj["sram_read_act"] += (
                act_bytes * self.energy_params["sram_read_power"]
            )
            energies_pj["stream_hop_act_external"] += (
                act_route["external_byte_hops"] * stream_hop_external_pj
            )
            energies_pj["stream_hop_act_internal"] += (
                act_route["internal_byte_hops"] * stream_hop_internal_pj
            )
            energies_pj["sxm_broadcast"] += (
                act_route["fabric_injection_bytes"] * sxm_broadcast_pj
            )

            # ==== Stage 2: Weight Programming ====
            if weight_changed:
                weight_bytes = k_actual * n_actual * self.storage_bytes
                n_wt_vectors = math.ceil(weight_bytes / stream_width)
                wt_issue_cycles = math.ceil(n_wt_vectors / n_streams)
                wt_transit = int(wt_route["max_total_hops"])
                wt_dispatch_cycles = max(
                    1,
                    math.ceil(wt_route["fabric_injection_bytes"] / spec.stream_bw_per_direction_bytes_per_cycle),
                ) if wt_route["fabric_injection_bytes"] > 0 else 0
                # Per-macro programming: all active macros write in parallel.
                weight_bytes_per_macro = wt_route["max_macro_bytes"]
                prog_cycles = math.ceil(weight_bytes_per_macro / wprog_bpc)

                d_func_wt_read = 1
                dur_prog = (wt_issue_cycles + d_func_wt_read +
                           STAGGER + wt_transit + prog_cycles + wt_dispatch_cycles)

                start_prog = max(
                    t_avail["MEM_Read"],
                    t_avail["Weight_Prog"]
                )
                t_avail["Weight_Prog"] = start_prog + dur_prog

                latency_detail["weight_prog_total"] += dur_prog
                latency_detail["stagger_overhead_total"] += STAGGER
                latency_detail["transit_wt_total"] += int(wt_route["external_hops"])
                latency_detail["macro_fanout_wt_total"] += int(wt_route["max_internal_hops"])
                latency_detail["macro_wt_dispatch_total"] += wt_dispatch_cycles

                energies_pj["sram_read_weight"] += (
                    weight_bytes * self.energy_params["sram_read_power"]
                )
                energies_pj["stream_hop_wt_external"] += (
                    wt_route["external_byte_hops"] * stream_hop_external_pj
                )
                energies_pj["stream_hop_wt_internal"] += (
                    wt_route["internal_byte_hops"] * stream_hop_internal_pj
                )
                energies_pj["weight_prog"] += (
                    weight_bytes * self.energy_params["weight_prog_power"]
                )
                energies_pj["sxm_broadcast"] += (
                    wt_route["fabric_injection_bytes"] * sxm_broadcast_pj
                )

                last_weight_tag = weight_tag
                weight_prog_count += 1

            # ==== Stage 3: CIM Compute ====
            num_switches = self.macro.n_loc_per_subarray
            num_chunks = math.ceil(
                self.effective_compute_bits / self.compute_bits
            )
            k_stage_count = max(1, int(macro_usage.get("serial_k_depth", 1)))
            dur_per_chunk = num_switches * (
                self.t_local_readout + m_actual * self.compute_bits
            )
            dur_cim_core = dur_per_chunk * num_chunks * k_stage_count
            dur_cim = dur_cim_core + max(STAGGER, k_stage_count - 1)

            start_cim = max(
                t_avail["MEM_Read"],
                t_avail["Weight_Prog"],
                t_avail["CIM_Compute"]
            )
            t_avail["CIM_Compute"] = start_cim + dur_cim

            latency_detail["cim_compute_total"] += dur_cim_core
            latency_detail["stagger_overhead_total"] += max(STAGGER, k_stage_count - 1)

            # Energy: CIM local readout
            active_macros_chip = max(1, int(macro_usage["active_macros"])) * spec.n_hemispheres
            base_readout_energy = (
                num_switches *
                self.bytes_per_local_readout *
                self.energy_params["cim_local_read_power"] *
                active_macros_chip
            )
            energies_pj["cim_local_readout"] += (
                base_readout_energy * self.activity_factor * num_chunks
            )

            # Energy: CIM compute
            total_ops = m_actual * k_actual * n_actual
            energy_scale = self.effective_compute_bits / 8.0
            energies_pj["cim_compute"] += (
                total_ops * energy_scale *
                self.energy_params["tpu_flop_power"] *
                self.overheads['macro_energy_scale'] *
                self.activity_factor
            )

            # Energy: pre-processing
            pre_bits = act_bytes * 8
            energies_pj["pre_process"] += (
                pre_bits * self.overheads['pre_energy_pj_per_bit']
            )

            # ==== Stage 4: CIM→VXM Transit (Chaining) ====
            reduce_parallel_groups = max(
                1,
                int(macro_usage.get("active_k_parallel_groups", 1)),
            )
            result_parallel_groups = max(
                1,
                int(macro_usage.get("active_n_parallel_groups", 1)),
            )
            reduce_occupancy_cycles = 0
            if result_route["reduction_bytes"] > 0:
                reduce_occupancy_cycles += math.ceil(
                    result_route["reduction_bytes"] /
                    (spec.stream_bw_per_direction_bytes_per_cycle * reduce_parallel_groups)
                )
            if result_route["total_result_bytes"] > 0:
                reduce_occupancy_cycles += math.ceil(
                    result_route["total_result_bytes"] /
                    (spec.stream_bw_per_direction_bytes_per_cycle * result_parallel_groups)
                )
            dur_transit = int(result_route["max_total_hops"]) + reduce_occupancy_cycles
            start_transit = t_avail["CIM_Compute"]
            t_avail["CIM_to_VXM"] = start_transit + dur_transit

            latency_detail["transit_cim_vxm_total"] += int(result_route["external_hops"])
            latency_detail["macro_return_total"] += int(result_route["max_internal_hops"])
            latency_detail["macro_reduce_occupancy_total"] += reduce_occupancy_cycles

            energies_pj["stream_hop_result_external"] += (
                result_route["external_byte_hops"] * stream_hop_external_pj
            )
            energies_pj["stream_hop_result_internal"] += (
                result_route["internal_byte_hops"] * stream_hop_internal_pj
            )
            energies_pj["reduce_collect"] += (
                result_route["reduction_bytes"] * reduce_collect_pj
            )
            energies_pj["reduce_buffer"] += (
                result_route["reduction_bytes"] * 2 * reduce_buffer_rw_pj
            )
            energies_pj["sram_psum_buffer_write"] += (
                result_route["reduction_bytes"] * self.energy_params["sram_write_power"]
            )
            energies_pj["sram_psum_buffer_read"] += (
                result_route["reduction_bytes"] * self.energy_params["sram_read_power"]
            )

            # ==== Stage 5: VXM Post-processing ====
            output_elements = m_actual * n_actual
            c_tag = (m_start, n_start)

            if c_tag not in psum_k_progress:
                psum_k_progress[c_tag] = 0
            psum_k_progress[c_tag] += k_actual
            is_first_k = (psum_k_progress[c_tag] == k_actual)
            active_k_groups = max(1, int(macro_usage["active_k_groups"]))
            serial_k_depth = max(1, int(macro_usage.get("serial_k_depth", 1)))

            vxm_ops = output_elements
            # Only serial K rounds require extra VXM accumulation; intra-round K reduction stays in the CIM fabric.
            if serial_k_depth > 1:
                vxm_ops += output_elements * (serial_k_depth - 1)
            if not is_first_k:
                vxm_ops = int(vxm_ops * 1.5)

            eff_vxm_throughput = (
                vxm_ops_per_cycle *
                self.overheads['post_throughput_scale']
            )
            d_func_vxm = max(1, math.ceil(vxm_ops / eff_vxm_throughput))
            dur_vxm = d_func_vxm + STAGGER

            start_vxm = max(
                t_avail["CIM_to_VXM"],
                t_avail["VXM_Post"]
            )
            t_avail["VXM_Post"] = start_vxm + dur_vxm

            latency_detail["vxm_compute_total"] += d_func_vxm
            latency_detail["stagger_overhead_total"] += STAGGER

            energies_pj["vxm_post_proc"] += (
                vxm_ops *
                self.energy_params["vect_flop_power"] *
                self.overheads['post_energy_scale']
            )
            energies_pj["vxm_operand_io"] += (
                (vxm_ops * 2 * PSUM_BYTES * self.energy_params["sram_read_power"]) +
                (vxm_ops * PSUM_BYTES * self.energy_params["sram_write_power"])
            )

            if not is_first_k:
                prior_psum_bytes = output_elements * PSUM_BYTES
                energies_pj["sram_psum_buffer_read"] += (
                    prior_psum_bytes * self.energy_params["sram_read_power"]
                )
                if not (psum_k_progress[c_tag] >= K):
                    energies_pj["sram_psum_buffer_write"] += (
                        prior_psum_bytes * self.energy_params["sram_write_power"]
                    )

            # ==== Stage 6: MEM Write ====
            is_k_complete = (psum_k_progress[c_tag] >= K)
            if is_k_complete:
                result_bytes = m_actual * n_actual * PSUM_BYTES
                n_result_vectors = math.ceil(result_bytes / stream_width)
                write_issue_cycles = math.ceil(n_result_vectors / n_streams)

                d_func_write = 1
                dur_write = (write_issue_cycles + d_func_write +
                            STAGGER + hops_vxm_to_result)

                start_write = max(
                    t_avail["VXM_Post"],
                    t_avail["MEM_Write"]
                )
                t_avail["MEM_Write"] = start_write + dur_write

                latency_detail["stagger_overhead_total"] += STAGGER
                latency_detail["transit_vxm_mem_total"] += hops_vxm_to_result

                energies_pj["sram_write_result"] += (
                    result_bytes * self.energy_params["sram_write_power"]
                )
                energies_pj["stream_hop_result_external"] += (
                    result_bytes * hops_vxm_to_result *
                    stream_hop_external_pj
                )

        # ==== Results ====
        total_latency = max(t_avail.values())
        hemisphere_speedup = spec.n_hemispheres
        effective_latency = math.ceil(total_latency / hemisphere_speedup)

        wall_clock_cycles = effective_latency
        for key, value in {
            **self._common_fixed_energy(wall_clock_cycles),
            **self._cim_fixed_energy(wall_clock_cycles),
            **self._fabric_fixed_energy(wall_clock_cycles, observed_spatial_flow),
        }.items():
            energies_pj[key] += value

        total_energy_pj = sum(energies_pj.values())
        total_mac_ops = M * K * N * 2

        throughput = (
            (total_mac_ops / effective_latency * spec.clock_freq_ghz)
            if effective_latency > 0 else 0
        )

        stream_hop_total = (
            energies_pj["stream_hop_act_external"] +
            energies_pj["stream_hop_act_internal"] +
            energies_pj["stream_hop_wt_external"] +
            energies_pj["stream_hop_wt_internal"] +
            energies_pj["stream_hop_result_external"] +
            energies_pj["stream_hop_result_internal"]
        )

        # MXM baseline (uses MKN order — MXM reinstalls weights fast anyway)
        mxm_baseline = self._calc_mxm_baseline(
            M, K, N, tile_m, tile_k, tile_n, hops_act_to_cim,
            hops_wt_to_cim, hops_cim_to_vxm, hops_vxm_to_result,
            STAGGER, n_streams, stream_width
        )

        return {
            "Format": self.data_format,
            "loop_order": loop_order,
            "Tile_M": tile_m,
            "Tile_K": tile_k,
            "Tile_N": tile_n,
            "weight_prog_count": weight_prog_count,
            "total_tiles": len(tile_iter),
            "latencies": t_avail,
            "latency_detail": latency_detail,
            "energies_pj": energies_pj,
            "total_latency_cycles": effective_latency,
            "wall_clock_cycles": wall_clock_cycles,
            "single_hem_latency_cycles": total_latency,
            "total_energy_pj": total_energy_pj,
            "throughput_gops": throughput,
            "energy_efficiency_tops_per_watt": (
                (total_mac_ops / 1e12) / (total_energy_pj / 1e12)
                if total_energy_pj > 0 else 0
            ),
            "bottleneck": max(t_avail, key=t_avail.get),
            "spatial_flow": {
                "hops_act_to_cim": hops_act_to_cim,
                "hops_wt_to_cim": hops_wt_to_cim,
                "hops_cim_to_vxm": hops_cim_to_vxm,
                "hops_vxm_to_result": hops_vxm_to_result,
                "act_fanout_internal_hops": observed_spatial_flow["act_fanout_internal_hops"],
                "result_return_internal_hops": observed_spatial_flow["result_return_internal_hops"],
                "active_k_groups": observed_spatial_flow["active_k_groups"],
                "active_n_groups": observed_spatial_flow["active_n_groups"],
                "active_k_parallel_groups": observed_spatial_flow["active_k_parallel_groups"],
                "active_n_parallel_groups": observed_spatial_flow["active_n_parallel_groups"],
                "serial_k_depth": observed_spatial_flow["serial_k_depth"],
                "serial_n_depth": observed_spatial_flow["serial_n_depth"],
                "active_superlanes": observed_spatial_flow["active_superlanes"],
                "active_macros": observed_spatial_flow["active_macros"],
                "macro_k_groups_per_superlane": nominal_macro_topology.get("macro_k_groups_per_superlane", 1),
                "macro_n_groups_per_superlane": nominal_macro_topology.get("macro_n_groups_per_superlane", 1),
                "macro_sheet_stack_k": nominal_macro_topology.get("macro_sheet_stack_k", 1),
                "macro_sheet_stack_n": nominal_macro_topology.get("macro_sheet_stack_n", 1),
                "stagger": STAGGER,
                "n_hemispheres": spec.n_hemispheres,
            },
            "macro_topology": nominal_macro_topology,
            "mem_alloc": mem_alloc,
            "sram_summary": spec.sram_summary(),
            "area_inventory": spec.area_inventory(),
            "stream_hop_energy_pj": stream_hop_total,
            "stream_hop_energy_pct": (
                stream_hop_total / total_energy_pj * 100
                if total_energy_pj > 0 else 0
            ),
            "mxm_baseline": mxm_baseline,
            "tsp_spec_summary": {
                "total_macros": spec.total_macros * spec.n_hemispheres,
                "macros_per_hemisphere": spec.macros_per_hemisphere,
                "macros_per_superlane": spec.n_macros_per_superlane,
                "macro_k_groups_per_superlane": spec.macro_k_groups_per_superlane,
                "macro_n_groups_per_superlane": spec.macro_n_groups_per_superlane,
                "active_k_parallel_groups": nominal_macro_topology.get("active_k_parallel_groups", 1),
                "active_n_parallel_groups": nominal_macro_topology.get("active_n_parallel_groups", 1),
                "serial_k_depth": nominal_macro_topology.get("serial_k_depth", 1),
                "serial_n_depth": nominal_macro_topology.get("serial_n_depth", 1),
                "total_lanes": spec.total_lanes,
                "lanes_per_superlane": spec.n_lanes_per_superlane,
                "total_vxm_alus": spec.total_vxm_alus,
                "mem_mb": spec.mem_total_bytes // (1024 * 1024),
                "physical_sram_mb": spec.physical_total_sram_bytes // (1024 * 1024),
                "global_sram_mb": spec.global_mem_sram_bytes // (1024 * 1024),
                "local_sram_mb": spec.reserved_local_sram_bytes // (1024 * 1024),
                "macro_block_k": spec.macro_spec.block_k_dim,
                "macro_block_n": spec.macro_spec.block_n_dim,
                "macro_supercell_k_span": spec.macro_supercell_k_span,
                "macro_supercell_n_span": spec.macro_supercell_n_span,
                "macro_rows_per_plane": spec.macro_rows_per_plane,
                "macro_cols_per_plane": spec.macro_cols_per_plane,
                "macro_sheets_per_hemisphere": spec.macro_sheets_per_hemisphere,
                "macro_sheet_stack_k": spec.macro_sheet_stack_k,
                "macro_sheet_stack_n": spec.macro_sheet_stack_n,
                "tile_k_capacity": spec.tile_k_capacity,
                "tile_n_capacity": spec.tile_n_capacity,
                "macro_weight_ratio_vs_supercell": spec.macro_weight_ratio_vs_supercell,
                "macro_peak_ratio_vs_supercell_int8": spec.macro_peak_ratio_vs_supercell_int8,
                "macros_per_supercell_peak_equivalent_int8": spec.macros_per_supercell_peak_equivalent_int8,
                "peak_equivalent_supercells_per_superlane_int8": spec.peak_equivalent_supercells_per_superlane_int8,
                "peak_equivalent_supercells_chip_int8": spec.peak_equivalent_supercells_chip_int8,
                "effective_n_columns_per_lane": spec.effective_n_columns_per_lane,
            },
        }

    def _calc_mxm_baseline(
        self, M, K, N, tile_m, tile_k, tile_n,
        hops_act_to_mxm, hops_wt_to_mxm, hops_mxm_to_vxm, hops_vxm_to_result,
        stagger, n_streams, stream_width
    ) -> Dict[str, Any]:
        """
        MXM baseline: 4 independent 320×320 planes per hemisphere
        Weight install: <40 cycles for 409,600 weights
        Uses KNM loop order for fair comparison (MXM also benefits from weight reuse)
        """
        spec = self.tsp_spec

        MXM_PLANES = 4
        MXM_DIM = 320
        MXM_WEIGHT_INSTALL_CYCLES = 40

        total_mxm_latency = 0
        last_wt_tag = None
        psum_k_progress = {}
        stream_hop_external_pj = self._energy_param(
            "stream_hop_external_power", legacy_name="stream_hop_power"
        )
        mxm_mac_energy_per_op = 0.5  # pJ
        mxm_energies = {
            "sram_read_act": 0.0,
            "sram_read_weight": 0.0,
            "sram_psum_buffer_read": 0.0,
            "sram_psum_buffer_write": 0.0,
            "weight_install": 0.0,
            "mxm_compute": 0.0,
            "vxm_post_proc": 0.0,
            "vxm_operand_io": 0.0,
            "sram_write_result": 0.0,
            "stream_hop": 0.0,
            "global_sram_standby": 0.0,
            "local_sram_standby": 0.0,
            "sxm_logic_leak": 0.0,
            "vxm_logic_leak": 0.0,
            "clock_backbone": 0.0,
            "control_backbone": 0.0,
        }

        tile_iter = _generate_tile_iter(
            'KNM', M, K, N, tile_m, tile_k, tile_n
        )

        for m_start, k_start, n_start, m_actual, k_actual, n_actual in tile_iter:
            wt_tag = (k_start, n_start)
            wt_changed = (wt_tag != last_wt_tag)
            c_tag = (m_start, n_start)

            # MEM Read
            act_bytes = m_actual * k_actual * self.storage_bytes
            n_vectors = math.ceil(act_bytes / stream_width)
            read_cycles = math.ceil(n_vectors / n_streams) + 1 + stagger + hops_act_to_mxm
            mxm_energies["sram_read_act"] += (
                act_bytes * self.energy_params["sram_read_power"]
            )
            mxm_energies["stream_hop"] += (
                act_bytes * hops_act_to_mxm * stream_hop_external_pj
            )

            # Weight install
            prog_cycles = MXM_WEIGHT_INSTALL_CYCLES if wt_changed else 0
            if wt_changed:
                weight_bytes = k_actual * n_actual * self.storage_bytes
                mxm_energies["sram_read_weight"] += (
                    weight_bytes * self.energy_params["sram_read_power"]
                )
                mxm_energies["weight_install"] += (
                    weight_bytes * self.energy_params["sram_write_power"]
                )
                mxm_energies["stream_hop"] += (
                    weight_bytes * hops_wt_to_mxm * stream_hop_external_pj
                )

            # MXM compute
            k_cols = math.ceil(k_actual / MXM_DIM)
            n_cols = math.ceil(n_actual / MXM_DIM)
            compute_cycles = m_actual * k_cols * math.ceil(n_cols / MXM_PLANES)
            compute_cycles += stagger
            tile_ops = m_actual * k_actual * n_actual * 2
            mxm_energies["mxm_compute"] += tile_ops * mxm_mac_energy_per_op

            # Transit + VXM
            transit = hops_mxm_to_vxm + stagger
            output_elements = m_actual * n_actual
            if c_tag not in psum_k_progress:
                psum_k_progress[c_tag] = 0
            psum_k_progress[c_tag] += k_actual
            is_first_k = (psum_k_progress[c_tag] == k_actual)
            vxm_ops = output_elements if is_first_k else int(output_elements * 1.5)
            vxm_cycles = math.ceil(vxm_ops / spec.total_vxm_alus) + stagger
            mxm_energies["stream_hop"] += (
                output_elements * 4 * hops_mxm_to_vxm * stream_hop_external_pj
            )
            mxm_energies["vxm_post_proc"] += (
                vxm_ops *
                self.energy_params["vect_flop_power"] *
                self.overheads['post_energy_scale']
            )
            mxm_energies["vxm_operand_io"] += (
                (vxm_ops * 2 * 4 * self.energy_params["sram_read_power"]) +
                (vxm_ops * 4 * self.energy_params["sram_write_power"])
            )
            if not is_first_k:
                prior_psum_bytes = output_elements * 4
                mxm_energies["sram_psum_buffer_read"] += (
                    prior_psum_bytes * self.energy_params["sram_read_power"]
                )
                if not (psum_k_progress[c_tag] >= K):
                    mxm_energies["sram_psum_buffer_write"] += (
                        prior_psum_bytes * self.energy_params["sram_write_power"]
                    )

            tile_latency = read_cycles + prog_cycles + compute_cycles + transit + vxm_cycles
            total_mxm_latency += tile_latency

            if psum_k_progress[c_tag] >= K:
                result_bytes = output_elements * 4
                mxm_energies["sram_write_result"] += (
                    result_bytes * self.energy_params["sram_write_power"]
                )
                mxm_energies["stream_hop"] += (
                    result_bytes * hops_vxm_to_result * stream_hop_external_pj
                )
            last_wt_tag = wt_tag

        effective_mxm = math.ceil(total_mxm_latency / spec.n_hemispheres)
        for key, value in self._common_fixed_energy(effective_mxm).items():
            mxm_energies[key] += value

        total_ops = M * K * N * 2
        mxm_total_energy = sum(mxm_energies.values())

        mxm_throughput = (
            (total_ops / effective_mxm * spec.clock_freq_ghz)
            if effective_mxm > 0 else 0
        )

        return {
            "total_latency_cycles": effective_mxm,
            "single_hem_latency_cycles": total_mxm_latency,
            "energies_pj": mxm_energies,
            "total_energy_pj": mxm_total_energy,
            "throughput_gops": mxm_throughput,
            "energy_efficiency_tops_per_watt": (
                (total_ops / 1e12) / (mxm_total_energy / 1e12)
                if mxm_total_energy > 0 else 0
            ),
            "scope_note": (
                "Includes SRAM/stream/VXM/common fixed overheads; "
                "does not yet model MXM-array-specific leakage."
            ),
        }

    def sweep_m(
        self,
        m_values: List[int],
        K: int,
        N: int,
        loop_order: str = 'KNM',
    ) -> List[Dict[str, Any]]:
        """扫描不同 M 值，分析 CIM vs MXM 收益变化"""
        results = []
        for m_val in m_values:
            old_verbose = self.verbose
            self.verbose = False
            res = self.simulate(m_val, K, N, loop_order=loop_order)
            self.verbose = old_verbose

            cim_lat = res['total_latency_cycles']
            mxm_lat = res['mxm_baseline']['total_latency_cycles']
            cim_energy = res['total_energy_pj']
            mxm_energy = res['mxm_baseline']['total_energy_pj']

            speedup = mxm_lat / cim_lat if cim_lat > 0 else 0
            energy_ratio = mxm_energy / cim_energy if cim_energy > 0 else 0

            results.append({
                'M': m_val,
                'K': K,
                'N': N,
                'cim_latency': cim_lat,
                'mxm_latency': mxm_lat,
                'speedup': speedup,
                'cim_energy_pj': cim_energy,
                'mxm_energy_pj': mxm_energy,
                'energy_ratio': energy_ratio,
                'cim_advantage': speedup > 1.0,
                'bottleneck': res['bottleneck'],
                'weight_prog_count': res['weight_prog_count'],
            })
        return results

    def compare_loop_orders(
        self,
        M: int, K: int, N: int,
        orders: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """对比不同循环顺序的性能差异"""
        if orders is None:
            orders = list(LOOP_ORDERS.keys())
        results = []
        for order in orders:
            old_verbose = self.verbose
            self.verbose = False
            res = self.simulate(M, K, N, loop_order=order)
            self.verbose = old_verbose
            results.append({
                'loop_order': order,
                'description': LOOP_ORDERS[order],
                'latency': res['total_latency_cycles'],
                'energy_pj': res['total_energy_pj'],
                'throughput_gops': res['throughput_gops'],
                'tops_per_watt': res['energy_efficiency_tops_per_watt'],
                'weight_prog_count': res['weight_prog_count'],
                'total_tiles': res['total_tiles'],
                'bottleneck': res['bottleneck'],
            })
        return results
