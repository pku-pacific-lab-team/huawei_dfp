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
            "weight_prog":        0.0,
            "cim_local_readout":  0.0,
            "cim_compute":        0.0,
            "pre_process":        0.0,
            "vxm_post_proc":      0.0,
            "sram_write_result":  0.0,
            "stream_hop_act":     0.0,
            "stream_hop_wt":      0.0,
            "stream_hop_result":  0.0,
        }

        # ---- Latency detail tracking ----
        latency_detail = {
            "stagger_overhead_total": 0,
            "transit_act_total": 0,
            "transit_wt_total": 0,
            "transit_cim_vxm_total": 0,
            "transit_vxm_mem_total": 0,
            "cim_compute_total": 0,
            "vxm_compute_total": 0,
            "mem_read_issue_total": 0,
            "weight_prog_total": 0,
        }

        # ---- Constants ----
        wprog_bpc = spec.weight_prog_bytes_per_cycle
        stream_width = spec.stream_width_bytes
        n_streams = spec.n_streams_per_direction
        vxm_ops_per_cycle = spec.total_vxm_alus
        total_macros_chip = spec.total_macros * spec.n_hemispheres
        PSUM_BYTES = 4

        # ---- Tracking ----
        last_weight_tag = None
        psum_k_progress = {}
        weight_prog_count = 0

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

            weight_tag = (k_start, n_start)
            weight_changed = (weight_tag != last_weight_tag)

            # ==== Stage 1: MEM Read Activation ====
            act_bytes = m_actual * k_actual * self.storage_bytes
            n_act_vectors = math.ceil(act_bytes / stream_width)
            read_issue_cycles = math.ceil(n_act_vectors / n_streams)
            d_func_read = 1
            dur_read = read_issue_cycles + d_func_read + STAGGER + hops_act_to_cim

            start_read = t_avail["MEM_Read"]
            t_avail["MEM_Read"] = start_read + dur_read

            latency_detail["mem_read_issue_total"] += read_issue_cycles + d_func_read
            latency_detail["stagger_overhead_total"] += STAGGER
            latency_detail["transit_act_total"] += hops_act_to_cim

            energies_pj["sram_read_act"] += (
                act_bytes * self.energy_params["sram_read_power"]
            )
            energies_pj["stream_hop_act"] += (
                act_bytes * hops_act_to_cim *
                self.energy_params["stream_hop_power"]
            )

            # ==== Stage 2: Weight Programming ====
            if weight_changed:
                weight_bytes = k_actual * n_actual * self.storage_bytes
                n_wt_vectors = math.ceil(weight_bytes / stream_width)
                wt_issue_cycles = math.ceil(n_wt_vectors / n_streams)
                wt_transit = hops_wt_to_cim
                # Per-macro programming: all macros write in parallel
                # Each macro receives k × block_n weights, total tile split across macros
                weight_bytes_per_macro = (
                    k_actual * self.macro.block_n_dim * self.storage_bytes
                )
                prog_cycles = math.ceil(weight_bytes_per_macro / wprog_bpc)

                d_func_wt_read = 1
                dur_prog = (wt_issue_cycles + d_func_wt_read +
                           STAGGER + wt_transit + prog_cycles)

                start_prog = max(
                    t_avail["MEM_Read"],
                    t_avail["Weight_Prog"]
                )
                t_avail["Weight_Prog"] = start_prog + dur_prog

                latency_detail["weight_prog_total"] += dur_prog
                latency_detail["stagger_overhead_total"] += STAGGER
                latency_detail["transit_wt_total"] += wt_transit

                energies_pj["sram_read_weight"] += (
                    weight_bytes * self.energy_params["sram_read_power"]
                )
                energies_pj["stream_hop_wt"] += (
                    weight_bytes * wt_transit *
                    self.energy_params["stream_hop_power"]
                )
                energies_pj["weight_prog"] += (
                    weight_bytes * self.energy_params["weight_prog_power"]
                )

                last_weight_tag = weight_tag
                weight_prog_count += 1

            # ==== Stage 3: CIM Compute ====
            num_switches = self.macro.n_loc_per_subarray
            num_chunks = math.ceil(
                self.effective_compute_bits / self.compute_bits
            )
            dur_per_chunk = num_switches * (
                self.t_local_readout + m_actual * self.compute_bits
            )
            dur_cim_core = dur_per_chunk * num_chunks
            dur_cim = dur_cim_core + STAGGER

            start_cim = max(
                t_avail["MEM_Read"],
                t_avail["Weight_Prog"],
                t_avail["CIM_Compute"]
            )
            t_avail["CIM_Compute"] = start_cim + dur_cim

            latency_detail["cim_compute_total"] += dur_cim_core
            latency_detail["stagger_overhead_total"] += STAGGER

            # Energy: CIM local readout
            base_readout_energy = (
                num_switches *
                self.bytes_per_local_readout *
                self.energy_params["cim_local_read_power"] *
                total_macros_chip
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
            dur_transit = hops_cim_to_vxm
            start_transit = t_avail["CIM_Compute"]
            t_avail["CIM_to_VXM"] = start_transit + dur_transit

            latency_detail["transit_cim_vxm_total"] += dur_transit

            result_elements = m_actual * n_actual
            result_stream_bytes = result_elements * PSUM_BYTES
            energies_pj["stream_hop_result"] += (
                result_stream_bytes * hops_cim_to_vxm *
                self.energy_params["stream_hop_power"]
            )

            # ==== Stage 5: VXM Post-processing ====
            output_elements = m_actual * n_actual
            c_tag = (m_start, n_start)

            if c_tag not in psum_k_progress:
                psum_k_progress[c_tag] = 0
            psum_k_progress[c_tag] += k_actual
            is_first_k = (psum_k_progress[c_tag] == k_actual)

            vxm_ops = output_elements
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
                energies_pj["stream_hop_result"] += (
                    result_bytes * hops_vxm_to_result *
                    self.energy_params["stream_hop_power"]
                )

        # ==== Results ====
        total_latency = max(t_avail.values())
        hemisphere_speedup = spec.n_hemispheres
        effective_latency = math.ceil(total_latency / hemisphere_speedup)

        total_energy_pj = sum(energies_pj.values())
        total_mac_ops = M * K * N * 2

        throughput = (
            (total_mac_ops / effective_latency * spec.clock_freq_ghz)
            if effective_latency > 0 else 0
        )

        stream_hop_total = (energies_pj["stream_hop_act"] +
                           energies_pj["stream_hop_wt"] +
                           energies_pj["stream_hop_result"])

        # MXM baseline (uses MKN order — MXM reinstalls weights fast anyway)
        mxm_baseline = self._calc_mxm_baseline(
            M, K, N, tile_m, tile_k, tile_n, hops_act_to_cim,
            hops_cim_to_vxm, hops_vxm_to_result, STAGGER, n_streams, stream_width
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
                "stagger": STAGGER,
                "n_hemispheres": spec.n_hemispheres,
            },
            "mem_alloc": mem_alloc,
            "stream_hop_energy_pj": stream_hop_total,
            "stream_hop_energy_pct": (
                stream_hop_total / total_energy_pj * 100
                if total_energy_pj > 0 else 0
            ),
            "mxm_baseline": mxm_baseline,
            "tsp_spec_summary": {
                "total_macros": spec.total_macros * spec.n_hemispheres,
                "total_lanes": spec.total_lanes,
                "total_vxm_alus": spec.total_vxm_alus,
                "mem_mb": spec.mem_total_bytes // (1024 * 1024),
            },
        }

    def _calc_mxm_baseline(
        self, M, K, N, tile_m, tile_k, tile_n,
        hops_act_to_mxm, hops_mxm_to_vxm, hops_vxm_to_result,
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

        tile_iter = _generate_tile_iter(
            'KNM', M, K, N, tile_m, tile_k, tile_n
        )

        for _, k_start, n_start, m_actual, k_actual, n_actual in tile_iter:
            wt_tag = (k_start, n_start)
            wt_changed = (wt_tag != last_wt_tag)

            # MEM Read
            act_bytes = m_actual * k_actual * self.storage_bytes
            n_vectors = math.ceil(act_bytes / stream_width)
            read_cycles = math.ceil(n_vectors / n_streams) + 1 + stagger + hops_act_to_mxm

            # Weight install
            prog_cycles = MXM_WEIGHT_INSTALL_CYCLES if wt_changed else 0

            # MXM compute
            k_cols = math.ceil(k_actual / MXM_DIM)
            n_cols = math.ceil(n_actual / MXM_DIM)
            compute_cycles = m_actual * k_cols * math.ceil(n_cols / MXM_PLANES)
            compute_cycles += stagger

            # Transit + VXM
            transit = hops_mxm_to_vxm + stagger
            vxm_cycles = math.ceil(m_actual * n_actual / spec.total_vxm_alus) + stagger

            tile_latency = read_cycles + prog_cycles + compute_cycles + transit + vxm_cycles
            total_mxm_latency += tile_latency
            last_wt_tag = wt_tag

        effective_mxm = math.ceil(total_mxm_latency / spec.n_hemispheres)

        total_ops = M * K * N * 2
        mxm_mac_energy_per_op = 0.5  # pJ
        mxm_total_energy = total_ops * mxm_mac_energy_per_op

        mxm_throughput = (
            (total_ops / effective_mxm * spec.clock_freq_ghz)
            if effective_mxm > 0 else 0
        )

        return {
            "total_latency_cycles": effective_mxm,
            "single_hem_latency_cycles": total_mxm_latency,
            "total_energy_pj": mxm_total_energy,
            "throughput_gops": mxm_throughput,
            "energy_efficiency_tops_per_watt": (
                (total_ops / 1e12) / (mxm_total_energy / 1e12)
                if mxm_total_energy > 0 else 0
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
