"""
TSP baseline simulator.
"""
import math
from typing import Any, Dict, List, Optional, Tuple

from ..hardware.dataformat import FORMAT_SPECS, get_format_overheads
from ..hardware.tsp import TSPSpec
from .tsp_tiling import TSPTilingEngine

LOOP_ORDERS = {
    "KNM": "K-outer: maximize weight reuse on native MXM",
    "KMN": "K-outer, M-middle: weight-stationary with M priority",
    "MKN": "M-outer: activation-stationary baseline order",
    "NKM": "N-outer: output-column priority",
}


def _generate_tile_iter(
    loop_order: str,
    M: int,
    K: int,
    N: int,
    tile_m: int,
    tile_k: int,
    tile_n: int,
) -> List[Tuple[int, int, int, int, int, int]]:
    m_tiles = [(ms, min(tile_m, M - ms)) for ms in range(0, M, tile_m)]
    k_tiles = [(ks, min(tile_k, K - ks)) for ks in range(0, K, tile_k)]
    n_tiles = [(ns, min(tile_n, N - ns)) for ns in range(0, N, tile_n)]

    result = []
    if loop_order == "KNM":
        for ks, ka in k_tiles:
            for ns, na in n_tiles:
                for ms, ma in m_tiles:
                    result.append((ms, ks, ns, ma, ka, na))
    elif loop_order == "KMN":
        for ks, ka in k_tiles:
            for ms, ma in m_tiles:
                for ns, na in n_tiles:
                    result.append((ms, ks, ns, ma, ka, na))
    elif loop_order == "MKN":
        for ms, ma in m_tiles:
            for ks, ka in k_tiles:
                for ns, na in n_tiles:
                    result.append((ms, ks, ns, ma, ka, na))
    elif loop_order == "NKM":
        for ns, na in n_tiles:
            for ks, ka in k_tiles:
                for ms, ma in m_tiles:
                    result.append((ms, ks, ns, ma, ka, na))
    else:
        raise ValueError(f"Unknown loop order: {loop_order}")
    return result


class TSPSimulator:
    """Native TSP simulator with explicit MEM/SXM/MXM/VXM staging."""

    def __init__(
        self,
        tsp_spec: TSPSpec,
        energy_params: Dict[str, float],
        data_format: str = "INT8",
        custom_overheads: Optional[Dict[str, float]] = None,
        verbose: bool = True,
    ):
        if data_format not in FORMAT_SPECS:
            raise ValueError(f"Unsupported format: {data_format}")

        self.tsp_spec = tsp_spec
        self.energy_params = energy_params
        self.data_format = data_format
        self.verbose = verbose

        fmt_spec = FORMAT_SPECS[data_format]
        self.storage_bytes = fmt_spec.storage_bytes
        self.overheads = get_format_overheads(data_format, custom_overheads)
        self.tiling_engine = TSPTilingEngine(tsp_spec, self.storage_bytes)

    def _energy_param(self, name: str, default: float = 0.0) -> float:
        return self.energy_params.get(name, default)

    def _mxm_op_energy(self) -> float:
        if self.data_format in {"FP16", "BF16", "INT16"}:
            return self._energy_param("mxm_fp16_op_power")
        return self._energy_param("mxm_int8_op_power")

    def _fixed_energy(self, wall_clock_cycles: int) -> Dict[str, float]:
        spec = self.tsp_spec
        clock_domains = spec.total_mxm_planes_chip
        control_domains = spec.total_mxm_planes_chip + spec.n_hemispheres
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
            "mxm_array_logic_leak": (
                wall_clock_cycles *
                spec.total_mxm_planes_chip *
                self._energy_param("mxm_array_logic_leak_power")
            ),
            "mem_slice_ctrl": (
                wall_clock_cycles *
                spec.mem_n_slices *
                self._energy_param("mem_slice_ctrl_power")
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
            "clock_domains": (
                wall_clock_cycles *
                clock_domains *
                self._energy_param("clock_domain_power")
            ),
            "control_backbone": (
                wall_clock_cycles *
                self._energy_param("control_backbone_power")
            ),
            "control_domains": (
                wall_clock_cycles *
                control_domains *
                self._energy_param("control_domain_power")
            ),
        }

    def simulate(
        self,
        M: int,
        K: int,
        N: int,
        loop_order: str = "KNM",
    ) -> Dict[str, Any]:
        spec = self.tsp_spec
        tile_info = self.tiling_engine.derive_tile_sizes(M, K, N, self.data_format)
        tile_m = tile_info["m"]
        tile_k = tile_info["k"]
        tile_n = tile_info["n"]
        topology = tile_info["topology"]
        mem_alloc = tile_info["mem_alloc"]

        stagger = spec.stagger_cycles
        hops_act_to_mxm = int(spec.avg_hops_act_to_mxm(mem_alloc["act_slices"]))
        hops_wt_to_mxm = int(spec.avg_hops_wt_to_mxm(mem_alloc["wt_slices"]))
        hops_mxm_to_vxm = spec.hops_mxm_to_vxm()
        hops_vxm_to_result = int(
            spec.avg_hops_vxm_to_result_mem(
                mem_alloc["result_slices"],
                mem_alloc["result_start"],
            )
        )

        t_avail = {
            "MEM_Read": 0,
            "Weight_Install": 0,
            "MXM_Compute": 0,
            "MXM_to_VXM": 0,
            "VXM_Post": 0,
            "MEM_Write": 0,
        }
        weight_ready_time = 0

        energies_pj = {
            "sram_read_act": 0.0,
            "sram_read_weight": 0.0,
            "sram_psum_buffer_read": 0.0,
            "sram_psum_buffer_write": 0.0,
            "weight_install": 0.0,
            "mxm_compute": 0.0,
            "vxm_post_proc": 0.0,
            "vxm_operand_io": 0.0,
            "sram_write_result": 0.0,
            "stream_hop_act": 0.0,
            "stream_hop_weight": 0.0,
            "stream_hop_result": 0.0,
            "sxm_permute": 0.0,
            "global_sram_standby": 0.0,
            "local_sram_standby": 0.0,
            "mxm_array_logic_leak": 0.0,
            "mem_slice_ctrl": 0.0,
            "sxm_logic_leak": 0.0,
            "vxm_logic_leak": 0.0,
            "clock_backbone": 0.0,
            "clock_domains": 0.0,
            "control_backbone": 0.0,
            "control_domains": 0.0,
        }

        latency_detail = {
            "mem_read_issue_total": 0,
            "transit_act_total": 0,
            "weight_install_total": 0,
            "transit_wt_total": 0,
            "mxm_compute_total": 0,
            "mxm_pipeline_fill_total": 0,
            "transit_mxm_vxm_total": 0,
            "vxm_compute_total": 0,
            "transit_vxm_mem_total": 0,
            "stagger_overhead_total": 0,
        }

        tile_iter = _generate_tile_iter(loop_order, M, K, N, tile_m, tile_k, tile_n)
        stream_bw = spec.stream_bw_per_direction_bytes_per_cycle
        weight_install_bw = spec.weight_install_bytes_per_cycle
        peak_ops_per_cycle = spec.peak_ops_per_cycle_per_hemisphere(self.data_format)
        vxm_alus = spec.vxm_alus_per_hemisphere
        psum_bytes = 4
        last_weight_tag = None
        psum_k_progress: Dict[Tuple[int, int], int] = {}
        weight_install_count = 0

        if self.verbose:
            print(f"  Loop order: {loop_order} ({LOOP_ORDERS[loop_order]})")
            print(f"  Peak INT8: {spec.peak_tops('INT8'):.2f} TOPS")
            print(f"  Peak {self.data_format}: {spec.peak_tops(self.data_format):.2f} TOPS")

        for m_start, k_start, n_start, m_actual, k_actual, n_actual in tile_iter:
            supercell_usage = spec.describe_supercell_usage(
                k_actual=k_actual,
                n_actual=n_actual,
                data_format=self.data_format,
                hemisphere=0,
            )
            c_tag = (m_start, n_start)
            act_bytes = m_actual * k_actual * self.storage_bytes
            weight_bytes = k_actual * n_actual * self.storage_bytes
            output_elements = m_actual * n_actual
            output_bytes = output_elements * psum_bytes

            # Stage 1: MEM -> MXM activation stream.
            read_issue_cycles = max(1, math.ceil(act_bytes / stream_bw))
            read_duration = read_issue_cycles + 1 + hops_act_to_mxm + stagger
            start_read = t_avail["MEM_Read"]
            done_read = start_read + read_duration
            t_avail["MEM_Read"] = done_read

            latency_detail["mem_read_issue_total"] += read_issue_cycles
            latency_detail["transit_act_total"] += hops_act_to_mxm
            latency_detail["stagger_overhead_total"] += stagger

            energies_pj["sram_read_act"] += (
                act_bytes * self._energy_param("sram_read_power")
            )
            energies_pj["stream_hop_act"] += (
                act_bytes * hops_act_to_mxm *
                self._energy_param("stream_hop_external_power")
            )

            # Stage 2: MEM -> MXM weight load/install.
            weight_tag = (k_start, n_start)
            if weight_tag != last_weight_tag:
                install_issue_cycles = max(1, math.ceil(weight_bytes / weight_install_bw))
                install_duration = install_issue_cycles + 1 + hops_wt_to_mxm + stagger
                start_install = t_avail["Weight_Install"]
                weight_ready_time = start_install + install_duration
                t_avail["Weight_Install"] = weight_ready_time
                weight_install_count += 1

                latency_detail["weight_install_total"] += install_issue_cycles + 1
                latency_detail["transit_wt_total"] += hops_wt_to_mxm
                latency_detail["stagger_overhead_total"] += stagger

                energies_pj["sram_read_weight"] += (
                    weight_bytes * self._energy_param("sram_read_power")
                )
                energies_pj["weight_install"] += (
                    weight_bytes * self._energy_param("weight_install_power")
                )
                energies_pj["stream_hop_weight"] += (
                    weight_bytes * hops_wt_to_mxm *
                    self._energy_param("stream_hop_external_power")
                )
            last_weight_tag = weight_tag

            # Stage 3: MXM compute.
            active_cols_total = max(1, int(supercell_usage["active_supercell_cols_total"]))
            plane_batches = max(
                1,
                math.ceil(
                    active_cols_total /
                    (
                        spec.logical_mxm_planes_per_hemisphere(self.data_format) *
                        spec.supercells_n_per_plane
                    )
                ),
            )
            row_issue_cycles = plane_batches * spec.plane_tandem_factor(self.data_format)
            compute_cycles = max(1, m_actual * row_issue_cycles)
            pipeline_fill = max(0, int(supercell_usage["reduction_chain_depth"]) - 1)
            compute_duration = compute_cycles + max(stagger, pipeline_fill)
            start_compute = max(done_read, weight_ready_time, t_avail["MXM_Compute"])
            done_compute = start_compute + compute_duration
            t_avail["MXM_Compute"] = done_compute

            latency_detail["mxm_compute_total"] += compute_cycles
            latency_detail["mxm_pipeline_fill_total"] += pipeline_fill
            latency_detail["stagger_overhead_total"] += stagger
            tile_ops = m_actual * k_actual * n_actual * 2
            energies_pj["mxm_compute"] += tile_ops * self._mxm_op_energy()

            # Stage 4: MXM -> VXM stream transit and SXM permutation.
            result_reduce_hops = max(0, int(supercell_usage["reduction_chain_depth"]) - 1)
            mxm_vxm_duration = hops_mxm_to_vxm + result_reduce_hops + stagger
            start_transit = max(done_compute, t_avail["MXM_to_VXM"])
            done_transit = start_transit + mxm_vxm_duration
            t_avail["MXM_to_VXM"] = done_transit

            latency_detail["transit_mxm_vxm_total"] += hops_mxm_to_vxm + result_reduce_hops
            latency_detail["stagger_overhead_total"] += stagger
            energies_pj["stream_hop_result"] += (
                output_bytes * (hops_mxm_to_vxm + result_reduce_hops) *
                self._energy_param("stream_hop_external_power")
            )
            energies_pj["sxm_permute"] += (
                output_bytes * max(1, result_reduce_hops) * self._energy_param("sxm_permute_power")
            )

            # Stage 5: VXM post-processing and K accumulation.
            previous_k = psum_k_progress.get(c_tag, 0)
            is_first_k = previous_k == 0
            psum_k_progress[c_tag] = previous_k + k_actual
            vxm_ops = output_elements if is_first_k else int(output_elements * 1.5)
            vxm_cycles = max(
                1,
                math.ceil(
                    vxm_ops /
                    max(1.0, vxm_alus * self.overheads["post_throughput_scale"])
                ),
            )
            vxm_duration = vxm_cycles + stagger
            start_vxm = max(done_transit, t_avail["VXM_Post"])
            done_vxm = start_vxm + vxm_duration
            t_avail["VXM_Post"] = done_vxm

            latency_detail["vxm_compute_total"] += vxm_cycles
            latency_detail["stagger_overhead_total"] += stagger
            energies_pj["vxm_post_proc"] += (
                vxm_ops *
                self._energy_param("vxm_op_power") *
                self.overheads["post_energy_scale"]
            )
            energies_pj["vxm_operand_io"] += (
                (vxm_ops * 2 * psum_bytes * self._energy_param("sram_read_power")) +
                (vxm_ops * psum_bytes * self._energy_param("sram_write_power"))
            )

            if not is_first_k:
                prior_psum_bytes = output_bytes
                energies_pj["sram_psum_buffer_read"] += (
                    prior_psum_bytes * self._energy_param("sram_read_power")
                )
                if psum_k_progress[c_tag] < K:
                    energies_pj["sram_psum_buffer_write"] += (
                        prior_psum_bytes * self._energy_param("sram_write_power")
                    )

            # Stage 6: VXM -> MEM final writeback.
            if psum_k_progress[c_tag] >= K:
                write_issue_cycles = max(1, math.ceil(output_bytes / stream_bw))
                write_duration = write_issue_cycles + 1 + hops_vxm_to_result + stagger
                start_write = max(done_vxm, t_avail["MEM_Write"])
                done_write = start_write + write_duration
                t_avail["MEM_Write"] = done_write

                latency_detail["transit_vxm_mem_total"] += hops_vxm_to_result
                latency_detail["stagger_overhead_total"] += stagger
                energies_pj["sram_write_result"] += (
                    output_bytes * self._energy_param("sram_write_power")
                )
                energies_pj["stream_hop_result"] += (
                    output_bytes * hops_vxm_to_result *
                    self._energy_param("stream_hop_external_power")
                )

        total_latency_single_hem = max(t_avail.values())
        effective_latency = math.ceil(total_latency_single_hem / spec.n_hemispheres)
        wall_clock_cycles = effective_latency

        for key, value in self._fixed_energy(wall_clock_cycles).items():
            energies_pj[key] += value

        total_energy_pj = sum(energies_pj.values())
        total_ops = M * K * N * 2

        throughput_gops = (
            total_ops / effective_latency * spec.clock_freq_ghz
            if effective_latency > 0 else 0.0
        )

        return {
            "Format": self.data_format,
            "loop_order": loop_order,
            "Tile_M": tile_m,
            "Tile_K": tile_k,
            "Tile_N": tile_n,
            "weight_install_count": weight_install_count,
            "total_tiles": len(tile_iter),
            "latencies": t_avail,
            "latency_detail": latency_detail,
            "energies_pj": energies_pj,
            "total_latency_cycles": effective_latency,
            "wall_clock_cycles": wall_clock_cycles,
            "single_hem_latency_cycles": total_latency_single_hem,
            "total_energy_pj": total_energy_pj,
            "throughput_gops": throughput_gops,
            "energy_efficiency_tops_per_watt": (
                (total_ops / 1e12) / (total_energy_pj / 1e12)
                if total_energy_pj > 0 else 0.0
            ),
            "bottleneck": max(t_avail, key=t_avail.get),
            "spatial_flow": {
                "hops_act_to_mxm": hops_act_to_mxm,
                "hops_wt_to_mxm": hops_wt_to_mxm,
                "hops_mxm_to_vxm": hops_mxm_to_vxm,
                "hops_vxm_to_result": hops_vxm_to_result,
                "stagger": stagger,
                "n_hemispheres": spec.n_hemispheres,
                "logical_mxm_planes_per_hemisphere": spec.logical_mxm_planes_per_hemisphere(self.data_format),
                "physical_mxm_planes_per_hemisphere": spec.n_mxm_planes_per_hemisphere,
                "plane_tandem_factor": spec.plane_tandem_factor(self.data_format),
                "supercell_dim": spec.supercell_dim,
                "supercells_k_per_plane": spec.supercells_k_per_plane,
                "supercells_n_per_plane": spec.supercells_n_per_plane,
            },
            "topology": topology,
            "mem_alloc": mem_alloc,
            "sram_summary": spec.sram_summary(),
            "area_inventory": spec.area_inventory(),
            "peak_tops": {
                "INT8": spec.peak_tops("INT8"),
                "BF16": spec.peak_tops("BF16"),
                "FP16": spec.peak_tops("FP16"),
            },
        }
