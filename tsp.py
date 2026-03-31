#!/usr/bin/env python3
"""
Native TSP baseline entry point.
"""
import argparse
from pathlib import Path

from src.core.area_model import AreaModel
from src.core.tsp_simulator import LOOP_ORDERS, TSPSimulator
from src.utils.tsp_config import TSPConfigLoader


def parse_args():
    parser = argparse.ArgumentParser(
        description="Native Groq TSP baseline simulator"
    )
    parser.add_argument(
        "--hardware", type=str, default="tsp.json",
        help="Hardware config file (default: tsp.json)"
    )
    parser.add_argument(
        "--energy", type=str, default="7nm_tsp.json",
        help="Energy config file (default: 7nm_tsp.json)"
    )
    parser.add_argument(
        "--area", type=str, default="7nm_tsp.json",
        help="Area config file (default: 7nm_tsp.json)"
    )
    parser.add_argument(
        "--workload", type=str, default="default_gemm.json",
        help="Workload config file (default: default_gemm.json)"
    )
    parser.add_argument(
        "--config-dir", type=str, default=None,
        help="Config directory (default: ./configs)"
    )
    parser.add_argument(
        "--formats", type=str, nargs="+",
        default=["BF16"],
        help="Data formats to test (default: BF16)"
    )
    parser.add_argument(
        "--loop-order", type=str, default="KNM",
        choices=list(LOOP_ORDERS.keys()),
        help="Tile loop order (default: KNM)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose output"
    )
    return parser.parse_args()


def print_spatial_flow(results: dict) -> None:
    sf = results.get("spatial_flow", {})
    ma = results.get("mem_alloc", {})
    topo = results.get("topology", {})
    if not sf:
        return

    print("\n[Spatial Data Flow Analysis]")
    print("  Chip Layout: VXM | MEM[0..43] | SXM | MXM  (x2 hemispheres)")
    print(
        f"  Act MEM slices:    [{ma.get('act_start', 0)}.."
        f"{ma.get('act_start', 0) + ma.get('act_slices', 1) - 1}]  (near VXM)"
    )
    print(
        f"  Weight MEM slices: [{ma.get('wt_start', 28)}.."
        f"{ma.get('wt_start', 28) + ma.get('wt_slices', 1) - 1}]  (near MXM)"
    )
    print(
        f"  Result MEM slices: [{ma.get('result_start', 0)}.."
        f"{ma.get('result_start', 0) + ma.get('result_slices', 1) - 1}]  (near VXM)"
    )

    print("\n  Stream Transit Delays:")
    print(f"    Act MEM -> MXM:       {sf['hops_act_to_mxm']:>3} hops")
    print(f"    Wt MEM -> MXM:        {sf['hops_wt_to_mxm']:>3} hops")
    print(f"    MXM -> VXM:           {sf['hops_mxm_to_vxm']:>3} hops")
    print(f"    VXM -> Result MEM:    {sf['hops_vxm_to_result']:>3} hops")
    print(f"    Stagger (20 superlanes): {sf['stagger']:>3} cyc/instruction")
    print(f"    Hemispheres:             {sf['n_hemispheres']}x parallel")

    if topo:
        print("\n[MXM Topology]")
        print(f"  Superlanes:               {topo.get('superlanes', 'N/A')}")
        print(f"  Lanes/superlane:          {topo.get('lanes_per_superlane', 'N/A')}")
        print(f"  Physical planes/hem:      {topo.get('mxm_planes_per_hemisphere', 'N/A')}")
        print(f"  Logical planes/hem:       {topo.get('logical_mxm_planes_per_hemisphere', 'N/A')}")
        print(f"  Active logical planes:    {topo.get('active_logical_planes', 'N/A')}")
        print(f"  Plane tandem factor:      {topo.get('plane_tandem_factor', 'N/A')}")
        print(f"  Supercells/plane:         {topo.get('supercells_per_plane', 'N/A')}")
        print(f"  Supercells/chip:          {topo.get('total_supercells_chip', 'N/A')}")
        print(f"  Supercell dimension:      {topo.get('supercell_dim', 'N/A')}x{topo.get('supercell_dim', 'N/A')}")
        print(f"  Active supercell rows:    {topo.get('active_supercell_rows', 'N/A')}")
        print(f"  Active supercell cols:    {topo.get('active_supercell_cols_total', 'N/A')}")
        print(f"  MXM plane dimension:      {topo.get('mxm_plane_dim', 'N/A')}x{topo.get('mxm_plane_dim', 'N/A')}")
        for plane in topo.get("planes", [])[:2]:
            if not plane.get("active"):
                continue
            print(f"  Plane {plane['plane']} active cols: {plane.get('active_supercell_cols', 0)}")
            active_rows = [row for row in plane.get("rows", []) if row.get("active")]
            for row in active_rows[:2]:
                active_cells = [cell for cell in row.get("supercells", []) if cell.get("active")]
                cell_desc = ", ".join(
                    f"SC[{cell['supercell_row']},{cell['supercell_col']}]"
                    f"(K={cell['k_start']}..{cell['k_end']},N={cell['n_start']}..{cell['n_end']})"
                    for cell in active_cells[:3]
                )
                print(
                    f"    SL{row['superlane']:>2} lanes[{row['lane_start']}..{row['lane_end']}] "
                    f"{cell_desc}"
                )


def print_latency_breakdown(results: dict) -> None:
    ld = results.get("latency_detail", {})
    if not ld:
        return

    print("\n[Latency Breakdown (single hemisphere)]")
    items = [
        ("MEM Read (issue+SRAM)", ld.get("mem_read_issue_total", 0)),
        ("Transit act->MXM", ld.get("transit_act_total", 0)),
        ("Weight Install", ld.get("weight_install_total", 0)),
        ("Transit wt->MXM", ld.get("transit_wt_total", 0)),
        ("MXM Compute", ld.get("mxm_compute_total", 0)),
        ("MXM Pipe Fill", ld.get("mxm_pipeline_fill_total", 0)),
        ("Transit MXM->VXM", ld.get("transit_mxm_vxm_total", 0)),
        ("VXM Compute", ld.get("vxm_compute_total", 0)),
        ("Transit VXM->MEM", ld.get("transit_vxm_mem_total", 0)),
        ("Stagger Overhead", ld.get("stagger_overhead_total", 0)),
    ]
    max_val = max(v for _, v in items) if items else 1
    for name, value in items:
        if value <= 0:
            continue
        bar_len = int(value / max_val * 25) if max_val > 0 else 0
        bar = "\u2588" * bar_len + "\u2591" * (25 - bar_len)
        print(f"  {name:<22} {bar} {value:>10,} cyc")


def print_pipeline_stages(results: dict) -> None:
    print("\n[Pipeline Stages]")
    max_lat = max(results["latencies"].values()) if results["latencies"] else 1
    for name, value in results["latencies"].items():
        bar_len = int(value / max_lat * 20) if max_lat > 0 else 0
        bar = "\u2588" * bar_len + "\u2591" * (20 - bar_len)
        marker = " <-- bottleneck" if name == results.get("bottleneck") else ""
        print(f"  {name:<15} {bar} {value:>12,} cyc{marker}")


def print_energy_breakdown(results: dict) -> None:
    print("\n[Energy Breakdown]")
    total_energy = results["total_energy_pj"]
    energies = results["energies_pj"]

    groups = {
        "SRAM Read": ["sram_read_act", "sram_read_weight", "sram_psum_buffer_read"],
        "Weight Install": ["weight_install"],
        "MXM Compute": ["mxm_compute"],
        "VXM Post": ["vxm_post_proc", "vxm_operand_io"],
        "SRAM Write": ["sram_write_result", "sram_psum_buffer_write"],
        "Interconnect": ["stream_hop_act", "stream_hop_weight", "stream_hop_result", "sxm_permute"],
        "Leakage & Static": [
            "global_sram_standby",
            "local_sram_standby",
            "mxm_array_logic_leak",
            "mem_slice_ctrl",
            "sxm_logic_leak",
            "vxm_logic_leak",
        ],
        "Clock/Control": [
            "clock_backbone",
            "clock_domains",
            "control_backbone",
            "control_domains",
        ],
    }

    for group_name, keys in groups.items():
        group_energy = sum(energies.get(key, 0.0) for key in keys)
        if group_energy <= 0:
            continue
        ratio = group_energy / total_energy * 100 if total_energy > 0 else 0.0
        bar = "\u2588" * int(ratio / 5)
        print(f"  {group_name:<18} {bar:<12} {ratio:5.1f}%  ({group_energy/1e9:,.4f} mJ)")
        for key in keys:
            value = energies.get(key, 0.0)
            if value <= 0:
                continue
            sub_ratio = value / total_energy * 100 if total_energy > 0 else 0.0
            print(f"    {key:<24} {sub_ratio:5.1f}%  ({value/1e9:,.4f} mJ)")


def print_performance_summary(results: dict) -> None:
    print("\n[Performance Summary]")
    print(f"  {'Total Latency (2-hem)':<30} {results['total_latency_cycles']:>12,} cycles")
    print(f"  {'Wall-clock Cycles':<30} {results['wall_clock_cycles']:>12,} cycles")
    print(f"  {'Single Hem Latency':<30} {results['single_hem_latency_cycles']:>12,} cycles")
    print(f"  {'Total Energy':<30} {results['total_energy_pj']/1e9:>12,.4f} mJ")
    print(f"  {'Throughput':<30} {results['throughput_gops']:>12,.2f} GOPS")
    print(f"  {'Energy Efficiency':<30} {results['energy_efficiency_tops_per_watt']:>12,.2f} TOPS/W")

    sram = results.get("sram_summary", {})
    if sram:
        print("\n[SRAM Summary]")
        print(f"  {'Physical SRAM':<30} {sram.get('physical_total_sram_mib', 0.0):>12,.1f} MiB")
        print(f"  {'Global MEM SRAM':<30} {sram.get('global_mem_sram_mib', 0.0):>12,.1f} MiB")
        print(f"  {'Local Reserved SRAM':<30} {sram.get('local_reserved_sram_mib', 0.0):>12,.1f} MiB")
        print(f"  {'Per-slice Capacity':<30} {sram.get('mem_slice_capacity_mib', 0.0):>12,.3f} MiB")

    area = results.get("area_report", {})
    if area:
        print("\n[Area Summary]")
        print(f"  {'Total Area':<30} {area.get('total_area_mm2', 0.0):>12,.2f} mm^2")
        print(f"  {'Global SRAM Area':<30} {area.get('global_sram_mm2', 0.0):>12,.2f} mm^2")
        print(f"  {'Local SRAM Area':<30} {area.get('local_sram_mm2', 0.0):>12,.2f} mm^2")
        print(f"  {'Compute Area':<30} {area.get('compute_area_mm2', 0.0):>12,.2f} mm^2")
        print(f"  {'Fabric Area':<30} {area.get('fabric_area_mm2', 0.0):>12,.2f} mm^2")
        print(f"  {'Area Efficiency':<30} {area.get('tops_per_mm2', 0.0):>12,.3f} TOPS/mm^2")


def main():
    args = parse_args()

    config_dir = args.config_dir or str(Path(__file__).parent / "configs")
    loader = TSPConfigLoader(config_dir)

    print("=" * 70)
    print("  Native TSP Baseline Simulator")
    print("  (Groq TSP - MEM/SXM/MXM/VXM Model)")
    print("=" * 70)

    energy_params = loader.load_energy_config(args.energy)
    area_params = loader.load_area_config(args.area)
    hw_config = loader.load_hardware_config(args.hardware)
    workload_config = loader.load_workload_config(args.workload)
    tsp_spec = loader.create_tsp_spec(hw_config)
    M, K, N = loader.get_workload(workload_config)

    print(f"\n[Workload] GEMM: M={M}, K={K}, N={N}")
    print(f"[Hardware] {tsp_spec}")
    print(f"[Formats]  {', '.join(args.formats)}")
    print(f"[Loop]     {args.loop_order} ({LOOP_ORDERS[args.loop_order]})")

    for fmt in args.formats:
        print(f"\n{'=' * 70}")
        print(f"  Format: {fmt}")
        print(f"{'=' * 70}")

        simulator = TSPSimulator(
            tsp_spec=tsp_spec,
            energy_params=energy_params,
            data_format=fmt,
            verbose=args.verbose,
        )
        results = simulator.simulate(M, K, N, loop_order=args.loop_order)
        results["area_report"] = AreaModel(area_params).evaluate(
            results.get("area_inventory", tsp_spec.area_inventory()),
            throughput_gops=results["throughput_gops"],
        )

        print_spatial_flow(results)
        print_pipeline_stages(results)
        print_latency_breakdown(results)
        print_energy_breakdown(results)
        print_performance_summary(results)

    print("")


if __name__ == "__main__":
    main()
