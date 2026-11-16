#!/usr/bin/env python3
"""
TSP-CIM Simulator V2 - Entry Point

Groq TSP 架构 + CIM Macro 替换 MXM 的 GEMM 仿真器。
精细化东西南北向数据流建模，含 MXM baseline 对比。

用法:
    python tsp_cim.py
    python tsp_cim.py --compare-mxm
    python tsp_cim.py --sweep-m
    python tsp_cim.py --compare-loops
    python tsp_cim.py --loop-order MKN  (对比不同循环顺序)
"""
import argparse
from pathlib import Path

from src.core.area_model import AreaModel
from src.utils.tsp_cim_config import TSPCIMConfigLoader
from src.core.tsp_cim_simulator import TSPCIMSimulator, LOOP_ORDERS


def parse_args():
    parser = argparse.ArgumentParser(
        description="TSP-CIM Simulator V2 - Directional Data Flow"
    )
    parser.add_argument(
        "--hardware", type=str, default="tsp_cim.json",
        help="Hardware config file (default: tsp_cim.json)"
    )
    parser.add_argument(
        "--energy", type=str, default="7nm_tsp_cim.json",
        help="Energy config file (default: 7nm_tsp_cim.json)"
    )
    parser.add_argument(
        "--area", type=str, default="7nm_tsp_cim.json",
        help="Area config file (default: 7nm_tsp_cim.json)"
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
        help="Tile loop order (default: KNM for CIM weight reuse)"
    )
    parser.add_argument(
        "--compare-mxm", action="store_true",
        help="Show CIM vs MXM comparison"
    )
    parser.add_argument(
        "--sweep-m", action="store_true",
        help="Sweep M values to show CIM advantage curve"
    )
    parser.add_argument(
        "--compare-loops", action="store_true",
        help="Compare different loop orders"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose output"
    )
    return parser.parse_args()


def print_spatial_flow(results: dict) -> None:
    sf = results.get('spatial_flow', {})
    ma = results.get('mem_alloc', {})
    topo = results.get('macro_topology', {})
    if not sf:
        return

    print(f"\n[Spatial Data Flow Analysis]")
    print(f"  Chip Layout: VXM | MEM[0..43] | SXM | CIM | C2C  (x2 hemispheres)")
    print(f"  Act MEM slices:    [{ma.get('act_start',0)}..{ma.get('act_start',0)+ma.get('act_slices',1)-1}]"
          f"  (near VXM)")
    print(f"  Weight MEM slices: [{ma.get('wt_start',28)}..{ma.get('wt_start',28)+ma.get('wt_slices',1)-1}]"
          f"  (near CIM)")
    print(f"  Result MEM slices: [{ma.get('result_start',0)}..{ma.get('result_start',0)+ma.get('result_slices',1)-1}]"
          f"  (near VXM)")

    print(f"\n  Stream Transit Delays:")
    print(f"    Act MEM -> CIM (outward):    {sf['hops_act_to_cim']:>3} hops")
    print(f"    Wt MEM -> CIM (outward):     {sf['hops_wt_to_cim']:>3} hops")
    print(f"    CIM -> VXM (inward chain):   {sf['hops_cim_to_vxm']:>3} hops")
    print(f"    VXM -> Result MEM (outward):  {sf['hops_vxm_to_result']:>3} hops")
    print(f"    Macro fanout depth:          {sf.get('act_fanout_internal_hops', 0):>3} hops")
    print(f"    Active K groups/tile:        {sf.get('active_k_groups', 1):>3}")
    print(f"    Parallel K slots/tile:       {sf.get('active_k_parallel_groups', 1):>3}")
    print(f"    K serial depth/tile:         {sf.get('serial_k_depth', 1):>3}")
    print(f"    Stagger (20 superlanes):     {sf['stagger']:>3} cyc/instruction")
    print(f"    Hemispheres:                  {sf['n_hemispheres']}x parallel")

    print(f"\n  Loop Order: {results.get('loop_order', 'N/A')}")
    print(f"  Weight Prog Events: {results.get('weight_prog_count', 'N/A')}"
          f" / {results.get('total_tiles', 'N/A')} tiles")

    if topo:
        print(f"\n[Macro Topology]")
        print(f"  Arrangement:       {topo.get('macro_arrangement', 'N/A')}")
        print(f"  Lanes/superlane:   {topo.get('lanes_per_superlane', 'N/A')}")
        print(f"  Macros/superlane:  {topo.get('macros_per_superlane', 'N/A')}")
        print(f"  K groups/superlane:{topo.get('macro_k_groups_per_superlane', 'N/A')}")
        print(f"  N groups/superlane:{topo.get('macro_n_groups_per_superlane', 'N/A')}")
        print(f"  Macro supercell span: {topo.get('macro_supercell_k_span', 'N/A')}K x "
              f"{topo.get('macro_supercell_n_span', 'N/A')}N supercells")
        print(f"  Macro grid/plane:  {topo.get('macro_rows_per_plane', 'N/A')} rows x "
              f"{topo.get('macro_cols_per_plane', 'N/A')} cols")
        print(f"  Macro sheets/hem:  {topo.get('macro_sheets_per_hemisphere', 'N/A')}")
        print(f"  Sheet stack:       {topo.get('macro_sheet_stack_k', 'N/A')}K x "
              f"{topo.get('macro_sheet_stack_n', 'N/A')}N")
        print(f"  Parallel K slots:  {topo.get('active_k_parallel_groups', 'N/A')}")
        print(f"  Parallel N slots:  {topo.get('active_n_parallel_groups', 'N/A')}")
        print(f"  Serial K depth:    {topo.get('serial_k_depth', 'N/A')}")
        print(f"  Macros/hemisphere: {topo.get('macros_per_hemisphere', 'N/A')}")
        print(f"  Active superlanes: {topo.get('active_superlanes', 'N/A')}")
        print(f"  Active macros:     {topo.get('active_macros', 'N/A')}")
        for sheet in topo.get('sheets', [])[:2]:
            if not sheet.get('active'):
                continue
            print(f"  Sheet[{sheet['sheet_k']},{sheet['sheet_n']}] base K={sheet['k_base']} N={sheet['n_base']}")
            for plane in sheet.get('planes', [])[:2]:
                if not plane.get('active'):
                    continue
                print(f"    Plane {plane['plane']} active cols={plane.get('active_supercell_cols', 0)}")
                active_rows = [row for row in plane.get('macro_rows', []) if row.get('active')]
                for row in active_rows[:2]:
                    macro_desc = ", ".join(
                        f"M[{m['macro_row']},{m['macro_col']}]"
                        f"(SC={m['supercell_k_start']}..{m['supercell_k_end']},"
                        f"{m['supercell_n_start']}..{m['supercell_n_end']};"
                        f"K={m['k_start']}..{m['k_end']},N={m['n_start']}..{m['n_end']})"
                        for m in row.get('macros', []) if m.get('active')
                    )
                    print(f"      Row {row['macro_row']}: {macro_desc}")


def print_latency_breakdown(results: dict) -> None:
    ld = results.get('latency_detail', {})
    if not ld:
        return

    total = results['single_hem_latency_cycles']
    if total <= 0:
        return

    print(f"\n[Latency Breakdown (single hemisphere)]")
    items = [
        ("MEM Read (issue+SRAM)", ld.get('mem_read_issue_total', 0)),
        ("Transit act->CIM", ld.get('transit_act_total', 0)),
        ("Macro act fanout", ld.get('macro_fanout_act_total', 0)),
        ("Macro act dispatch", ld.get('macro_act_dispatch_total', 0)),
        ("Weight Programming", ld.get('weight_prog_total', 0)),
        ("Transit wt->CIM", ld.get('transit_wt_total', 0)),
        ("Macro wt fanout", ld.get('macro_fanout_wt_total', 0)),
        ("Macro wt dispatch", ld.get('macro_wt_dispatch_total', 0)),
        ("CIM Compute", ld.get('cim_compute_total', 0)),
        ("Transit CIM->VXM", ld.get('transit_cim_vxm_total', 0)),
        ("Macro result return", ld.get('macro_return_total', 0)),
        ("Reduce occupancy", ld.get('macro_reduce_occupancy_total', 0)),
        ("VXM Compute", ld.get('vxm_compute_total', 0)),
        ("Transit VXM->MEM", ld.get('transit_vxm_mem_total', 0)),
        ("Stagger Overhead", ld.get('stagger_overhead_total', 0)),
    ]

    max_val = max(v for _, v in items) if items else 1
    for name, val in items:
        if val <= 0:
            continue
        bar_len = int(val / max_val * 25) if max_val > 0 else 0
        bar = "\u2588" * bar_len + "\u2591" * (25 - bar_len)
        print(f"  {name:<22} {bar} {val:>10,} cyc")


def print_pipeline_stages(results: dict) -> None:
    print(f"\n[Pipeline Stages]")
    max_lat = max(results['latencies'].values()) if results['latencies'] else 1
    for k, v in results['latencies'].items():
        bar_len = int(v / max_lat * 20) if max_lat > 0 else 0
        bar = "\u2588" * bar_len + "\u2591" * (20 - bar_len)
        marker = " <-- bottleneck" if k == results.get('bottleneck') else ""
        print(f"  {k:<15} {bar} {v:>12,} cyc{marker}")


def print_energy_breakdown(results: dict) -> None:
    print(f"\n[Energy Breakdown]")
    total_e = results['total_energy_pj']
    e = results['energies_pj']

    groups = {
        "SRAM Read":       ["sram_read_act", "sram_read_weight", "sram_psum_buffer_read"],
        "Weight Prog":     ["weight_prog"],
        "CIM Compute":     ["cim_local_readout", "cim_compute"],
        "Pre-process":     ["pre_process"],
        "VXM Post":        ["vxm_post_proc", "vxm_operand_io"],
        "SRAM Write":      ["sram_write_result", "sram_psum_buffer_write"],
        "Interconnect":    [
            "stream_hop_act_external",
            "stream_hop_act_internal",
            "stream_hop_wt_external",
            "stream_hop_wt_internal",
            "stream_hop_result_external",
            "stream_hop_result_internal",
            "sxm_broadcast",
        ],
        "Reduction":       ["reduce_collect", "reduce_buffer"],
        "Leakage & Static": [
            "global_sram_standby",
            "local_sram_standby",
            "macro_logic_leak",
            "fabric_retention",
            "fabric_logic_leak",
            "sxm_logic_leak",
            "vxm_logic_leak",
        ],
        "Clock/Control":   [
            "clock_backbone",
            "clock_domains",
            "control_backbone",
            "control_domains",
        ],
    }

    for group_name, keys in groups.items():
        group_e = sum(e.get(k, 0) for k in keys)
        if group_e <= 0:
            continue
        ratio = (group_e / total_e * 100) if total_e > 0 else 0
        bar_len = int(ratio / 5)
        bar = "\u2588" * bar_len
        print(f"  {group_name:<18} {bar:<12} {ratio:5.1f}%  ({group_e/1e9:,.4f} mJ)")
        for k in keys:
            v = e.get(k, 0)
            if v <= 0:
                continue
            sub_ratio = (v / total_e * 100) if total_e > 0 else 0
            print(f"    {k:<28} {sub_ratio:5.1f}%  ({v/1e9:,.4f} mJ)")


def print_mxm_comparison(results: dict) -> None:
    mxm = results.get('mxm_baseline', {})
    if not mxm:
        return

    cim_lat = results['total_latency_cycles']
    mxm_lat = mxm['total_latency_cycles']
    cim_e = results['total_energy_pj']
    mxm_e = mxm['total_energy_pj']

    speedup = mxm_lat / cim_lat if cim_lat > 0 else 0
    energy_ratio = mxm_e / cim_e if cim_e > 0 else 0

    print(f"\n[CIM vs MXM Comparison]")
    print(f"  {'Metric':<28} {'CIM':>14} {'MXM':>14} {'Ratio':>10}")
    print(f"  {'-'*28} {'-'*14} {'-'*14} {'-'*10}")
    print(f"  {'Latency (cycles)':<28} {cim_lat:>14,} {mxm_lat:>14,} {speedup:>9.2f}x")
    print(f"  {'Throughput (GOPS)':<28} "
          f"{results['throughput_gops']:>14,.1f} {mxm['throughput_gops']:>14,.1f}")
    print(f"  {'Energy (mJ)':<28} "
          f"{cim_e/1e9:>14,.4f} {mxm_e/1e9:>14,.4f} {energy_ratio:>9.2f}x")
    print(f"  {'Eff. (TOPS/W)':<28} "
          f"{results['energy_efficiency_tops_per_watt']:>14,.2f} "
          f"{mxm['energy_efficiency_tops_per_watt']:>14,.2f}")
    if mxm.get("scope_note"):
        print(f"  Scope note: {mxm['scope_note']}")

    if speedup > 1.0:
        print(f"\n  --> CIM ADVANTAGE: {speedup:.2f}x faster, "
              f"{energy_ratio:.2f}x energy ratio")
    else:
        print(f"\n  --> MXM ADVANTAGE: CIM is {1/speedup:.2f}x slower"
              f" (but {energy_ratio:.1f}x more energy-efficient)")


def print_performance_summary(results: dict) -> None:
    print(f"\n[Performance Summary]")
    print(f"  {'Total Latency (2-hem)':<30} {results['total_latency_cycles']:>12,} cycles")
    print(f"  {'Wall-clock Cycles':<30} {results.get('wall_clock_cycles', results['total_latency_cycles']):>12,} cycles")
    print(f"  {'Single Hem Latency':<30} {results['single_hem_latency_cycles']:>12,} cycles")
    print(f"  {'Total Energy':<30} {results['total_energy_pj']/1e9:>12,.4f} mJ")
    print(f"  {'Throughput':<30} {results['throughput_gops']:>12,.2f} GOPS")
    print(f"  {'Energy Efficiency':<30} "
          f"{results['energy_efficiency_tops_per_watt']:>12,.2f} TOPS/W")
    print(f"  {'Stream Hop Energy':<30} {results['stream_hop_energy_pct']:>11,.1f}%")
    print(f"  {'Bottleneck':<30} {results.get('bottleneck', 'N/A')}")
    print(f"  {'Weight Prog Count':<30} {results.get('weight_prog_count', 'N/A')}"
          f" / {results.get('total_tiles', 'N/A')} tiles")

    sram = results.get('sram_summary', {})
    if sram:
        print(f"\n[SRAM Summary]")
        print(f"  {'Physical SRAM':<30} {sram.get('physical_total_sram_mib', 0.0):>12,.1f} MiB")
        print(f"  {'Global MEM SRAM':<30} {sram.get('global_mem_sram_mib', 0.0):>12,.1f} MiB")
        print(f"  {'Local Reserved SRAM':<30} {sram.get('local_reserved_sram_mib', 0.0):>12,.1f} MiB")
        print(f"  {'Per-slice Capacity':<30} {sram.get('mem_slice_capacity_mib', 0.0):>12,.3f} MiB")

    area = results.get('area_report', {})
    if area:
        print(f"\n[Area Summary]")
        print(f"  {'Total Area':<30} {area.get('total_area_mm2', 0.0):>12,.2f} mm^2")
        print(f"  {'Global SRAM Area':<30} {area.get('global_sram_mm2', 0.0):>12,.2f} mm^2")
        print(f"  {'Local SRAM Area':<30} {area.get('local_sram_mm2', 0.0):>12,.2f} mm^2")
        print(f"  {'Compute Area':<30} {area.get('compute_area_mm2', 0.0):>12,.2f} mm^2")
        print(f"  {'Fabric Area':<30} {area.get('fabric_area_mm2', 0.0):>12,.2f} mm^2")
        print(f"  {'Area Efficiency':<30} {area.get('tops_per_mm2', 0.0):>12,.3f} TOPS/mm^2")

    spec = results.get('tsp_spec_summary', {})
    if spec:
        print(f"\n[TSP-CIM Architecture]")
        print(f"  Total CIM Macros:  {spec.get('total_macros', 'N/A')}")
        print(f"  Macros/Hemisphere: {spec.get('macros_per_hemisphere', 'N/A')}")
        print(f"  Macros/Superlane:  {spec.get('macros_per_superlane', 'N/A')} "
              f"({spec.get('macro_k_groups_per_superlane', 'N/A')}K x "
              f"{spec.get('macro_n_groups_per_superlane', 'N/A')}N groups)")
        print(f"  Macro block:       K={spec.get('macro_block_k', 'N/A')}, N={spec.get('macro_block_n', 'N/A')}")
        print(f"  Macro supercell span: "
              f"{spec.get('macro_supercell_k_span', 'N/A')}x{spec.get('macro_supercell_n_span', 'N/A')}")
        print(f"  Macro grid / plane: {spec.get('macro_rows_per_plane', 'N/A')} x "
              f"{spec.get('macro_cols_per_plane', 'N/A')}")
        print(f"  Macro sheets / hem: {spec.get('macro_sheets_per_hemisphere', 'N/A')}")
        print(f"  Sheet stack K x N:  {spec.get('macro_sheet_stack_k', 'N/A')} x "
              f"{spec.get('macro_sheet_stack_n', 'N/A')}")
        print(f"  Parallel K slots:   {spec.get('active_k_parallel_groups', 'N/A')}")
        print(f"  Parallel N slots:   {spec.get('active_n_parallel_groups', 'N/A')}")
        print(f"  Serial K depth:     {spec.get('serial_k_depth', 'N/A')}")
        print(f"  Tile capacity:      K={spec.get('tile_k_capacity', 'N/A')}, "
              f"N={spec.get('tile_n_capacity', 'N/A')}")
        print(f"  Macro/Supercell weight ratio: "
              f"{spec.get('macro_weight_ratio_vs_supercell', 0):.2f}x")
        print(f"  Macro/Supercell INT8 peak ratio: "
              f"{spec.get('macro_peak_ratio_vs_supercell_int8', 0):.2f}x")
        print(f"  Macros per supercell (INT8 peak): "
              f"{spec.get('macros_per_supercell_peak_equivalent_int8', 0):.2f}")
        print(f"  Peak-equiv supercells/superlane: "
              f"{spec.get('peak_equivalent_supercells_per_superlane_int8', 0):.2f}")
        print(f"  Peak-equiv supercells/chip:      "
              f"{spec.get('peak_equivalent_supercells_chip_int8', 0):.2f}")
        print(f"  Total Lanes:       {spec.get('total_lanes', 'N/A')}")
        print(f"  Effective N/lane:  {spec.get('effective_n_columns_per_lane', 0):.2f} columns")
        print(f"  VXM ALUs:          {spec.get('total_vxm_alus', 'N/A')}")
        print(f"  Global MEM SRAM:   {spec.get('global_sram_mb', spec.get('mem_mb', 'N/A'))} MB")
        print(f"  Local SRAM:        {spec.get('local_sram_mb', 'N/A')} MB")
        print(f"  Physical SRAM:     {spec.get('physical_sram_mb', 'N/A')} MB")


def print_sweep_results(sweep_results: list) -> None:
    print(f"\n{'='*85}")
    print(f"  M-Sweep: CIM vs MXM Advantage Analysis")
    print(f"{'='*85}")

    print(f"\n  {'M':>6} {'K':>6} {'N':>6} | "
          f"{'CIM Lat':>10} {'MXM Lat':>10} {'Speedup':>8} | "
          f"{'E_ratio':>8} {'WtProg':>6} {'Advantage':>10}")
    print(f"  {'-'*6} {'-'*6} {'-'*6} | "
          f"{'-'*10} {'-'*10} {'-'*8} | "
          f"{'-'*8} {'-'*6} {'-'*10}")

    for r in sweep_results:
        adv = "CIM" if r['cim_advantage'] else "MXM"
        bar_len = int(abs(r['speedup'] - 1.0) * 20)
        if r['cim_advantage']:
            bar = "+" * min(bar_len, 10)
        else:
            bar = "-" * min(bar_len, 10)

        print(f"  {r['M']:>6} {r['K']:>6} {r['N']:>6} | "
              f"{r['cim_latency']:>10,} {r['mxm_latency']:>10,} "
              f"{r['speedup']:>7.2f}x | "
              f"{r['energy_ratio']:>7.2f}x "
              f"{r['weight_prog_count']:>6} "
              f"{adv:>5} {bar}")


def print_loop_comparison(loop_results: list) -> None:
    print(f"\n{'='*85}")
    print(f"  Loop Order Comparison (CIM weight reuse analysis)")
    print(f"{'='*85}")

    print(f"\n  {'Order':<6} {'Description':<40} | "
          f"{'Latency':>10} {'WtProg':>6} {'TOPS/W':>8} {'Bottleneck':>12}")
    print(f"  {'-'*6} {'-'*40} | "
          f"{'-'*10} {'-'*6} {'-'*8} {'-'*12}")

    for r in loop_results:
        print(f"  {r['loop_order']:<6} {r['description']:<40} | "
              f"{r['latency']:>10,} {r['weight_prog_count']:>6} "
              f"{r['tops_per_watt']:>8.2f} {r['bottleneck']:>12}")


def main():
    args = parse_args()

    config_dir = args.config_dir or str(Path(__file__).parent / "configs")
    loader = TSPCIMConfigLoader(config_dir)

    print("=" * 70)
    print("  TSP-CIM Simulator V2")
    print("  (Groq TSP + CIM Macro - Directional Data Flow Model)")
    print("=" * 70)

    # Load configs
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

    # Run simulation
    for fmt in args.formats:
        print(f"\n{'='*70}")
        print(f"  Format: {fmt}")
        print(f"{'='*70}")

        sim = TSPCIMSimulator(
            tsp_spec=tsp_spec,
            energy_params=energy_params,
            data_format=fmt,
            verbose=args.verbose,
        )

        results = sim.simulate(M, K, N, loop_order=args.loop_order)
        results['area_report'] = AreaModel(area_params).evaluate(
            results.get('area_inventory', tsp_spec.area_inventory()),
            throughput_gops=results['throughput_gops'],
        )

        # Print results
        print_spatial_flow(results)
        print_pipeline_stages(results)
        print_latency_breakdown(results)
        print_energy_breakdown(results)
        print_performance_summary(results)

        if args.compare_mxm:
            print_mxm_comparison(results)

        if args.sweep_m:
            m_values = [64, 128, 256, 512, 1024, 2048, 4096]
            sweep_results = sim.sweep_m(m_values, K, N, loop_order=args.loop_order)
            print_sweep_results(sweep_results)

        if args.compare_loops:
            loop_results = sim.compare_loop_orders(M, K, N)
            print_loop_comparison(loop_results)

    print("")


if __name__ == "__main__":
    main()
