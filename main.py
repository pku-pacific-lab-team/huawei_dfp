#!/usr/bin/env python3
"""
Digital CIM Simulator - Main Entry Point

用法:
    python main.py                              # 使用默认配置运行DSE
    python main.py --energy configs/energy/default_22nm.json
    python main.py --hardware configs/hardware/default_core.json
    python main.py --workload configs/workload/default_gemm.json
"""
import argparse
import itertools
from pathlib import Path

from src.hardware.macro import CIMMacroSpec
from src.dse.explorer import CIMDesignExplorer
from src.utils.config import ConfigLoader


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Digital CIM Simulator - Design Space Exploration"
    )
    parser.add_argument(
        "--energy",
        type=str,
        default="7nm_cim.json",
        help="Energy configuration file name (default: 7nm_cim.json)"
    )
    parser.add_argument(
        "--hardware",
        type=str,
        default="default_core.json",
        help="Hardware configuration file name (default: default_core.json)"
    )
    parser.add_argument(
        "--workload",
        type=str,
        default="default_gemm.json",
        help="Workload configuration file name (default: default_gemm.json)"
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default=None,
        help="Configuration directory path (default: ./configs)"
    )
    parser.add_argument(
        "--formats",
        type=str,
        nargs="+",
        default=["INT8", "BF16", "MXINT8", "FP8", "MXFP8_M4"],
        help="Data formats to test (default: all supported formats)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 1. 加载配置
    config_dir = args.config_dir or str(Path(__file__).parent / "configs")
    loader = ConfigLoader(config_dir)

    print("=" * 60)
    print("  Digital CIM Simulator - Design Space Exploration")
    print("=" * 60)

    # 加载配置（静默模式）
    energy_params = loader.load_energy_config(args.energy)
    hardware_config = loader.load_hardware_config(args.hardware)
    workload_config = loader.load_workload_config(args.workload)

    # 2. 创建硬件规格
    macro_spec = loader.create_macro_spec(hardware_config)

    # 3. 获取参数
    workload = loader.get_workload(workload_config)
    l0_params = loader.get_l0_params(hardware_config)
    fixed_bw = loader.get_bandwidth_params(hardware_config)

    # 打印关键配置信息
    print(f"\n[Workload] GEMM: M={workload[0]}, K={workload[1]}, N={workload[2]}")
    print(f"[Hardware] Macros=32, L0={l0_params['input']}B, BW: NoC={fixed_bw['noc_to_l1']}Gbps")
    print(f"[Formats]  {', '.join(args.formats)}")

    # 4. 定义搜索空间
    l1_candidates = [16*1024, 32*1024, 64*1024, 128*1024, 256*1024, 512*1024, 1024*1024, 2048*1024, 4096*1024, 8192*1024]
    loop_orders = [''.join(p) for p in itertools.permutations('MKN')]

    # 5. 初始化DSE引擎
    total_macros = hardware_config.get("core", {}).get("n_macros", 32)

    dse = CIMDesignExplorer(
        macro_spec=macro_spec,
        l0_params=l0_params,
        energy_params=energy_params,
        workload=workload,
        fixed_bw=fixed_bw
    )

    # 6. 运行DSE
    print(f"\n[DSE] Search space: {len(l1_candidates)} L1 sizes × 6 strategies × 6 loop orders")
    print("-" * 60)

    all_results = []
    for fmt in args.formats:
        print(f"  [{fmt}]", end=" ")
        results = dse.run_dse_hybrid(
            l1_candidates,
            loop_orders,
            data_format=fmt,
            total_macros=total_macros,
            verbose=args.verbose
        )
        all_results.extend(results)

    # 7. 分析结果
    if all_results:
        df_all = dse.analyze_results(all_results, show_summary=False)

        # 输出每种数据格式的最佳配置
        print(f"\n{'='*120}")
        print(f"  BEST CONFIGURATION PER FORMAT (by EDP)")
        print(f"{'='*120}")
        print(f"{'Format':<12} {'L1(KB)':<8} {'Strategy':<12} {'Order':<7} "
              f"{'Throughput':>10} {'Energy':>10} {'w/o DRAM':>10} "
              f"{'TOPS/W':>8} {'w/o DRAM':>8} {'Bottleneck':<12}")
        print("-" * 120)

        for fmt in args.formats:
            df_fmt = df_all[df_all['Data_Format'] == fmt]
            if df_fmt.empty:
                continue
            best = df_fmt.sort_values(by="EDP").iloc[0]
            total_e = best['Energy_pJ']
            compute_e = best['Compute_Energy_pJ']
            total_ops = best['Throughput_GOPS'] * best['Latency_Cyc']
            eff_full = total_ops / total_e if total_e > 0 else 0
            eff_no_dram = total_ops / compute_e if compute_e > 0 else 0
            print(f"{best['Data_Format']:<12} {best['L1_Size_KB']:<8.0f} "
                  f"{best['Strategy']:<12} {best['Loop_Order']:<7} "
                  f"{best['Throughput_GOPS']:>10.1f} {total_e/1e9:>9.2f}  {compute_e/1e9:>9.2f}  "
                  f"{eff_full:>7.2f}  {eff_no_dram:>7.2f} {best['Bottleneck']:<12}")
        print("-" * 120)

        # 找到全局最佳配置并复现详细报告
        best_config_overall = df_all.sort_values(by="EDP").iloc[0]
        dse.reproduce_specific_config(best_config_overall, total_macros=total_macros)
    else:
        print("\n[ERROR] No valid results. Check configuration.")


if __name__ == "__main__":
    main()
