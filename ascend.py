#!/usr/bin/env python3
"""
Cube Simulator - Ascend 910 AI Processor Simulation

用法:
    python ascend.py                              
    python ascend.py --energy configs/energy/12nm.json
    python ascend.py --hardware configs/hardware/ascend.json
    python ascend.py --workload configs/workload/default_gemm.json
"""
import argparse
from pathlib import Path

from src.hardware.cube import CubeSpec
from src.core.cube_simulator import CubeCoreSimulator
from src.utils.cube_config import ConfigLoader


def resolve_config_path(file_path: str, base_dir: Path) -> str:
    """
    解析配置文件路径，相对于ascend.py文件所在目录
    
    如果用户提供的是相对路径（如 configs/energy/12nm.json），
    会将其解析为相对于ascend.py的路径，并提取出文件名供ConfigLoader使用。
    
    Args:
        file_path: 用户提供的文件路径（可能是文件名、相对路径或绝对路径）
        base_dir: ascend.py文件所在的目录
        
    Returns:
        解析后的文件名，供ConfigLoader在相应的子目录中查找
    """
    path = Path(file_path)
    
    # 如果是绝对路径
    if path.is_absolute():
        configs_dir = base_dir / "configs"
        try:
            # 检查是否在configs目录下
            relative_path = path.relative_to(configs_dir)
            # 如果在configs的子目录中（如 energy/12nm.json），返回文件名
            if len(relative_path.parts) >= 2:
                return relative_path.name
        except ValueError:
            pass
        # 不在configs目录下，返回文件名
        return path.name
    
    # 如果是相对路径，相对于base_dir解析
    full_path = (base_dir / path).resolve()
    configs_dir = base_dir / "configs"
    
    try:
        # 检查解析后的路径是否在configs目录下
        relative_path = full_path.relative_to(configs_dir)
        # 如果在configs的子目录中（如 energy/12nm.json），返回文件名
        if len(relative_path.parts) >= 2:
            return relative_path.name
        elif len(relative_path.parts) == 1:
            # 直接在configs目录下
            return relative_path.name
    except ValueError:
        # 不在configs目录下，可能是纯文件名，直接返回
        pass
    
    # 默认返回文件名
    return path.name


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Cube Simulator - Ascend 910 AI Processor Simulation"
    )
    parser.add_argument(
        "--energy",
        type=str,
        default="12nm.json",
        help="Energy configuration file name or path relative to ascend.py (default: 12nm.json)"
    )
    parser.add_argument(
        "--hardware",
        type=str,
        default="ascend.json",
        help="Hardware configuration file name or path relative to ascend.py (default: ascend.json)"
    )
    parser.add_argument(
        "--workload",
        type=str,
        default="tiny_gemm.json",
        help="Workload configuration file name or path relative to ascend.py (default: default_gemm.json)"
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default=None,
        help="Configuration directory path (default: ./configs relative to ascend.py)"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="INT8",
        choices=["INT8", "INT16", "FP16"],
        help="Data format to test (default: INT8)"
    )
    parser.add_argument(
        "--loop-order",
        type=str,
        default="MKN",
        choices=["MKN", "KMN", "MNK", "NKM", "KNM", "NMK"],
        help="Loop order (default: MKN)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    return parser.parse_args()


def print_simulation_result(results: dict) -> None:
    """
    打印Cube仿真结果
    
    Args:
        results: 仿真结果字典
    """
    print(f"\n{'='*80}")
    print(f"  CUBE SIMULATION RESULTS")
    print(f"{'='*80}")
    
    # 1. 配置信息
    print(f"\n[Configuration]")
    print(f"  Format: {results.get('Format', 'N/A')}")
    print(f"  L0 Tile: M={results.get('L0_Tile_M', 'N/A')}, "
          f"N={results.get('L0_Tile_N', 'N/A')}, "
          f"K={results.get('L0_Tile_K', 'N/A')}")
    
    # 2. 流水线延迟
    print(f"\n[Pipeline Latency]")
    if 'latencies' in results:
        max_latency = max(results['latencies'].values()) if results['latencies'].values() else 1
        for k, v in results['latencies'].items():
            bar_len = int(v / max_latency * 30) if max_latency > 0 else 0
            bar = "█" * bar_len + "░" * (30 - bar_len)
            print(f"  {k:<15} {bar} {v:>12,} cyc")
    
    print(f"\n  Bottleneck: {results.get('bottleneck', 'N/A')}")
    
    # 3. 能耗分解
    print(f"\n[Energy Breakdown]")
    total_e = results.get('total_energy_pj', 0)
    
    if total_e > 0 and 'energies_pj' in results:
        # 按能耗排序，显示所有项
        sorted_energy = sorted(results['energies_pj'].items(), key=lambda x: -x[1])
        for k, v in sorted_energy:
            ratio = (v / total_e * 100) if total_e > 0 else 0
            bar_len = int(ratio / 5)  # 每5%一个块
            bar = "█" * bar_len
            print(f"  {k:<20} {bar:<10} {ratio:5.1f}%  ({v:,.0f} pJ)")
    
    # 4. 总体性能
    print(f"\n[Performance Summary]")
    print(f"  Total Latency : {results.get('total_latency_cycles', 0):>12,} Cycles")
    print(f"  Total Energy  : {results.get('total_energy_pj', 0):>12,.0f} pJ")
    print(f"  Throughput    : {results.get('throughput_gops', 0):>12,.2f} GFLOPS")
    print(f"  Efficiency    : {results.get('energy_efficiency_tops_per_watt', 0):>12,.2f} TFLOPS/W")
    print(f"{'='*80}\n")


def main():
    """主函数"""
    args = parse_args()

    # 获取ascend.py文件所在目录作为基准目录
    base_dir = Path(__file__).parent
    
    # 解析配置文件路径，使其相对于ascend.py文件
    energy_file = resolve_config_path(args.energy, base_dir)
    hardware_file = resolve_config_path(args.hardware, base_dir)
    workload_file = resolve_config_path(args.workload, base_dir)
    
    # 1. 加载配置
    if args.config_dir:
        # 如果用户指定了config_dir，解析为相对于ascend.py的路径
        config_dir_path = Path(args.config_dir)
        if not config_dir_path.is_absolute():
            config_dir = str((base_dir / config_dir_path).resolve())
        else:
            config_dir = args.config_dir
    else:
        config_dir = str(base_dir / "configs")
    
    loader = ConfigLoader(config_dir)

    print("=" * 80)
    print("  Cube Simulator - Ascend 910 AI Processor")
    print("=" * 80)

    # 加载配置
    energy_params = loader.load_energy_config(energy_file)
    hardware_config = loader.load_hardware_config(hardware_file)
    workload_config = loader.load_workload_config(workload_file)

    # 2. 创建Cube规格
    cube_spec = loader.create_cube_spec(hardware_config)

    # 3. 获取参数
    workload = loader.get_workload(workload_config)  # (M, K, N)
    l0_params = loader.get_l0_cube_params(hardware_config)
    l1_params = loader.get_l1_cube_params(hardware_config)
    bw_params = loader.get_cube_bandwidth_params(hardware_config)
    clock_freq = loader.get_clock_freq(hardware_config)

    # 打印关键配置信息
    print(f"\n[Workload] GEMM: M={workload[0]}, K={workload[1]}, N={workload[2]}")
    print(f"[Hardware] Cube: {cube_spec}")
    print(f"[Memory]  L1: {l1_params['size_l1_buffer_bytes']/1024:.0f} KB, "
          f"L0_A: {l0_params['size_l0_A_input_buffer_bytes']/1024:.0f} KB, "
          f"L0_B: {l0_params['size_l0_B_input_buffer_bytes']/1024:.0f} KB, "
          f"L0_C: {l0_params['size_l0_C_output_buffer_bytes']/1024:.0f} KB")
    print(f"[Bandwidth] L1->L0: {bw_params['bw_l1_to_l0_gbps']:.1f} Gbps, "
          f"L0->Cube: {bw_params['bw_l0_to_cube_gbps']:.1f} Gbps")
    print(f"[Format]   {args.format}")
    print(f"[Loop Order] {args.loop_order}")

    # 4. 创建Cube仿真器
    M, K, N = workload
    
    # 从配置中获取compute_latency，如果没有则使用默认值1
    cube_config = hardware_config.get("Cube", {})
    compute_latency = cube_config.get("compute_latency", 1)
    
    simulator = CubeCoreSimulator(
        macro_spec=cube_spec,
        size_l0_A_input_buffer_bytes=l0_params['size_l0_A_input_buffer_bytes'],
        size_l0_B_input_buffer_bytes=l0_params['size_l0_B_input_buffer_bytes'],
        size_l0_C_output_buffer_bytes=l0_params['size_l0_C_output_buffer_bytes'],
        bw_l0_to_cube_gbps=bw_params['bw_l0_to_cube_gbps'],
        energy_params=energy_params,
        clock_freq_ghz=clock_freq,
        data_format=args.format,
        custom_overheads=None,
        compute_latency=compute_latency,
        size_l1_buffer_bytes=l1_params['size_l1_buffer_bytes'],
        bw_l1_to_l0_gbps=bw_params['bw_l1_to_l0_gbps'],
        verbose=args.verbose
    )

    # 5. 运行仿真（Cube架构不支持并行，partition_strategy设为(1,1)）
    print(f"\n[Simulation] Running with loop order: {args.loop_order}")
    print("-" * 80)
    
    try:
        results = simulator.simulate(
            M=M,
            N=N,
            K=K,
            partition_strategy=(1, 1),  # Cube架构不支持并行
            loop_order=args.loop_order
        )
        
        # 6. 打印结果
        print_simulation_result(results)
        
    except Exception as e:
        print(f"\n[ERROR] Simulation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
