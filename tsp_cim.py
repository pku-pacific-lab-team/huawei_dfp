#!/usr/bin/env python3
"""TSP-CIM single Cluster counts/timing and separate whole-chip specifications."""
# 装配单 domain 仿真；循环顺序对比和 M 扫描都重新运行模型，不继承前次 context 状态。
import argparse
import json
from pathlib import Path
import sys

from src.core.tsp_cim.simulator import TSPCIMSimulator
from src.utils.tsp_cim_config import TSPCIMConfigLoader
from src.utils.workload import parse_workloads


def parse_args():
    parser = argparse.ArgumentParser(description="TSP-CIM INT8 single Cluster count or simplified serial timing")
    parser.add_argument("--hardware", default="tsp_cim.json")
    parser.add_argument("--workload", default="tiny_gemm.json")
    parser.add_argument("--config-dir")
    parser.add_argument("--formats", nargs="+")
    parser.add_argument("--loop-order", default="KNM")
    parser.add_argument("--mode", choices=["count", "timing"], default="timing",
                        help="count: actions only; timing (default): counts and task time with explicit assumptions")
    parser.add_argument("--scope", choices=["single_cluster", "chip"], default="single_cluster",
                        help="Execution scope; chip execution is unavailable, inventory is always reported")
    parser.add_argument("--cluster", nargs=3, type=int, default=(0, 0, 0), metavar=("HEMISPHERE", "PLANE", "CLUSTER"))
    parser.add_argument("--tile-m", type=int, help="Logical row grouping for reuse experiments; default is all M rows")
    parser.add_argument("--compare-mxm", action="store_true")
    parser.add_argument("--compare-loops", action="store_true")
    parser.add_argument("--sweep-m", nargs="+", type=int)
    parser.add_argument("--energy", help="Unavailable until new-model calibration")
    parser.add_argument("--area", help="Unavailable until new-model calibration")
    parser.add_argument("--output", default=str(Path(__file__).resolve().parent / "results" / "tsp_cim.json"),
                        help="JSON report path (default: %(default)s; overwrites on success)")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        loader = TSPCIMConfigLoader(args.config_dir)
        workloads = parse_workloads(loader.load_workload_config(args.workload), args.formats)
        for workload in workloads:
            print("[Effective workload] " + json.dumps(workload.to_dict(), ensure_ascii=False))
        if args.energy or args.area:
            raise ValueError("New CIM Macro energy/area unavailable; legacy coefficients are unsupported")
        spec = loader.create_tsp_spec(loader.load_hardware_config(args.hardware))
        results = []
        for workload in workloads:
            sim = TSPCIMSimulator(spec, data_format=workload.data_format)
            # 三个 cluster 参数依次选择半片、Plane、domain，不表示一次运行三个层级。
            policy = dict(mode=args.mode, scope=args.scope, hemisphere=args.cluster[0],
                          plane=args.cluster[1], cluster=args.cluster[2], tile_m=args.tile_m)
            result = sim.simulate(*workload.shape, args.loop_order, **policy)
            # 保留公共任务解析器记录的声明精度、覆盖标记和累加/输出类型。
            result["workload"] = workload.to_dict()
            if args.compare_mxm:
                result["comparison"] = sim.compare_mxm(*workload.shape, args.loop_order, workload)
            if args.compare_loops:
                result["loop_order_counts"] = sim.compare_loop_orders(*workload.shape, **policy)
            if args.sweep_m:
                result["m_sweep"] = sim.sweep_m(args.sweep_m, workload.K, workload.N, args.loop_order, **policy)
            results.append(result)
        output = results[0] if len(results) == 1 else results
        print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))
        if args.output:
            destination = Path(args.output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            print(f"[Report] {args.output}")
        return 0
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
