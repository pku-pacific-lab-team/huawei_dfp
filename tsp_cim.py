#!/usr/bin/env python3
"""TSP-CIM hemisphere execution and separate whole-chip specifications."""
# 默认装配半片执行；循环顺序对比和 M 扫描不继承前次 context 状态。
import argparse
import json
from pathlib import Path
import sys

from src.core.tsp_cim.simulator import TSPCIMSimulator
from src.utils.tsp_cim_config import TSPCIMConfigLoader
from src.utils.workload import parse_workloads
from src.operators import adapt_gemm_report
from src.runtime.cli import add_vector_arguments, handles_operator_workload, operator_options, write_operator_result, write_operator_results
from src.runtime.workloads import execute_workload


def parse_args():
    parser = argparse.ArgumentParser(description="TSP-CIM INT8 hemisphere counts and analytical scheduling")
    parser.add_argument("--hardware", default="tsp_cim.json")
    parser.add_argument("--workload", default="tiny_gemm.json")
    parser.add_argument("--config-dir")
    parser.add_argument("--formats", nargs="+")
    parser.add_argument("--loop-order", default="KNM")
    parser.add_argument("--mode", choices=["count", "timing"], default="timing",
                        help="count: actions only; timing (default): counts and task time with explicit assumptions")
    parser.add_argument("--hemisphere", type=int, default=0)
    parser.add_argument("--tile-m", type=int, help="Row grouping; hemisphere default is min(M,320)")
    parser.add_argument("--mem-slices", nargs="+", type=int)
    parser.add_argument("--streams", type=int)
    parser.add_argument("--reserved-bytes", type=int, default=0)
    parser.add_argument("--load-overlap", choices=["serialized", "background"])
    parser.add_argument("--compare-load-policies", action="store_true")
    parser.add_argument("--compare-mxm", action="store_true")
    parser.add_argument("--compare-loops", action="store_true")
    parser.add_argument("--sweep-m", nargs="+", type=int)
    parser.add_argument("--report-format", choices=["legacy", "operator"], default="legacy",
                        help="operator: unified GEMM result and action ledger; legacy (default): existing report")
    parser.add_argument("--output", default=str(Path(__file__).resolve().parent / "results" / "tsp_cim.json"),
                        help="JSON report path (GEMM default: %(default)s; Vector/composite: project results/<arch>_<op>.json; overwrites on success)")
    add_vector_arguments(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        loader = TSPCIMConfigLoader(args.config_dir)
        config = loader.load_workload_config(args.workload)
        if handles_operator_workload(config, args):
            result = execute_workload(config, "tsp_cim", lambda: loader.create_tsp_spec(loader.load_hardware_config(args.hardware)), mode=args.mode,
                                      execution_file=args.vector_config, config_dir=loader.config_dir,
                                      **operator_options(args, "tsp_cim"))
            write_operator_result(result, args)
            return 0
        workloads = parse_workloads(config, args.formats)
        if args.report_format == "legacy":
            for workload in workloads:
                print("[Effective workload] " + json.dumps(workload.to_dict(), ensure_ascii=False))
        spec = loader.create_tsp_spec(loader.load_hardware_config(args.hardware))
        results = []
        for workload in workloads:
            sim = TSPCIMSimulator(spec, data_format=workload.data_format)
            policy = dict(mode=args.mode, hemisphere=args.hemisphere, tile_m=args.tile_m,
                          mem_slices=args.mem_slices, streams=args.streams, reserved_bytes=args.reserved_bytes, load_overlap=args.load_overlap)
            result = sim.simulate(*workload.shape, args.loop_order, **policy)
            # 保留公共任务解析器记录的声明精度、覆盖标记和累加/输出类型。
            result["workload"] = workload.to_dict()
            if args.compare_load_policies:
                result["load_policy_comparison"] = sim.compare_load_policies(*workload.shape, args.loop_order, **policy)
            if args.compare_mxm:
                result["comparison"] = sim.compare_mxm(*workload.shape, args.loop_order, workload, result=result)
            if args.compare_loops:
                result["loop_order_counts"] = sim.compare_loop_orders(*workload.shape, **policy)
            if args.sweep_m:
                result["m_sweep"] = sim.sweep_m(args.sweep_m, workload.K, workload.N, args.loop_order, **policy)
            results.append(result)
        if args.report_format == "operator":
            write_operator_results([adapt_gemm_report("tsp_cim", report, include_details=bool(args.details_output)) for report in results], args)
            return 0
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
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    raise SystemExit(main())
