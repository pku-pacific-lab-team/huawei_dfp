#!/usr/bin/env python3
"""Native TSP issue #2 analytical reference."""
# 独立 TSP 与 CIM 对比共用 run_reference/TSPSimulator，确保原生基线的计算口径一致。
import argparse
import json
from pathlib import Path
import sys
from src.utils.tsp_config import TSPConfigLoader
from src.utils.tsp_reference import run_reference
from src.utils.tsp_report import print_report, write_report
from src.utils.workload import parse_workloads
from src.core.tsp.simulator import LOOP_ORDERS
from src.operators import adapt_gemm_report
from src.runtime.model_cli import add_model_arguments, model_run
from src.runtime.cli import add_vector_arguments, handles_operator_workload, operator_options, write_operator_result, write_operator_results
from src.runtime.workloads import execute_workload


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Native TSP single-hemisphere analytical reference")
    p.add_argument("--hardware", default="tsp.json", help="Hardware profile (default: tsp.json)")
    p.add_argument("--workload", default="tiny_gemm.json")
    p.add_argument("--config-dir")
    p.add_argument("--formats", nargs="+", help="Explicit precision override; reference supports INT8")
    p.add_argument("--loop-order", choices=list(LOOP_ORDERS), default="KNM")
    p.add_argument("--mode", choices=["count","timing"], default="timing",
                   help="count: operations/traffic only; timing (default): counts and task time with explicit assumptions")
    p.add_argument("--scope", choices=["single_hem","chip"], default="single_hem")
    p.add_argument("--tile", type=int, nargs=3, default=None, metavar=("M","K","N"))
    p.add_argument("--mem-slices", type=int, nargs="+")
    p.add_argument("--streams", type=int)
    p.add_argument("--reserved-bytes", type=int, default=0)
    p.add_argument("--output", default=str(Path(__file__).resolve().parent / "results" / "tsp.json"),
                   help="JSON report path (GEMM default: %(default)s; Vector/composite: project results/<arch>_<op>.json; overwrites on success)")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--report-format", choices=["legacy", "operator"], default="legacy",
                   help="operator: unified GEMM result and action ledger; legacy (default): existing report")
    add_model_arguments(p)
    add_vector_arguments(p)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        loader = TSPConfigLoader(args.config_dir)
        result = model_run(args, "tsp", lambda: loader.create_tsp_spec(loader.load_hardware_config(args.hardware)), argv)
        if result is not None:
            return result
        config = loader.load_workload_config(args.workload)
        if handles_operator_workload(config, args, argv):
            result = execute_workload(config, "tsp", lambda: loader.create_tsp_spec(loader.load_hardware_config(args.hardware)), mode=args.mode,
                                      execution_file=args.vector_config, config_dir=loader.config_dir,
                                      **operator_options(args, "tsp"))
            write_operator_result(result, args, argv)
            return 0
        workloads = parse_workloads(config, args.formats)
        reports = []
        for w in workloads:
            # 每个精度任务独立运行；打印实际任务，便于确认是否使用了显式精度覆盖。
            if args.report_format == "legacy":
                print("[Effective workload] " + json.dumps(w.to_dict(), ensure_ascii=False))
            result = run_reference(w, hardware=args.hardware,config_dir=args.config_dir,
                mode=args.mode,loop_order=args.loop_order,scope=args.scope,tile=tuple(args.tile) if args.tile is not None else None,
                mem_slices=args.mem_slices,streams=args.streams,reserved_bytes=args.reserved_bytes)
            if args.report_format == "legacy":
                print_report(result)
            if args.verbose and args.report_format == "legacy":
                print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
            reports.append(result)
        if args.report_format == "operator":
            write_operator_results([adapt_gemm_report("tsp", report, include_details=bool(args.details_output)) for report in reports], args)
            return 0
        if args.output:
            write_report(args.output, reports[0] if len(reports)==1 else {"runs":reports})
            print(f"[Report] {args.output}")
        return 0
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(f"[ERROR] {exc}",file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    raise SystemExit(main())
