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
    p.add_argument("--tile", type=int, nargs=3, default=(320,320,640), metavar=("M","K","N"))
    p.add_argument("--mem-slices", type=int, nargs="+")
    p.add_argument("--streams", type=int)
    p.add_argument("--reserved-bytes", type=int, default=0)
    p.add_argument("--output", default=str(Path(__file__).resolve().parent / "results" / "tsp.json"),
                   help="JSON report path (default: %(default)s; overwrites on success)")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        loader = TSPConfigLoader(args.config_dir)
        workloads = parse_workloads(loader.load_workload_config(args.workload), args.formats)
        reports = []
        for w in workloads:
            # 每个精度任务独立运行；打印实际任务，便于确认是否使用了显式精度覆盖。
            print("[Effective workload] " + json.dumps(w.to_dict(), ensure_ascii=False))
            result = run_reference(w, hardware=args.hardware,config_dir=args.config_dir,
                mode=args.mode,loop_order=args.loop_order,scope=args.scope,tile=tuple(args.tile),
                mem_slices=args.mem_slices,streams=args.streams,reserved_bytes=args.reserved_bytes)
            print_report(result)
            if args.verbose:
                print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
            reports.append(result)
        if args.output:
            write_report(args.output, reports[0] if len(reports)==1 else {"runs":reports})
            print(f"[Report] {args.output}")
        return 0
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(f"[ERROR] {exc}",file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
