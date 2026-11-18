#!/usr/bin/env python3
"""Original Ascend 910 specification and single-core analytical GEMM smoke."""
# 入口只装配规格、任务与报告；分块和五条搬运路径的计算在 src/core/ascend 中。
import argparse
from pathlib import Path
import sys

from src.core.ascend import AscendSimulator
from src.utils.ascend_config import AscendConfigLoader
from src.utils.ascend_report import print_report, write_report
from src.utils.workload import load_workload_config, parse_workloads


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Ascend 910 coarse analytical GEMM (single core, INT8)")
    parser.add_argument("--hardware", default="ascend910_reference.json")
    parser.add_argument("--workload", default="tiny_gemm.json")
    parser.add_argument("--config-dir")
    parser.add_argument("--format", default=None, help="Explicit precision override; supports INT8")
    parser.add_argument("--mode", choices=["count", "timing"], default="timing",
                        help="count: operations/traffic only; timing (default): counts and task time with explicit assumptions")
    parser.add_argument("--tile", nargs=3, type=int, default=None, metavar=("M", "K", "N"))
    parser.add_argument("--l1-k", type=int, default=None)
    parser.add_argument("--buffers", type=int, choices=[1, 2], default=None,
                        help="Maximum slots per local buffer; falls back to one where two do not fit")
    parser.add_argument("--trace", action="store_true", help="Include task and buffer-lifetime trace (timing only)")
    parser.add_argument("--output", default=str(Path(__file__).resolve().parent / "results" / "ascend.json"),
                        help="JSON report path (default: %(default)s; overwrites on success)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        loader = AscendConfigLoader(args.config_dir)
        simulator = AscendSimulator(loader.load_hardware(args.hardware))
        config = load_workload_config(args.workload, loader.config_dir)
        workload = parse_workloads(config, args.format)[0]
        # --format 是显式覆盖，未指定时保留 workload 声明，不自动转成 INT8。
        result = simulator.simulate(workload, args.mode, tuple(args.tile) if args.tile is not None else None, args.l1_k,
                                    buffers=args.buffers, trace=args.trace)
        print_report(result)
        if args.output:
            write_report(result, args.output)
            print(f"[Report] {args.output}")
        return 0
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    raise SystemExit(main())
