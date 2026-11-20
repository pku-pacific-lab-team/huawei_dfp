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
from src.operators import adapt_gemm_report
from src.runtime.cli import add_vector_arguments, handles_operator_workload, operator_options, write_operator_result, write_operator_results
from src.runtime.workloads import execute_workload


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
    parser.add_argument("--report-format", choices=["legacy", "operator"], default="legacy",
                        help="operator: unified GEMM result and action ledger; legacy (default): existing report")
    parser.add_argument("--output", default=str(Path(__file__).resolve().parent / "results" / "ascend.json"),
                        help="JSON report path (GEMM default: %(default)s; Vector/composite: project results/<arch>_<op>.json; overwrites on success)")
    add_vector_arguments(parser)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        loader = AscendConfigLoader(args.config_dir)
        spec = loader.load_hardware(args.hardware)
        config = load_workload_config(args.workload, loader.config_dir)
        if handles_operator_workload(config, args, argv):
            result = execute_workload(config, "ascend", lambda: spec, mode=args.mode,
                                      execution_file=args.vector_config, config_dir=loader.config_dir,
                                      **operator_options(args, "ascend"))
            write_operator_result(result, args, argv)
            return 0
        workload = parse_workloads(config, args.format)[0]
        # --format 是显式覆盖，未指定时保留 workload 声明，不自动转成 INT8。
        simulator = AscendSimulator(spec)
        result = simulator.simulate(workload, args.mode, tuple(args.tile) if args.tile is not None else None, args.l1_k,
                                    buffers=args.buffers, trace=args.trace)
        if args.report_format == "operator":
            write_operator_results([adapt_gemm_report("ascend", result, include_details=bool(args.details_output))], args)
            return 0
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
