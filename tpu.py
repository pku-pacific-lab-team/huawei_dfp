#!/usr/bin/env python3
"""TPU v4 chip hierarchy and single-TensorCore INT8 count/timing entry."""
# 入口默认报告计数与时间估算，执行范围仅覆盖一个 TensorCore 的逻辑 GEMM。
import argparse
from pathlib import Path
import sys

from src.core.tpu import TPUSimulator
from src.utils.tpu_config import TPUConfigLoader
from src.utils.tpu_report import print_report, write_report
from src.utils.workload import load_workload_config, parse_workloads
from src.operators import adapt_gemm_report
from src.runtime.model_cli import add_model_arguments, model_run
from src.runtime.cli import add_vector_arguments, handles_operator_workload, operator_options, write_operator_result, write_operator_results
from src.runtime.workloads import execute_workload


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="TPU v4 single-TensorCore INT8 GEMM counts and analytical timing")
    parser.add_argument("--hardware", default="tpu_v4_reference.json")
    parser.add_argument("--workload", default="tiny_gemm.json")
    parser.add_argument("--config-dir")
    parser.add_argument("--format", help="Explicit workload precision override; only INT8 is supported")
    parser.add_argument("--mode", choices=["count", "timing"], default="timing",
                        help="count: logical actions only; timing (default): counts and task time with explicit assumptions")
    parser.add_argument("--tile", nargs=3, type=int, default=None, metavar=("M", "K", "N"),
                        help="Logical block upper bounds; timing splits each output block into MXU waves")
    parser.add_argument("--schedule", choices=["resource_pipeline", "serial"])
    parser.add_argument("--buffers", type=int, choices=[1, 2])
    parser.add_argument("--cmem-buffers", type=int, choices=[1, 2],
                        help="Requested CMEM panel slots, separate from VMEM --buffers; capacity may reduce to one")
    parser.add_argument("--cmem-mode", choices=["tiled", "bypass"],
                        help="CMEM input-panel reuse or direct HBM-to-VMEM comparison; defaults to hardware policy")
    parser.add_argument("--cmem-groups", nargs=2, type=int, metavar=("M", "N"),
                        help="Maximum VMEM output-block counts per CMEM outer tile; capacity may shrink them")
    parser.add_argument("--report-format", choices=["legacy", "operator"], default="legacy",
                        help="operator: unified GEMM result and action ledger; legacy (default): existing report")
    parser.add_argument("--output", default=str(Path(__file__).resolve().parent / "results" / "tpu.json"),
                        help="JSON report path (GEMM default: %(default)s; Vector/composite: project results/<arch>_<op>.json; overwrites on success)")
    add_model_arguments(parser)
    add_vector_arguments(parser)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        loader = TPUConfigLoader(args.config_dir)
        result = model_run(args, "tpu", lambda: loader.load_hardware(args.hardware), argv)
        if result is not None:
            return result
        spec = loader.load_hardware(args.hardware)
        config = load_workload_config(args.workload, loader.config_dir)
        if handles_operator_workload(config, args, argv):
            result = execute_workload(config, "tpu", lambda: spec, mode=args.mode,
                                      execution_file=args.vector_config, config_dir=loader.config_dir,
                                      **operator_options(args, "tpu"))
            write_operator_result(result, args, argv)
            return 0
        workload = parse_workloads(config, args.format)[0]
        # tile 是逻辑块上限，具体可行尺寸由 tiling.py 按 VMEM 容量确定。
        report = TPUSimulator(spec).simulate(workload, tile=tuple(args.tile) if args.tile is not None else None,
                    mode=args.mode, schedule=args.schedule, buffers=args.buffers,
                    cmem_mode=args.cmem_mode, cmem_groups_mn=args.cmem_groups, cmem_buffers=args.cmem_buffers)
        if args.report_format == "operator":
            write_operator_results([adapt_gemm_report("tpu", report, include_details=bool(args.details_output))], args)
            return 0
        print_report(report)
        if args.output:
            write_report(report, args.output)
            print(f"[Report] {args.output}")
        return 0
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    raise SystemExit(main())
