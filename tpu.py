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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="TPU v4 single-TensorCore INT8 GEMM counts or serial timing")
    parser.add_argument("--hardware", default="tpu_v4_reference.json")
    parser.add_argument("--workload", default="tiny_gemm.json")
    parser.add_argument("--config-dir")
    parser.add_argument("--format", help="Explicit workload precision override; only INT8 is supported")
    parser.add_argument("--mode", choices=["count", "timing"], default="timing",
                        help="count: logical actions only; timing (default): counts and task time with explicit assumptions")
    parser.add_argument("--tile", nargs=3, type=int, default=(128, 128, 128), metavar=("M", "K", "N"),
                        help="Logical block upper bounds; timing splits each output block into MXU waves")
    parser.add_argument("--output", default=str(Path(__file__).resolve().parent / "results" / "tpu.json"),
                        help="JSON report path (default: %(default)s; overwrites on success)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        loader = TPUConfigLoader(args.config_dir)
        spec = loader.load_hardware(args.hardware)
        workload = parse_workloads(load_workload_config(args.workload, loader.config_dir), args.format)[0]
        # tile 是逻辑块上限，具体可行尺寸由 tiling.py 按 VMEM 容量确定。
        report = TPUSimulator(spec).simulate(workload, tile=tuple(args.tile), mode=args.mode)
        print_report(report)
        if args.output:
            write_report(report, args.output)
            print(f"[Report] {args.output}")
        return 0
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
