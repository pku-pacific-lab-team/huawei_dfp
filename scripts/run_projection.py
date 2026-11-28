"""Run one Issue #4 column-parallel dense FFN projection on a selected backend."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.chip import load_hardware
from src.models.config import DecodeRequest, SOURCES
from src.models.projection import run_up_projection
from src.runtime.backends import BACKENDS
from src.utils.operator_report import write_report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=tuple(BACKENDS), required=True)
    parser.add_argument("--model", choices=tuple(SOURCES), default="llama3_70b")
    parser.add_argument("--layer", type=int, default=0, help="zero-based dense layer index")
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, help="valid KV length before step one (model default)")
    parser.add_argument("--decode-steps", type=int, default=2)
    parser.add_argument("--mode", choices=("count", "timing"), default="timing")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        request = DecodeRequest.for_model(args.model, args.batch, args.sequence_length, args.decode_steps)
        report = run_up_projection(args.model, request, args.architecture, load_hardware(args.architecture),
                                   tp=args.tp, layer=args.layer, mode=args.mode)
        output = args.output or ROOT / "results" / "issue4" / f"{args.architecture}_{args.model}_up_{args.mode}.json"
        write_report(report, output)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Projection failed: {exc}\n")
    print(f"{args.architecture} | TP={args.tp} | {report['useful_macs_per_step']} useful MAC/step | {output}")


if __name__ == "__main__":
    main()
