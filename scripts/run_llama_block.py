"""Simulate one LLaMA3-70B decoder layer on a tensor-parallel chip group."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.chip import load_hardware
from src.models.block_graph import BlockCapacityError
from src.models.config import DEFAULT_INITIAL_KV, DecodeRequest
from src.models.llama_block import run_llama_block
from src.runtime.backends import BACKENDS
from src.utils.operator_report import write_report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=tuple(BACKENDS), required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=DEFAULT_INITIAL_KV["llama3_70b"])
    parser.add_argument("--decode-steps", type=int, default=2)
    parser.add_argument("--mode", choices=("timing", "count"), default="timing")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        request = DecodeRequest(args.batch, args.sequence_length, args.decode_steps)
        report = run_llama_block(request, args.architecture, load_hardware(args.architecture),
                                 tp=args.tp, layer=args.layer, mode=args.mode)
        output = args.output or ROOT / "results" / "issue4" / f"{args.architecture}_llama_block_{args.mode}.json"
        write_report(report, output)
    except BlockCapacityError as exc:
        parser.exit(2, json.dumps(dict(status="insufficient_capacity", **exc.to_dict())) + "\n")
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Decoder block failed: {exc}\n")
    print(f"{args.architecture} | TP={args.tp} | layer {args.layer} | {args.decode_steps} steps | {output}")


if __name__ == "__main__":
    main()
