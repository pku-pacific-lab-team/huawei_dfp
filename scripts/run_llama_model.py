"""Simulate full LLaMA3-70B Decode with explicit PP stages and TP chip groups."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.block_graph import BlockCapacityError
from src.models.chip import load_hardware
from src.models.config import DEFAULT_INITIAL_KV, DecodeRequest
from src.models.llama_model import run_llama_model
from src.runtime.backends import BACKENDS
from src.runtime.model_cli import write_model_reports


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=tuple(BACKENDS), required=True)
    parser.add_argument("--plan", type=Path, help="JSON list of stages with half-open layers and ordered chip IDs")
    parser.add_argument("--pp", type=int, help="Number of stages for uniform placement")
    parser.add_argument("--tp", type=int, help="Chips per stage for uniform placement")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=DEFAULT_INITIAL_KV["llama3_70b"])
    parser.add_argument("--decode-steps", type=int, default=2)
    parser.add_argument("--mode", choices=("timing", "count"), default="timing")
    parser.add_argument("--auto-tile", action="store_true", help="Compare software tiles by Decode compute critical path")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--communication-output", type=Path)
    args = parser.parse_args(argv)
    suffix = "_auto_tile" if args.auto_tile else ""
    output = args.output or ROOT / "results" / "issue4" / f"{args.architecture}_llama_model_{args.mode}{suffix}.json"
    network = args.communication_output or output.with_suffix(".communication.json")
    try:
        if output.resolve() == network.resolve():
            raise ValueError("Computation and communication outputs need distinct paths")
        stages = None
        if args.plan:
            stages = json.loads(args.plan.read_text(encoding="utf-8-sig"))
            if not isinstance(stages, list):
                raise ValueError("Expected a JSON list of PP stages")
        request = DecodeRequest(args.batch, args.sequence_length, args.decode_steps)
        report = run_llama_model(request, args.architecture, load_hardware(args.architecture),
                                 stages=stages, pp=args.pp, tp=args.tp, mode=args.mode, auto_tile=args.auto_tile)
        write_model_reports(report, output, network, communication_kind="llama_communication")
    except BlockCapacityError as exc:
        parser.exit(2, json.dumps(dict(status="insufficient_capacity", **exc.to_dict())) + "\n")
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Model simulation failed: {exc}\n")
    print(f"{args.architecture} | {len(report['stages'])} stages | {len(report['chips'])} chips | "
          f"{len(report['communication_events'])} communication events | {output} | {network}")


if __name__ == "__main__":
    main()
