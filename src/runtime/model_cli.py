"""Shared whole-model options for the four architecture entry points."""

from contextlib import ExitStack, contextmanager
import json
import os
from pathlib import Path
from shutil import copy2
import sys
from tempfile import NamedTemporaryFile, TemporaryDirectory

from .cli import supplied_option
from ..models.block_graph import BlockCapacityError
from ..models.config import DecodeRequest
from ..models.deepseek_model import run_deepseek_model
from ..models.llama_model import run_llama_model


ROOT = Path(__file__).resolve().parents[2]
MODEL_FLAGS = (
    "--plan",
    "--pp",
    "--tp",
    "--ep",
    "--etp",
    "--routing",
    "--batch",
    "--sequence-length",
    "--decode-steps",
    "--communication-output",
    "--auto-tile",
)


def add_model_arguments(parser):
    group = parser.add_argument_group("whole-model Decode")
    group.add_argument("--model", choices=("llama3_70b", "deepseek_r1"))
    group.add_argument("--plan", type=Path, help="JSON array of PP stages and physical chip groups")
    for flag, description in (("pp", "pipeline stages"), ("tp", "attention/dense ranks"),
                              ("ep", "expert groups"), ("etp", "ranks per expert group")):
        group.add_argument("--" + flag, type=int, help=f"Uniform placement: number of {description}")
    group.add_argument("--routing", help="DeepSeek fixed routing: concentrated, spread, or JSON file")
    group.add_argument("--batch", type=int, default=1)
    group.add_argument("--sequence-length", type=int, help="Initial KV length (defaults to the model baseline)")
    group.add_argument("--decode-steps", type=int, default=2)
    group.add_argument("--communication-output", type=Path)
    group.add_argument("--auto-tile", action="store_true", help="Compare legal software tiles by Decode compute critical path")


@contextmanager
def _lock_report_paths(paths):
    """Hold publisher locks through both replacements and any rollback."""
    with ExitStack() as locks:
        if os.name == "nt":
            for path in paths:
                marker = path.with_name("." + path.name + ".publish.lock")
                if marker in paths:
                    raise ValueError(f"Report path conflicts with publication lock: {marker}")
                # Exclusive creation coordinates processes; Windows removes the
                # marker on close, including when the process terminates.
                fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_TEMPORARY | os.O_WRONLY)
                locks.callback(os.close, fd)
        else:
            import fcntl
            for directory in sorted({p.parent.resolve() for p in paths}):
                fd = os.open(directory, os.O_RDONLY)
                locks.callback(os.close, fd)
                fcntl.flock(fd, fcntl.LOCK_EX)
        yield


def write_model_reports(report, output, network, *, communication_kind=None):
    if output.resolve() == network.resolve():
        raise ValueError("Computation and communication require distinct output paths")
    communication = {
        k: report[k] for k in ("schema_version", "model", "source", "architecture", "mode", "request", "stages")
    }
    communication.update(kind=communication_kind or report["model"] + "_communication",
                         events=report["communication_events"])
    compute = {k: v for k, v in report.items() if k != "communication_events"}
    with ExitStack() as stack:
        files = []
        for path, data in ((network, communication), (output, compute)):
            if path.is_dir():
                raise ValueError(f"Expected a report file: {path}")
            path.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as f:
                tmp = Path(f.name)
                stack.callback(tmp.unlink, missing_ok=True)
                json.dump(data, f, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
                f.write("\n")
            files.append(tmp)
        network_tmp, compute_tmp = files
        targets = [network.absolute(), output.absolute()]
        with _lock_report_paths(targets), TemporaryDirectory(dir=network.parent) as folder:
            backup = Path(folder) / "previous"
            had_network = os.path.lexists(network)
            if had_network:
                copy2(network, backup, follow_symlinks=False)
            network_tmp.replace(network)
            try:
                compute_tmp.replace(output)
            except BaseException:
                if had_network:
                    backup.replace(network)
                else:
                    network.unlink()
                raise


def model_run(args, arch, load_spec, argv=None):
    if args.model is None:
        if any(supplied_option(key, argv) for key in MODEL_FLAGS):
            raise ValueError("Model placement and Decode options require --model")
        return None
    allowed = (*MODEL_FLAGS, "--model", "--hardware", "--config-dir", "--output", "--mode")
    supplied = [t.split("=", 1)[0] for t in (sys.argv[1:] if argv is None else argv) if t.startswith("--")]
    if any(not any(flag.startswith(t) for flag in allowed) for t in supplied):
        raise ValueError("Operator-specific options are incompatible with a whole-model workload")

    def read(path):
        return Path(path).read_text(encoding="utf-8-sig")

    stage_plan = json.loads(read(args.plan)) if args.plan else None
    if args.plan and not isinstance(stage_plan, list):
        raise ValueError("Expected an array of PP stages")
    request = DecodeRequest.for_model(args.model, args.batch, args.sequence_length, args.decode_steps)
    options = dict(stages=stage_plan, pp=args.pp, tp=args.tp, mode=args.mode, auto_tile=args.auto_tile,
                   config_dir=args.config_dir)
    output = (
        Path(args.output)
        if supplied_option("--output", argv)
        else ROOT / "results" / "issue4" / f"{arch}_{args.model}_{args.mode}{'_auto_tile' if args.auto_tile else ''}.json"
    )
    network = args.communication_output or output.with_suffix(".communication.json")
    try:
        if args.model == "deepseek_r1":
            routing = args.routing or "spread"
            if routing not in ("spread", "concentrated"):
                routing = json.loads(read(routing))
            report = run_deepseek_model(
                request, arch, load_spec(), ep=args.ep, etp=args.etp, routing=routing, **options
            )
        else:
            if any(v is not None for v in (args.ep, args.etp, args.routing)):
                raise ValueError("Expert groups and routing require DeepSeek-R1")
            report = run_llama_model(request, arch, load_spec(), **options)
        write_model_reports(report, output, network)
    except BlockCapacityError as exc:
        print(json.dumps(dict(status="insufficient_capacity", **exc.to_dict())), file=sys.stderr)
        return 2
    print(f"{arch} | {args.model} | {len(report['chips'])} chips | {output} | {network}")
    return 0
