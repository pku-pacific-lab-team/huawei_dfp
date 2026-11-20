"""CLI option checks and presentation; no simulation execution."""
import json
from pathlib import Path
import sys

from ..utils.operator_report import print_report, write_report


def add_vector_arguments(parser):
    parser.add_argument("--attention-config", help="Decode/cache execution profile (default: configs/execution/attention.json)")
    parser.add_argument("--block-config", help="Coarse block execution profile (default: configs/execution/blocks.json)")
    parser.add_argument("--details-output", help="Optional separate operator diagnostics JSON (requires operator report)")
    parser.add_argument("--vector-config", help="Vector/composite execution profile (default: configs/execution/vector.json)")
    parser.add_argument("--vector-tile-elements", type=int, help="Vector/composite tile upper bound; local capacity may reduce it")


def supplied_option(name, argv=None):
    supplied = {token.split("=", 1)[0] for token in (sys.argv[1:] if argv is None else argv) if token.startswith("--")}
    return any(name.startswith(token) for token in supplied)


def handles_operator_workload(config, args, argv=None):
    blocks = ("residual", "swiglu", "swiglu_ffn", "rmsnorm_int8", "qkv_projection", "attention_output_projection",
              "rope", "embedding", "vocabulary_projection", "mla_preprocess")
    attention = ("kv_cache", "decode_attention", "mla_cache", "mla_decode_attention")
    if supplied_option("--attention-config", argv) and (not isinstance(config, dict) or config.get("operation") not in attention):
        raise ValueError("--attention-config requires kv_cache/decode_attention")
    if supplied_option("--block-config", argv) and (not isinstance(config, dict) or config.get("operation") not in (*blocks, "decode_attention", "mla_decode_attention")):
        raise ValueError("--block-config requires a materialized block or decode_attention")
    if not isinstance(config, dict) or config.get("operation", "gemm") not in ("vector", "rmsnorm", "softmax", *blocks, *attention):
        if args.details_output and args.report_format != "operator":
            raise ValueError("--details-output requires --report-format operator for GEMM")
        if args.report_format == "operator" and not args.details_output and any(supplied_option(name, argv) for name in (
                "--trace", "--compare-load-policies", "--compare-mxm", "--compare-loops", "--sweep-m")):
            raise ValueError("Trace/comparison output requires --details-output with an operator report")
        if any(supplied_option(name, argv) for name in ("--vector-config", "--vector-tile-elements")):
            raise ValueError("Vector options require operation=vector/rmsnorm/softmax")
        return False
    forbidden = {"--format", "--formats", "--tile", "--tile-m", "--l1-k", "--buffers", "--trace",
                 "--schedule", "--cmem-buffers", "--cmem-mode", "--cmem-groups", "--loop-order",
                 "--load-overlap", "--compare-load-policies", "--compare-mxm", "--compare-loops", "--sweep-m"}
    rejected = {name for name in forbidden if supplied_option(name, argv)}
    if rejected:
        raise ValueError("GEMM-only options are not valid for Vector/composite: " + ", ".join(sorted(rejected)))
    if supplied_option("--report-format", argv) and args.report_format != "operator":
        raise ValueError("Vector/composite uses the operator report; omit --report-format or select operator")
    if getattr(args, "scope", "single_hem") != "single_hem":
        raise ValueError("Vector/composite chip execution is not modeled")
    return True


def operator_options(args, architecture):
    policy = {"tile_elements": args.vector_tile_elements, "include_details": bool(args.details_output)}
    if args.block_config is not None:
        policy["block_execution_file"] = args.block_config
    if args.attention_config is not None:
        policy["attention_execution_file"] = args.attention_config
    if architecture in {"tsp", "tsp_cim"}:
        policy.update(mem_slices=args.mem_slices, streams=args.streams,
                      reserved_bytes=args.reserved_bytes, hemisphere=getattr(args, "hemisphere", 0))
    return policy


def write_operator_result(result, args, argv=None):
    output = args.output if supplied_option("--output", argv) else (
        Path(__file__).resolve().parents[2] / "results" / f"{result.architecture}_{result.operator.kind}.json")
    write_operator_results([result], args, output=output)


def write_operator_results(results, args, *, output=None):
    """Serialize both views before writes; diagnostics never overwrite the summary."""
    output = args.output if output is None else output
    details_output = args.details_output
    if output and details_output and Path(output).resolve() == Path(details_output).resolve():
        raise ValueError("Summary and details output must use different paths")
    reports = [result.to_dict() for result in results]
    details = [result.to_detailed_dict() for result in results] if details_output else None
    pack = lambda rows: rows[0] if len(rows) == 1 else {"runs": rows}
    report = pack(reports)
    json.dumps(report, allow_nan=False)
    if details is not None:
        json.dumps(pack(details), allow_nan=False)
    for row in reports:
        print_report(row)
    if details_output:
        write_report(pack(details), details_output)
        print(f"[Details] {details_output}")
    if output:
        write_report(report, output)
        print(f"[Report] {output}")
