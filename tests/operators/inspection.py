"""Join requested diagnostics for model tests; this is not a report format.

The existing hand-calculated tests inspect schedules, exact allocations and
action provenance. Compact-result serialization is tested independently.
"""
from src.operators.connections import io_contract
import json


def inspect_result(result):
    native = result.native_report
    return {**result.to_dict(), "native_report": native,
            "latency": {"status": result.timing_status, "cycle_count": result.cycle_count,
                        "latency_s": result.latency_s},
            "execution_contract": native["execution_contract"],
            "resource_requirements": {"memory": native["memory"]},
            "action_ledger": result.action_ledger.to_dict(), "io_contract": io_contract(result)}


def inspect_cli_report(summary_path, details_path):
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    native = json.loads(details_path.read_text(encoding="utf-8"))["native_report"]
    return {**summary, "native_report": native, "execution_contract": native["execution_contract"],
            "latency": {"status": summary["timing_status"], "cycle_count": summary["cycle_count"],
                        "latency_s": summary["latency_s"]}}
