"""Run four architecture CLIs against the Issue #4 two-step Decode acceptance cases."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.verification import discover, fingerprints, output_directory, run_tests, save
from src.models.compute_path import compute_path_cycles


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_snapshot(root):
    # Only inputs, so rerunning against an existing results directory is stable.
    paths = [p for pattern in ("*.py", "src/**/*.py", "scripts/*.py", "tests/__init__.py",
                               "tests/models/**/*.py")
             for p in root.glob(pattern)]
    for folder, names in (("models", ("llama3_70b", "deepseek_r1")),
                          ("hardware", ("ascend910_reference", "tpu_v4_reference", "tsp", "tsp_cim", "cim_macro")),
                          ("execution", ("blocks", "vector", "attention"))):
        paths.extend(root / "configs" / folder / (name + ".json") for name in names)
    files = fingerprints((p for p in paths if p.is_file()), root)
    manifest = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {"sha256": hashlib.sha256(manifest).hexdigest(), "files": files}


def git_version(root):
    commit, dirty = None, None
    if (root / ".git").exists():
        try:
            commit = subprocess.check_output(["git", "rev-parse", "--verify", "HEAD"], cwd=root,
                                             stderr=subprocess.DEVNULL).decode("ascii").strip()
            status = subprocess.check_output(["git", "status", "--porcelain=v1", "--untracked-files=normal"],
                                             cwd=root, stderr=subprocess.DEVNULL)
            dirty = bool(status)
        except (OSError, subprocess.CalledProcessError):
            pass  # The content snapshot also identifies an exported source tree.
    return {"git_commit": commit, "git_dirty": dirty}


def invoke(arch, model, path, mode, *options, reject=False):
    network = path.with_suffix(".communication.json")
    cmd = [
        sys.executable,
        "-B",
        "-X",
        "utf8",
        str(ROOT / f"{arch}.py"),
        "--model",
        model,
        "--mode",
        mode,
        "--output",
        str(path),
        *options,
    ]
    done = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if reject:
        require(done.returncode == 2, "Expected capacity rejection")
        error = json.loads(done.stderr)
        require(
            error["status"] == "insufficient_capacity" and error["required_bytes"] > error["capacity_bytes"],
            "Expected structured capacity coordinates and byte counts",
        )
        return error
    require(done.returncode == 0, done.stderr)
    return json.loads(path.read_bytes()), json.loads(network.read_bytes())


def audit_placement(r, is_r1, small):
    if is_r1:
        tps = [16, 8, 8] + [32] * 57 + [128] if small else [1] * 31
        layer_counts = [1] * 61 if small else [2] * 30 + [1]
        expected_chips = 4942 if small else 31
    else:
        tps = [16] + [4] * 76 + [16] if small else [1] * 3
        layer_counts = [2] + [1] * 76 + [2] if small else [27, 27, 26]
        expected_chips = 336 if small else 3
    chips = {c["chip_id"]: c for c in r["chips"]}
    require(len(chips) == len(r["chips"]) == expected_chips, "Physical chip count")
    start = 0
    for stage, tp, count in zip(r["stages"], tps, layer_counts, strict=True):
        stop = start + count
        require(stage["layers"] == [start, stop], "PP layer boundaries")
        start = stop
        attention = stage.get("attention_chips", stage["chips"])
        require(len(attention) == tp, "Attention placement")
        width = (128 if is_r1 else 64) // tp
        for rank, cid in enumerate(attention):
            lo, hi = rank * width, (rank + 1) * width
            chip = chips[cid]
            require(chip["layers"] == stage["layers"]
                    and chip["heads" if is_r1 else "query_heads"] == [lo, hi], "Per-chip attention heads")
            if not is_r1:
                require(chip["kv_heads"] == [lo // 8, (hi - 1) // 8 + 1], "GQA KV head replicas")
        if is_r1 and stop > 3:
            groups = stage["expert_groups"]
            root = attention[0]
            require(len(groups) == (52 if small else 1) and all(len(group) == 1 for group in groups)
                    and groups[stage["shared_group"]] == [root] == groups[0], "Expert groups and shared expert")
            owners = [0] + [g for g in range(1, 52) for _ in range(5)] if small else [0] * 256
            require(stage["expert_owners"] == owners and set(attention).intersection(g[0] for g in groups) == {root},
                    "Routed expert placement")


def audit(model, r, comm):
    is_r1 = model == "deepseek_r1"
    small = r["architecture"] in ("tsp", "tsp_cim")
    initial = 131072 if is_r1 else 8190
    lengths = [initial + 1, initial + 2]
    require(
        all(r[key] == comm[key] for key in ("source", "model", "architecture", "mode", "request", "stages")),
        "Mismatched compute and communication reports",
    )
    require(r["request"] == dict(batch=1, sequence_length=initial, decode_steps=2)
            and r["effective_lengths"] == lengths and r["cache_capacity_tokens"] == lengths[-1], "Decode lengths")
    audit_placement(r, is_r1, small)
    schedule = {}
    for n in r["schedule"]:
        require(
            n["id"] not in schedule and set(n["depends_on"]) <= schedule.keys(), "Dangling dependency or cycle"
        )
        schedule[n["id"]] = n["depends_on"]
    ancestors = {"model.logits"}
    for name in reversed(schedule):
        if name in ancestors:
            ancestors.update(schedule[name])
    require(
        ancestors == schedule.keys() and "step.0/head/select_token" in schedule["step.1/token_ids"],
        "Token feedback",
    )
    kinds = Counter()
    for e in comm["events"]:
        kinds[e["kind"]] += 1
        require(
            e["id"] in schedule
            and e["producers"] == schedule[e["id"]]
            and e["communication_cycles"] is None
            and e["consumers"]
            and all(e["id"] in schedule[n] for n in e["consumers"]),
            "Communication DAG",
        )
    require(kinds["pp_activation"] == 2 * (len(r["stages"]) - 1) and kinds["decode_feedback"] == 1,
            "PP boundaries and token feedback")
    require((kinds["tp_all_reduce_sum"] > 0) == small, "TP reductions")
    if is_r1:
        require(all((kinds[k] > 0) == small for k in ("ep_dispatch", "ep_return")), "Remote expert traffic")
        require(all(e.get("expert") != "shared" for e in comm["events"]), "Remote shared expert traffic")
    for i, (total, length) in enumerate(zip(r["step_totals"], lengths, strict=True)):
        macs = sum(chip["steps"][i]["useful_macs"] for chip in r["chips"])
        if is_r1:
            rows = [row for name, row in r["routing"].items() if name.startswith(f"step.{i}/")]
            require(
                len(rows) == 58
                and all(row["total_assignments"] == 8 and row["shared_tokens"] == 1 for row in rows),
                "Routed/shared logical invocation counts",
            )
        # Independent B=1 fixtures from the published model dimensions. Growing
        # the context adds one content/position dot and one weighted-value row.
        expected = (
            37_711_970_304 + (length - 128) * 61 * 128 * (512 + 64 + 512)
            if is_r1 else 69_669_486_592 + (length - 128) * 80 * 2 * 8192
        )
        require(
            total["effective_length"] == length and total["logical_useful_macs"] == expected
            and total["executed_useful_macs"] == macs,
            "MAC conservation",
        )
        appended = 0
        for chip in r["chips"]:
            step = chip["steps"][i]
            if is_r1:
                active = [ref for name, ref in step["runs"].items() if name.endswith("/block")]
                repeat = 1
            else:
                active, repeat = [step["block"]], chip["layers"][1] - chip["layers"][0]
            for ref in active:
                row = next(n for n in r["profiles"][ref]["nodes"] if n["id"] == "kv.append")
                require(row["token_interval"] == [length - 1, length], "Per-layer append position")
                appended += repeat * row["appended_bytes"]
        require(
            total["physical_kv_append_bytes"] == appended
            and total["logical_kv_append_bytes"] == (61 * (512 + 64) if is_r1 else 80 * 2 * 8 * 128),
            "KV conservation",
        )
    capacities = [p["capacity"] for p in r["profiles"].values()]
    for cap in capacities:
        require(cap["peak"]["required_bytes"] <= cap["capacity_bytes"], "Chip peak capacity")
        if cap["unit_capacity_bytes"]:
            require(max(cap["unit_peak_bytes"]) <= cap["unit_capacity_bytes"], "Private MEM capacity")
    max_chip = max(c["peak"]["required_bytes"] for c in r["chips"])
    peak, limit = max_chip, capacities[0]["capacity_bytes"]
    if small:
        # R1 also has expert chips that may own weights without being invoked.
        units = r["chips"] if is_r1 else capacities
        peak = max(max(row["unit_peak_bytes"]) for row in units)
        limit = capacities[0]["unit_capacity_bytes"]
    require(peak <= limit, "Default placement capacity including idle chips")
    return dict(
        initial_kv=initial,
        effective_lengths=lengths,
        pp=len(r["stages"]),
        chips=len(r["chips"]),
        max_chip_bytes=max_chip,
        capacity=dict(level="half_chip_MEM" if small else "chip_HBM", peak_bytes=peak, capacity_bytes=limit),
        network_events=dict(kinds),
        step_totals=r["step_totals"],
    )


def case(arch, model, out):
    # Fresh paths prevent an old success from satisfying a broken CLI.
    with TemporaryDirectory(dir=out) as temp:
        folder = Path(temp)
        runs = []
        for mode in ("count", "timing"):
            file = folder / f"{arch}_{model}_{mode}.json"
            r, comm = invoke(arch, model, file, mode)
            result = audit(model, r, comm)
            runs.append((r, comm))
        (count, cm), (timed, tm) = runs
        require(
            count["stages"] == timed["stages"] and count["step_totals"] == timed["step_totals"]
            and cm["events"] == tm["events"], "Mode consistency"
        )
        for key, p in timed["profiles"].items():
            q = count["profiles"][key]
            require(
                p["capacity"] == q["capacity"]
                and p["local_compute_cycles"] is not None
                and q["local_compute_cycles"] is None
                and [n["private_memory"] for n in p["nodes"]] == [n["private_memory"] for n in q["nodes"]]
                and [n["action_counts"] for n in p["nodes"]] == [n["action_counts"] for n in q["nodes"]],
                "Timing/count profile mismatch",
            )
        tuned, network = invoke(arch, model, folder / f"{arch}_{model}_timing_auto_tile.json", "timing", "--auto-tile")
        audit(model, tuned, network)
        require(tuned["schedule"] == timed["schedule"] and network == tm and tuned["step_totals"] == timed["step_totals"]
                and tuned.get("routing") == timed.get("routing"), "Tile search changed the Decode workload")
        search = tuned["tile_search"]
        def replay(report):
            return compute_path_cycles(report["schedule"], tm["events"], report["chips"], report["stages"], report["profiles"])
        original, selected = replay(timed), replay(tuned)
        if arch == "tsp":
            for label, report in (("baseline", timed), ("selected", tuned)):
                peak = max(n["private_memory"]["weight_buffer"]["required_bytes"]
                           for p in report["profiles"].values() for n in p["nodes"] if n["useful_macs"])
                require(peak > 0 and search[label]["peak_private_bytes_per_unit"]["weight_buffer"] == peak,
                        "TSP pending weight peak disagrees with the executed GEMMs")
        require(original == search["baseline"]["critical_path_cycles"] and selected == search["selected"]["critical_path_cycles"]
                and original - selected == search["saved_cycles"] >= 0, "Tile search critical path mismatch")
        if not arch.startswith("tsp"):
            # TP=EP=1: one stream visits every stage sequentially on these placements.
            require(selected == sum(s["local_compute_cycles"] for c in tuned["chips"] for s in c["steps"]),
                    "Serial PP reference time disagrees with the selected path")
        for profile, nodes in search["profile_trials"].items():
            reported = {n["id"]: n for n in tuned["profiles"][profile]["nodes"]}
            for name, trial in nodes.items():
                cycles = [c["cycles"] for c in trial["candidates"] if "cycles" in c]
                winner = next(c for c in trial["candidates"] if c.get("cycles") == min(cycles))
                require(winner["requested"] == trial["selected_requested"], "Wrong tile or tie break")
                if search["selected_policy"] == "per_node":
                    require(reported[name]["local_compute_cycles"] == winner["cycles"], "Selected ledger mismatch")
        result["tile_search"] = {key: search[key] for key in ("baseline", "selected", "selected_policy",
                                                               "saved_cycles", "latency_reduction_fraction")}
        before = (digest(file), digest(file.with_suffix(".communication.json")))
        args = ("--pp", "1", "--tp", "1") + (("--ep", "1") if model == "deepseek_r1" else ())
        result["capacity_rejection"] = invoke(arch, model, file, "timing", *args, reject=True)
        require(
            before == (digest(file), digest(file.with_suffix(".communication.json"))),
            "Failed run overwrote outputs",
        )
        if model == "deepseek_r1":
            clustered, network = invoke(
                arch, model, folder / f"{arch}_{model}_concentrated.json", "count", "--routing", "concentrated"
            )
            audit(model, clustered, network)

            def distribution(r, net):
                layer = "step.0/layer.3"
                rows = r["routing"][layer]["assignments_per_group"]
                size = sum(
                    e["logical_tensor_bytes"] * len(e["deliveries"])
                    for e in net["events"]
                    if e["id"].startswith(layer + "/expert.")
                    and e["kind"] in ("ep_dispatch", "ep_return")
                    and e["expert"] != "shared"
                )
                return dict(busiest_group_tokens=max(rows), remote_payload_bytes=size)

            a, b = distribution(clustered, network), distribution(count, cm)
            small = arch in ("tsp", "tsp_cim")
            # Both routes select expert zero, local to the router. All seven other
            # hits are remote on TSP and local on the two HBM architectures.
            payload = 7 * 7168 * (1 + 4) if small else 0
            require((a["busiest_group_tokens"], b["busiest_group_tokens"]) == ((5, 1) if small else (8, 8))
                    and a["remote_payload_bytes"] == b["remote_payload_bytes"] == payload,
                    "Unexpected routed expert load or delivered bytes")
            result["routing_comparison"] = dict(concentrated=a, spread=b)
        for artifact in folder.iterdir():
            artifact.replace(out / artifact.name)
        return result, timed["stages"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "issue4" / "acceptance")
    args = parser.parse_args(argv)
    try:
        out = output_directory(args.output_dir, "issue4", root=ROOT)
    except ValueError as exc:
        parser.error(str(exc))
    out.mkdir(parents=True, exist_ok=True)
    version = git_version(ROOT)
    before = source_snapshot(ROOT)
    _, counts, passed = run_tests(discover(("tests/models",)), out)
    report = dict(status="passed" if passed else "failed", source_version={**version, **before},
                  model_tests=counts, cases={}, placement_alignment={})
    stages = {}
    for arch in ("ascend", "tpu", "tsp", "tsp_cim"):
        for model in ("llama3_70b", "deepseek_r1"):
            key = f"{arch}/{model}"
            try:
                result, stages[key] = case(arch, model, out)
                report["cases"][key] = dict(status="passed", **result)
            except (ValueError, OSError, KeyError, StopIteration) as exc:
                report["cases"][key] = dict(status="failed", error=str(exc))
                report["status"] = "failed"
            print(key, report["cases"][key]["status"], flush=True)
    for a, b in (("ascend", "tpu"), ("tsp", "tsp_cim")):
        for model in ("llama3_70b", "deepseek_r1"):
            left, right = f"{a}/{model}", f"{b}/{model}"
            aligned = left in stages and right in stages and stages[left] == stages[right]
            report["placement_alignment"][f"{a}+{b}/{model}"] = "passed" if aligned else "failed"
            if not aligned:
                report["status"] = "failed"
    report["snapshot_unchanged_during_run"] = source_snapshot(ROOT) == before
    if not report["snapshot_unchanged_during_run"]:
        report["status"] = "failed"
    save(out / "acceptance.json", report)
    return int(report["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
