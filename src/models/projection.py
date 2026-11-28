"""Column-parallel dense FFN up projection, using native local GEMM services."""
from collections import Counter
from dataclasses import asdict

from ..operators.primitives import Gemm
from ..runtime.simulation import simulate_gemm
from .chip import chip_resources
from .config import SOURCES, load_model


def _column_ranges(width, parts):
    """Balanced, nonempty half-open column intervals."""
    if type(parts) is not int or not 1 <= parts <= width:
        raise ValueError("Column partition count must be a positive integer <= output width")
    base, tail = divmod(width, parts)
    start = 0
    for index in range(parts):
        stop = start + base + int(index < tail)
        yield start, stop
        start = stop


def _chip_profile(slices, local_runs):
    copies = Counter(stop - start for start, stop in slices)
    actions = {}
    cycles, seconds = [], []
    hbm_required = 0
    hbm_capacity = None
    for n, instances in copies.items():
        result = local_runs[str(n)]
        cycles.append(result["cycle_count"])
        seconds.append(result["latency_s"])
        # Each unit gets a separate native input/weight/output allocation.
        hbm = result["resource_requirements"].get("hbm")
        if hbm:
            hbm_required += instances * hbm["required_bytes"]
            hbm_capacity = hbm["capacity_bytes"]
        for key, row in result["action_counts"].items():
            if key not in actions:
                actions[key] = {**row, "scope": "chip", "native_scope": row["scope"], "count": 0}
            total = actions[key]["count"]
            if row["count"] is None or total is None:
                actions[key].update(count=None, status="unavailable")
            else:
                actions[key]["count"] += instances * row["count"]
    if hbm_capacity is not None and hbm_required > hbm_capacity:
        raise ValueError(f"Projection requires {hbm_required} B across units, chip HBM has {hbm_capacity} B")
    return {
        "unit_slices": [{"unit_id": i, "chip_columns": [lo, hi], "local_run": str(hi - lo)}
                        for i, (lo, hi) in enumerate(slices)],
        "instances_per_local_run": {str(n): count for n, count in copies.items()},
        "useful_macs": sum(local_runs[str(n)]["useful_macs"] * count for n, count in copies.items()),
        "cycle_count": None if None in cycles else max(cycles),
        "latency_s": None if None in seconds else max(seconds),
        "action_counts": actions,
        "hbm": None if hbm_capacity is None else {"required_bytes": hbm_required, "capacity_bytes": hbm_capacity},
    }


def _place_projection(m, k, n, tp, units):
    """Place weights/output by column; all A rows are available at every unit."""
    if type(tp) is not int or tp < 1 or n // tp < units:
        raise ValueError("TP must be positive and leave at least one output column per execution unit")
    chips = []
    slices_by_width = {}
    for rank, (start, stop) in enumerate(_column_ranges(n, tp)):
        width = stop - start
        if width not in slices_by_width:
            slices_by_width[width] = list(_column_ranges(width, units))
        chips.append({"chip_id": rank, "global_output_columns": [start, stop], "weight_shape": [k, width],
                      "output_shape": [m, width], "chip_profile": str(width)})
    return chips, slices_by_width


def run_up_projection(model, request, architecture, spec, *, tp, layer=0, mode="count"):
    """Describe every TP rank and simulate each distinct local shape once per call."""
    config = load_model(model)
    lengths = request.effective_lengths(config)
    dense_layers = config.get("first_k_dense_replace", config["num_hidden_layers"])
    if type(layer) is not int or not 0 <= layer < dense_layers:
        raise ValueError(f"Expected a dense FFN layer in [0, {dense_layers})")
    if mode not in ("count", "timing"):
        raise ValueError("mode must be count or timing")
    resources = chip_resources(architecture, spec)
    units = resources["units_per_chip"]
    m, k, n = request.batch, config["hidden_size"], config["intermediate_size"]
    chips, slices_by_width = _place_projection(m, k, n, tp, units)
    local_widths = {stop - start for slices in slices_by_width.values() for start, stop in slices}
    local_runs = {}
    for local_n in sorted(local_widths):
        op = Gemm(m, k, local_n)
        result = simulate_gemm(op, architecture, spec, mode=mode)
        local_runs[str(local_n)] = {**result.to_dict(), "useful_macs": op.useful_macs}
    profiles = {str(width): _chip_profile(slices, local_runs) for width, slices in slices_by_width.items()}
    return {
        "schema_version": 1, "kind": "dense_up_projection", "architecture": architecture, "mode": mode,
        "model": model, "source": dict(SOURCES[model]), "layer": layer,
        "request": asdict(request), "effective_lengths": list(lengths),
        "predecessors": [f"layers.{layer}.ffn_norm"],
        "global_operator": Gemm(m, k, n).spec.to_dict(), "tp": tp,
        "resources": resources, "chips": chips, "chip_profiles": profiles, "local_runs": local_runs,
        "useful_macs_per_step": sum(profiles[c["chip_profile"]]["useful_macs"] for c in chips),
        "execution": {"scope": "one_dense_up_projection_per_decode_step",
                      "partition_axis": "output_columns", "input_placement": "replicated_per_chip_and_unit",
                      "unit_execution": "concurrent_with_native_bandwidth_shares",
                      "intra_chip_handoff_cycles": 0, "inter_chip_communication": "not_evaluated",
                      "weight_reuse_between_steps": "cold_native_run_per_step"},
    }
