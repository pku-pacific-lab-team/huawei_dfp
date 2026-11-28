"""Bounded software-tile candidates, scored by the native timing kernels."""
from .block_graph import BlockCapacityError


def node_candidates(kernels, phases):
    calls = {call for _, runs, _ in phases for call in runs if call is not None}
    if not calls:
        return None, ()
    first = next(iter(calls))
    if first.kind == "gemm":
        base = tuple(kernels.spec.execution["default_tile_mkn"])
        # Preserve native fractals/supercells. CIM only exposes M grouping.
        arch = kernels.arch
        alignment = (16, 32, 16) if arch == "ascend" else (1, 1, 1)
        if arch == "tsp":
            alignment = (1, kernels.spec.supercell_dim, kernels.spec.supercell_dim)
        limits = [max(c.args[i] for c in calls) for i in range(3)]
        if arch == "tsp":
            limits[1] = min(limits[1], kernels.spec.mxm_plane_dim)
            limits[2] = min(limits[2], kernels.spec.tile_n_capacity_per_hemisphere("INT8"))
        candidates = []
        for axis in range(1 if arch == "tsp_cim" else 3):
            unit = alignment[axis]
            limit = (limits[axis] + unit - 1) // unit * unit
            current = min(base[axis], limit)
            for size in (max(unit, current // 2 // unit * unit), min(limit, 2 * current)):
                if size != current:
                    candidates.append(base[:axis] + (size,) + base[axis + 1:])
        return base, candidates
    if first.kind == "route_phase":
        # Selection is a fixed distributed scalar scan in the current backend.
        return None, ()
    base = kernels.execution.policy["tile_elements"]
    # Requested vector lengths are capped by the native kernel's row/fragment.
    axis = {"embedding": 2, "rope": 2, "append": 2, "cache_component": 1, "combine": 2}.get(first.kind, 0)
    width = max(call.args[axis] for call in calls)
    minimum = 2 if first.kind == "rope" else 1
    return base, [n for n in (max(minimum, base // 2), 2 * base)
                  if min(n, width) != min(base, width)]


def select_node_tile(graph, node, phases):
    base, candidates = node_candidates(graph.k, phases)
    if not candidates:
        graph.execute_node(node, phases)
        return
    trials, best, selected, first_error = [], None, None, None
    for tile in (None, *candidates):
        trial = node.copy()
        entry = {"requested": base if tile is None else tile}
        trials.append(entry)
        try:
            graph.execute_node(trial, phases, tile)
        except BlockCapacityError as exc:
            first_error = first_error or exc
            entry["capacity_error"] = exc.to_dict()
            continue
        entry["cycles"] = trial["local_compute_cycles"]
        if best is None or entry["cycles"] < best["local_compute_cycles"]:
            best, selected = trial, entry["requested"]
    if best is None:
        raise first_error
    node.update(best)
    node["tile_search"] = dict(parameter="tile_mkn" if isinstance(base, tuple) else "tile_elements",
                               default_requested=base, selected_requested=selected, candidates=trials)
