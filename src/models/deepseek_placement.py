"""Physical PP/attention TP/expert groups and reproducible per-token routing."""

from collections import Counter

from ..operators.moe import MoERoutingPlan
from .projection import _column_ranges


def uniform_stages(c, arch, pp=None, tp=None, ep=None, etp=None):
    small = arch in ("tsp", "tsp_cim")
    defaults = (61, 16, 128, 1) if small else (4, 4, 8, 1)
    caps = (c["num_hidden_layers"], c["num_attention_heads"], c["n_routed_experts"], c["moe_intermediate_size"])
    degrees = []
    for value, default, cap in zip((pp, tp, ep, etp), defaults, caps):
        value = min(default, cap) if value is None else value
        if type(value) is not int or not 1 <= value <= cap:
            raise ValueError("Invalid PP, TP, EP or expert TP degree")
        degrees.append(value)
    pp, tp, ep, etp = degrees
    stages, offset = [], 0
    for start, end in _column_ranges(c["num_hidden_layers"], pp):
        moe = end > c["first_k_dense_replace"]
        chips = list(range(offset, offset + (max(tp, ep * etp) if moe else tp)))
        groups = [chips[i * etp : (i + 1) * etp] for i in range(ep)] if moe else []
        stages.append(dict(layers=[start, end], chips=chips, attention_chips=chips[:tp], expert_groups=groups))
        offset += len(chips)
    return stages


def recommended_stages(c, arch):
    if arch not in ("tsp", "tsp_cim"):
        return uniform_stages(c, arch, pp=31, tp=1, ep=1, etp=1)

    layers = c["num_hidden_layers"]
    dense_layers = c["first_k_dense_replace"]
    # Group zero shares the router chip with one routed expert and the shared expert.
    # The remaining groups hold up to five whole routed experts on a dedicated chip.
    owners = [0] + [1 + (expert - 1) // 5 for expert in range(1, c["n_routed_experts"])]
    dedicated_count = max(owners)  # Group numbers 1 through dedicated_count.
    stages, offset = [], 0
    for layer in range(layers):
        if layer == 0:
            tp = 16  # The first dense stage also owns the embedding.
        elif layer == layers - 1:
            tp = 128
        else:
            tp = 8 if layer < dense_layers else 32
        attention = list(range(offset, offset + tp))
        stage = dict(layers=[layer, layer + 1], chips=attention, attention_chips=attention)
        if layer >= dense_layers:
            dedicated = list(range(offset + tp, offset + tp + dedicated_count))
            stage.update(
                chips=attention + dedicated,
                expert_groups=[[chip] for chip in [attention[0], *dedicated]],
                expert_owners=owners.copy(),
                shared_group=0,
            )
        stages.append(stage)
        offset += len(stage["chips"])
    return stages


def placements(c, stages):
    if not isinstance(stages, list) or not stages:
        raise ValueError("Expected a nonempty PP stage list")
    out, used, start = [], set(), 0
    allowed = {"layers", "chips", "attention_chips", "expert_groups", "expert_owners", "shared_group"}

    def ids(group):
        return (
            isinstance(group, list)
            and bool(group)
            and all(type(i) is int and i >= 0 for i in group)
            and len(set(group)) == len(group)
        )

    for item in stages:
        if not isinstance(item, dict) or set(item) - allowed:
            raise ValueError("Invalid stage fields")
        interval, chips = item.get("layers"), item.get("chips")
        if (
            not isinstance(interval, list)
            or len(interval) != 2
            or any(type(x) is not int for x in interval)
            or interval[0] != start
            or not start < interval[1] <= c["num_hidden_layers"]
        ):
            raise ValueError("PP requires consecutive, ordered, nonempty layers")
        att = item.get("attention_chips", chips)
        if (
            not ids(chips)
            or used.intersection(chips)
            or not ids(att)
            or not set(att) <= set(chips)
            or len(att)
            > min(c["num_attention_heads"], c["intermediate_size"], c["hidden_size"], c["vocab_size"])
        ):
            raise ValueError("Invalid physical or attention chip group")
        moe = interval[1] > c["first_k_dense_replace"]
        groups = item.get("expert_groups", [[i] for i in chips] if moe else [])
        if not isinstance(groups, list) or any(
            not ids(g) or len(g) > c["moe_intermediate_size"] for g in groups
        ):
            raise ValueError("Invalid expert group")
        flat = [i for group in groups for i in group]
        if (
            len(set(flat)) != len(flat)
            or set(att) | set(flat) != set(chips)
            or bool(groups) != moe
            or len(groups) > c["n_routed_experts"]
        ):
            raise ValueError("Invalid EP placement or unused chip")
        owners, shared = [], None
        if moe:
            owners = item.get("expert_owners")
            if owners is None:
                owners = [
                    i
                    for i, (a, b) in enumerate(_column_ranges(c["n_routed_experts"], len(groups)))
                    for _ in range(a, b)
                ]
            shared = item.get("shared_group", len(groups) - 1)
            if (
                not isinstance(owners, list)
                or len(owners) != c["n_routed_experts"]
                or any(type(x) is not int or not 0 <= x < len(groups) for x in [*owners, shared])
            ):
                raise ValueError("Each expert requires exactly one owning group")
        elif item.get("expert_owners") or item.get("shared_group") is not None:
            raise ValueError("Dense stages do not own experts")
        out.append(
            dict(
                layers=interval,
                chips=chips,
                attention_chips=att,
                expert_groups=groups,
                expert_owners=owners,
                shared_group=shared,
            )
        )
        used.update(chips)
        start = interval[1]
    if start != c["num_hidden_layers"]:
        raise ValueError("PP plan must cover all layers")
    return out


def route_table(c, request, scenario):
    experts, k, groups = c["n_routed_experts"], c["num_experts_per_tok"], c["n_group"]
    width, selected = experts // groups, c["topk_group"]

    def checked(rows):
        plan = MoERoutingPlan(experts, rows)
        if (
            plan.batch != request.batch
            or plan.top_k != k
            or any(len({x // width for x in row}) > selected for row in plan.assignments)
        ):
            raise ValueError("Route must match batch, top-k and selected group limits")
        return plan

    if isinstance(scenario, str) and scenario in ("concentrated", "spread"):
        selected_groups = [i * groups // selected for i in range(selected)]
        candidates = [g * width + j for g in selected_groups for j in range(width)]
        row = (
            list(range(k))
            if scenario == "concentrated"
            else [candidates[i * len(candidates) // k] for i in range(k)]
        )
        scenario = {"assignments": [row for _ in range(request.batch)]}
    if (
        not isinstance(scenario, dict)
        or "assignments" not in scenario
        or set(scenario) - {"assignments", "overrides"}
    ):
        raise ValueError("Expected concentrated, spread, or a fixed-routing JSON scenario")
    base = checked(scenario["assignments"])
    overrides = {}
    if not isinstance(scenario.get("overrides", []), list):
        raise ValueError("Expected routing overrides list")
    for item in scenario.get("overrides", []):
        if not isinstance(item, dict) or set(item) != {"step", "layer", "assignments"}:
            raise ValueError("Invalid step/layer route override")
        s, l = item["step"], item["layer"]
        if (
            type(s) is not int
            or not 0 <= s < request.decode_steps
            or type(l) is not int
            or not c["first_k_dense_replace"] <= l < c["num_hidden_layers"]
            or (s, l) in overrides
        ):
            raise ValueError("Invalid or duplicate step/layer in routing scenario")
        overrides[s, l] = checked(item["assignments"])
    return {
        (s, l): overrides.get((s, l), base)
        for s in range(request.decode_steps)
        for l in range(c["first_k_dense_replace"], c["num_hidden_layers"])
    }


def assignments(stage, plan):
    ledgers, load = [], Counter()
    for expert, slots in enumerate(plan.slots_by_expert):
        if slots:
            group = stage["expert_owners"][expert]
            ledgers.append(
                dict(expert=expert, group=group, slots=[list(pair) for pair in slots], tokens=len(slots))
            )
            load[group] += len(slots)
    return dict(
        assignments=[list(row) for row in plan.assignments],
        active=ledgers,
        assignments_per_group=[load[i] for i in range(len(stage["expert_groups"]))],
        total_assignments=sum(load.values()),
        shared_tokens=plan.batch,
    )


def span(chip, key):
    a, b = chip[key]
    return b - a


def attention_shape(chip, stage, length, dense):
    return (
        "dense" if dense else "mla",
        length,
        len(stage["attention_chips"]),
        span(chip, "heads"),
        span(chip, "dense_columns") if dense else 0,
    )


def expert_shape(chip, stage, b):
    group = stage["expert_groups"][chip["expert_group"]]
    return ("expert", b, span(chip, "expert_columns"), len(group), chip["expert_leader"])
