"""Capacity-first placement and execution of the 61-layer DeepSeek-R1 decoder."""

from dataclasses import asdict
from functools import lru_cache
from collections import Counter

from .block_graph import BlockCapacityError, widths
from .config import SOURCES, load_model
from .deepseek_block import MLARank
from .deepseek_experts import ExpertOperators, R1Kernels
from .deepseek_placement import (
    attention_shape, expert_shape, placements, recommended_stages, route_table, span, uniform_stages,
)
from .deepseek_schedule import decode_schedule
from .llama_model import ModelProfiles, resident
from .projection import _column_ranges


class R1Profiles(ModelProfiles):
    def __init__(self, kernels, c, request):
        super().__init__(kernels, c, request)
        self.alloc = lru_cache(maxsize=None)(self._allocation)

    def _shape(self, kind, *dims):
        if kind in ("embedding", "head"):
            return super()._shape(kind, *dims)
        if kind in ("dense", "mla"):
            return MLARank(self.k, self.config, self.request, *dims).build()
        b, *args = dims
        graph = ExpertOperators(self.k, self.config, b)
        return getattr(graph, kind)(*args)

    def _allocation(self, shape, role):
        return resident(self.shapes(*shape), role)

    def bind_chip(self, shape, chip, location):
        balance = chip["resident_bytes_per_unit"].copy()
        for role in ("weight", "kv_cache", "rope_table"):
            own = self.alloc(shape, role)
            if any(own):
                group = shape[0] + "_weights" if role == "weight" else role
                values = [a - b for a, b in zip(balance[group], own, strict=True)]
                if min(values) < 0:
                    raise ValueError(f"Physical allocation exceeds inventory at {location}")
                balance[group] = values
        return self.bind_reserved(shape, balance, chip["chip_id"], location)


def rank_layouts(c, stage):
    att = stage["attention_chips"]
    layouts = {chip: {} for chip in stage["chips"]}
    for field, width in (
        ("heads", c["num_attention_heads"]),
        ("dense_columns", c["intermediate_size"]),
        ("embedding_columns", c["hidden_size"]),
        ("vocab_columns", c["vocab_size"]),
    ):
        for chip, interval in zip(att, _column_ranges(width, len(att)), strict=True):
            layouts[chip][field] = list(interval)
    for i, group in enumerate(stage["expert_groups"]):
        for chip, interval in zip(group, _column_ranges(c["moe_intermediate_size"], len(group)), strict=True):
            layouts[chip].update(expert_group=i, expert_columns=list(interval), expert_leader=chip == group[0])
    return layouts


def inventory(c, request, stages, profiles):
    chips = {}
    b, h, k = request.batch, c["hidden_size"], c["num_experts_per_tok"]
    final = request.sequence_length + request.decode_steps
    for s, stage in enumerate(stages):
        first, end = stage["layers"]
        dense = max(0, min(end, c["first_k_dense_replace"]) - first)
        moe = end - first - dense
        owner_counts = Counter(stage["expert_owners"])
        root = stage["attention_chips"][0]
        for cid, layout in rank_layouts(c, stage).items():
            chip = dict(chip_id=cid, stage=s, layers=[first, end], **layout)
            sizes = {}

            def add(name, values, count=1):
                if count:
                    row = sizes.setdefault(name, [0] * profiles.k.units)
                    for i, n in enumerate(values):
                        row[i] += count * n

            def weight(shape, count=1):
                add(shape[0] + "_weights", profiles.alloc(shape, "weight"), count)

            if "heads" in chip:
                for is_dense, count in ((True, dense), (False, moe)):
                    if count:
                        shape = attention_shape(chip, stage, final, is_dense)
                        weight(shape, count)
                        add("kv_cache", profiles.alloc(shape, "kv_cache"), count)
                add("rope_table", profiles.alloc(shape, "rope_table"))
                if s == 0:
                    weight(("embedding", span(chip, "embedding_columns")))
                if s == len(stages) - 1:
                    weight(("head", span(chip, "vocab_columns"), cid == root, False))
            if moe and cid == root:
                weight(("router", b), moe)
                # Stable receive/send slots, separate from each kernel's local
                # input/output backing. Reused across all MoE layers and steps.
                for elements, byte in (
                    (b * h, 1),
                    (b * h, 1),
                    (b * k, 4),
                    (b * k, 4),
                    (b * k * h, 4),
                    (b * h, 4),
                ):
                    add(
                        "moe_exchange_buffers",
                        [profiles.k.size(n * byte) for n in widths(elements, profiles.k.units)],
                    )
            if "expert_group" in chip:
                group = chip["expert_group"]
                count = owner_counts[group] + int(group == stage["shared_group"])
                chip.update(
                    resident_routed_experts_per_layer=owner_counts[group],
                    owns_shared_expert=group == stage["shared_group"],
                )
                weight(expert_shape(chip, stage, 1), moe * count)
            total = list(map(sum, zip(*sizes.values()))) if sizes else [0] * profiles.k.units
            unit = total.index(max(total)) if profiles.k.resident else None
            required = sum(total) if unit is None else total[unit]
            if required > profiles.k.capacity:
                raise BlockCapacityError(
                    cid,
                    unit,
                    "HBM" if unit is None else "MEM",
                    f"stage.{s}/resident",
                    required,
                    profiles.k.capacity,
                    {n: sum(v) if unit is None else v[unit] for n, v in sizes.items()},
                )
            chip.update(
                resident_bytes_per_unit=sizes,
                resident_bytes=sum(total),
                unit_peak_bytes=total,
                peak=dict(required_bytes=sum(total), node="resident"),
                steps=[dict(runs={}) for _ in range(request.decode_steps)],
            )
            chips[cid] = chip
    return chips


def reconcile(c, request, stages, length, ledger):
    b, h = request.batch, c["hidden_size"]
    q, r, n, d, v = (
        c[key] for key in ("q_lora_rank", "kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim")
    )
    heads = c["num_attention_heads"]
    common = h * (q + r + d)
    by_head = q * (n + d) + n * r + (2 * r + d) * length + r * v + v * h
    macs, physical_kv = b * h * c["vocab_size"], 0
    for stage in stages:
        for layer in range(*stage["layers"]):
            tp = len(stage["attention_chips"])
            macs += b * (common * tp + heads * by_head)
            physical_kv += b * (r + d) * tp
            if layer < c["first_k_dense_replace"]:
                macs += b * 3 * h * c["intermediate_size"]
            else:
                macs += b * (
                    h * c["n_routed_experts"]
                    + 3 * h * c["moe_intermediate_size"] * (c["num_experts_per_tok"] + 1)
                )
    if macs != ledger:
        raise ValueError(f"R1 MAC conservation failed at length {length}: {ledger} vs {macs}")
    logical_kv = b * (r + d) * c["num_hidden_layers"]
    copies = sum((s["layers"][1] - s["layers"][0]) * (len(s["attention_chips"]) - 1) for s in stages)
    moe = c["num_hidden_layers"] - c["first_k_dense_replace"]
    return dict(
        effective_length=length,
        executed_useful_macs=ledger,
        logical_useful_macs=ledger - b * common * copies,
        replica_projection_macs=b * common * copies,
        logical_kv_append_bytes=logical_kv,
        physical_kv_append_bytes=physical_kv,
        routed_assignments=b * c["num_experts_per_tok"] * moe,
        shared_expert_tokens=b * moe,
    )


def _run_deepseek(c, request, architecture, spec, stages, routing, mode, auto_tile=False, *, config_dir=None):
    if mode not in ("count", "timing") or c["n_shared_experts"] != 1 or c["moe_layer_freq"] != 1:
        raise ValueError("Expected count/timing and one shared expert on every post-dense layer")
    lengths = request.effective_lengths(c)
    stages = placements(c, stages)
    routes = route_table(c, request, routing)
    planner = R1Profiles(R1Kernels(architecture, spec, "count", c, config_dir=config_dir), c, request)
    chips = inventory(c, request, stages, planner)
    schedule, events, ledgers = decode_schedule(c, request, stages, chips, planner, routes)
    profiles, comparison = planner.evaluate(mode, auto_tile, schedule, events, chips.values(), stages)
    totals = [0] * request.decode_steps
    for chip in chips.values():
        for i, step in enumerate(chip["steps"]):
            cycles, macs = 0, 0
            for name, p in step["runs"].items():
                row = profiles[p]
                cycles += row["local_compute_cycles"] or 0
                macs += row["useful_macs"]
                cap = row["capacity"]
                chip["unit_peak_bytes"] = list(map(max, chip["unit_peak_bytes"], cap["unit_peak_bytes"]))
                if cap["peak"]["required_bytes"] > chip["peak"]["required_bytes"]:
                    chip["peak"] = dict(
                        required_bytes=cap["peak"]["required_bytes"], node=name + "/" + cap["peak"]["node"]
                    )
            totals[i] += macs
            step.update(
                useful_macs=macs,
                local_compute_cycles=cycles if mode == "timing" else None,
                local_compute_seconds=cycles / planner.k.clock
                if mode == "timing" and planner.k.clock
                else None,
            )
    return dict(
        schema_version=1,
        kind="deepseek_decode_model",
        model="deepseek_r1",
        architecture=architecture,
        mode=mode,
        request=asdict(request),
        effective_lengths=list(lengths),
        cache_capacity_tokens=lengths[-1],
        resources=planner.k.resources,
        stages=stages,
        chips=list(chips.values()),
        profiles=profiles,
        schedule=schedule,
        communication_events=events,
        routing=ledgers,
        intra_chip_handoff_cycles=0,
        capacity_bytes=planner.k.capacity * (planner.k.units if planner.k.resident else 1),
        unit_capacity_bytes=planner.k.capacity if planner.k.resident else None,
        step_totals=[reconcile(c, request, stages, length, total) for length, total in zip(lengths, totals)],
        latency_scope="local compute; inter-chip communication and token selection pending",
        **comparison,
    )


def run_deepseek_model(
    request,
    architecture,
    spec,
    *,
    stages=None,
    pp=None,
    tp=None,
    ep=None,
    etp=None,
    routing="spread",
    mode="timing",
    auto_tile=False,
    config_dir=None,
):
    c = load_model("deepseek_r1")
    if stages is None and all(v is None for v in (pp, tp, ep, etp)):
        stages = recommended_stages(c, architecture)
    elif stages is None:
        stages = uniform_stages(c, architecture, pp, tp, ep, etp)
    elif any(v is not None for v in (pp, tp, ep, etp)):
        raise ValueError("Use a stage plan or uniform parallelism degrees")
    return dict(
        _run_deepseek(c, request, architecture, spec, stages, routing, mode, auto_tile, config_dir=config_dir),
        source=dict(SOURCES["deepseek_r1"]),
    )
