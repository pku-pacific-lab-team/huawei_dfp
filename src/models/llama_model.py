"""Place all LLaMA layers, verify resident storage, then execute local profiles."""
from collections import Counter
from copy import copy, deepcopy
from dataclasses import asdict
from functools import lru_cache

from .block_graph import BlockCapacityError
from .config import SOURCES, load_model
from .compute_path import tile_metrics
from .decoder_kernels import DecoderKernels
from .llama_block import RankBlock, layer_work, tp_layout
from .llama_placement import Endpoints, recommended_plan, uniform_plan, validate_plan
from .llama_schedule import model_schedule
from .projection import _column_ranges


def resident(graph, role):
    vectors = [t["bytes_per_unit"] for t in graph.tensors.values() if t["role"] == role]
    return [sum(lane) for lane in zip(*vectors)] if vectors else [0]*graph.k.units


class ModelProfiles:
    def __init__(self, kernels, config, request):
        self.k, self.config, self.request = kernels, config, request
        self.shapes = lru_cache(maxsize=None)(self._shape)
        self.by_key, self.graphs = {}, {}

    def _shape(self, kind, *dims):
        if kind == "block":
            tp, q, kv, width, length = dims
            layout = dict(chip_id=0, query_heads=[0,q], kv_heads=[0,kv], ffn_columns=[0,width])
            return RankBlock(self.k, layout, self.config, self.request, length, tp).build()
        endpoint = Endpoints(self.k, 0, self.config, self.request)
        return endpoint.embedding(*dims) if kind == "embedding" else endpoint.head(*dims)

    def bind(self, shape, breakdown, chip, location):
        base = self.shapes(*shape)
        remaining = breakdown.copy()
        if shape[0] == "block":
            for group, role in (("layer_weights", "weight"), ("layer_kv_cache", "kv_cache"),
                                ("shared_rope_tables", "rope_table")):
                remaining[group] = [a-b for a, b in zip(breakdown[group], resident(base, role), strict=True)]
        else:
            del remaining["embedding_weight" if shape[0] == "embedding" else "final_norm_and_lm_head"]
        return self.bind_reserved(shape, remaining, chip, location)

    def bind_reserved(self, shape, remaining, chip, location):
        other = tuple(map(sum, zip(*remaining.values())))
        key = (shape, other)
        if key not in self.by_key:
            graph = copy(self.shapes(*shape))
            graph.nodes = deepcopy(graph.nodes)
            graph.rank, graph.reserved = chip, other
            name = f"p{len(self.graphs)}"
            self.by_key[key] = name
            self.graphs[name] = (graph, location, remaining)
            self.checked(graph.capacity, location, remaining)
        return self.by_key[key]

    @staticmethod
    def checked(run, location, remaining):
        try:
            return run()
        except BlockCapacityError as exc:
            exc.node = f"{location}/{exc.node}"
            if "other_resident_tensors" in exc.tensors:
                del exc.tensors["other_resident_tensors"]
                for group, sizes in remaining.items():
                    value = sum(sizes) if exc.unit is None else sizes[exc.unit]
                    if value:
                        exc.tensors["other."+group] = value
            raise

    def execute(self, kernels, *, search_tiles=False):
        profiles = {}
        for name, (graph, location, remaining) in self.graphs.items():
            graph.k = kernels
            profile = self.checked(lambda: graph.execute(search_tiles=search_tiles), location, remaining)
            profiles[name] = dict(profile, other_resident_bytes_per_unit=graph.reserved)
        return profiles

    def evaluate(self, mode, auto_tile, schedule, events, chips, stages):
        if not auto_tile:
            profiles = self.execute(self.k)
            if mode == "timing":
                profiles = self.execute(self.k.for_timing())
            return profiles, {}
        if mode != "timing":
            raise ValueError("--auto-tile requires timing mode to compare candidates")
        timing = self.k.for_timing()
        def metrics(profiles):
            return tile_metrics(schedule, events, chips, stages, profiles, timing.clock)

        try:
            baseline = self.execute(timing)
        except BlockCapacityError as exc:
            baseline = None
            original = {"capacity_error": exc.to_dict()}
        else:
            original = metrics(baseline)
        tuned = self.execute(timing, search_tiles=True)
        proposed = metrics(tuned)
        trials = {p: {n["id"]: n.pop("tile_search") for n in row["nodes"] if "tile_search" in n}
                  for p, row in tuned.items()}
        use_tuned = baseline is None or proposed["critical_path_cycles"] < original["critical_path_cycles"]
        selected = proposed if use_tuned else original
        saved = None if baseline is None else original["critical_path_cycles"] - selected["critical_path_cycles"]
        comparison = dict(scope="single-request compute critical path; communication and token selection cost zero",
                          baseline=original, per_node_candidate=proposed, selected=selected,
                          selected_policy="per_node" if use_tuned else "default", saved_cycles=saved,
                          latency_reduction_fraction=saved / original["critical_path_cycles"] if baseline else None,
                          profile_trials=trials)
        return tuned if use_tuned else baseline, {"tile_search": comparison}


def place_chips(config, lengths, stages, profiles):
    chips = []
    final = lengths[-1]
    for stage_id, stage in enumerate(stages):
        start, end = stage["layers"]
        tp = len(stage["chips"])
        layouts = tp_layout(config, stage["chips"])
        hidden_slices = list(_column_ranges(config["hidden_size"], tp))
        vocab_slices = list(_column_ranges(config["vocab_size"], tp))
        for rank, layout in enumerate(layouts):
            def span(name):
                lo, hi = layout[name]
                return hi-lo

            block = ("block", tp, span("query_heads"), span("kv_heads"), span("ffn_columns"))
            last_block = profiles.shapes(*block, final)
            breakdown = {
                "layer_weights": [v*(end-start) for v in resident(last_block, "weight")],
                "layer_kv_cache": [v*(end-start) for v in resident(last_block, "kv_cache")],
                "shared_rope_tables": resident(last_block, "rope_table"),
            }
            chip = {k: v for k, v in layout.items() if k != "profile"}
            chip.update(stage=stage_id, rank=rank, layers=[start,end], steps=[])
            if stage_id == 0:
                lo, hi = hidden_slices[rank]
                chip["embedding_hidden_columns"] = [lo, hi]
                embed = ("embedding", hi-lo)
                breakdown["embedding_weight"] = resident(profiles.shapes(*embed), "weight")
            if stage_id == len(stages)-1:
                lo, hi = vocab_slices[rank]
                chip["logit_vocab_range"] = [lo, hi]
                head = ("head", hi-lo, rank == 0)
                breakdown["final_norm_and_lm_head"] = resident(profiles.shapes(*head, False), "weight")
            total = [sum(lane) for lane in zip(*breakdown.values())]
            chip["resident_bytes_per_unit"] = breakdown
            chip["resident_bytes"] = sum(total)
            for length in lengths:
                def bind(shape, label):
                    return profiles.bind(shape, breakdown, chip["chip_id"], f"stage.{stage_id}/{label}/length.{length}")

                step = {}
                if stage_id == 0:
                    step["embedding"] = bind(embed, "embedding")
                step["block"] = bind((*block, length), f"layer.{start}")
                if stage_id == len(stages)-1:
                    step["head"] = bind((*head, length != final and rank == 0), "head")
                chip["steps"].append(step)
            chips.append(chip)
    return chips


def _run_model(config, request, arch, spec, stages, mode, auto_tile=False, *, config_dir=None):
    lengths = request.effective_lengths(config)
    validate_plan(stages, config)
    if mode not in ("timing", "count") or config.get("tie_word_embeddings", False):
        raise ValueError("Expected count or timing mode and untied LLaMA embedding weights")
    planner = ModelProfiles(DecoderKernels(arch, spec, "count", config_dir=config_dir), config, request)
    chips = place_chips(config, lengths, stages, planner)
    structure = {name: {"nodes": graph.nodes, "tensors": graph.tensors}
                 for name, (graph, _, _) in planner.graphs.items()}
    schedule, events = model_schedule(config, request, stages, chips, structure)
    profiles, comparison = planner.evaluate(mode, auto_tile, schedule, events, chips, stages)
    for chip in chips:
        peak = None
        for length, step in zip(lengths, chip["steps"], strict=True):
            cycles, macs = 0, 0
            for kind, name in step.items():
                profile = profiles[name]
                repeats = chip["layers"][1] - chip["layers"][0] if kind == "block" else 1
                cycles += repeats * (profile["local_compute_cycles"] or 0)
                macs += repeats * profile["useful_macs"]
                current = profile["capacity"]["peak"]
                if peak is None or current["required_bytes"] > peak["required_bytes"]:
                    peak = dict(required_bytes=current["required_bytes"], node=current["node"], profile=name,
                                kind=kind, effective_length=length)
            step.update(local_compute_cycles=cycles if mode == "timing" else None, useful_macs=macs,
                        local_compute_seconds=cycles/planner.k.clock if mode == "timing" and planner.k.clock else None)
        chip["peak"] = peak
    kv_heads = Counter()
    for chip in chips:
        lo, hi = chip["kv_heads"]
        kv_heads[chip["stage"]] += hi-lo
    totals = []
    for index, length in enumerate(lengths):
        head_macs = request.batch * config["hidden_size"] * config["vocab_size"]
        work = Counter(logical_useful_macs=head_macs, executed_useful_macs=head_macs)
        for i, stage in enumerate(stages):
            count = stage["layers"][1] - stage["layers"][0]
            per_layer = layer_work(config, request.batch, length, kv_heads[i])
            work.update({key: value*count for key, value in per_layer.items()})
        if work["executed_useful_macs"] != sum(chip["steps"][index]["useful_macs"] for chip in chips):
            raise ValueError("Placed chip MACs disagree with decoder layers, KV replicas and the output projection")
        totals.append(dict(effective_length=length, **work))
    return dict(schema_version=1, kind="llama_decode_model", mode=mode, architecture=arch, request=asdict(request),
                resources=planner.k.resources, stages=stages, chips=chips, profiles=profiles, schedule=schedule,
                effective_lengths=list(lengths), cache_capacity_tokens=lengths[-1], intra_chip_handoff_cycles=0,
                step_totals=totals, communication_events=events,
                latency_scope="per-chip local compute; network and token selection pending", **comparison)


def run_llama_model(request, architecture, spec, *, stages=None, pp=None, tp=None, mode="timing",
                    auto_tile=False, config_dir=None):
    config = load_model("llama3_70b")
    if stages is None and pp is None and tp is None:
        stages = recommended_plan(config["num_hidden_layers"], architecture)
    elif stages is None:
        default_pp, default_tp = (40,16) if architecture in ("tsp", "tsp_cim") else (2,2)
        pp = default_pp if pp is None else pp
        tp = default_tp if tp is None else tp
        if type(tp) is not int or not 1 <= tp <= config["num_attention_heads"]:
            raise ValueError("TP must fit the model's query head count")
        stages = uniform_plan(config["num_hidden_layers"], pp, tp)
    elif pp is not None or tp is not None:
        raise ValueError("Use an explicit stage plan or the uniform PP and TP options")
    result = _run_model(config, request, architecture, spec, stages, mode, auto_tile, config_dir=config_dir)
    return dict(result, model="llama3_70b", source=dict(SOURCES["llama3_70b"]))
