"""One GQA/SwiGLU LLaMA decoder layer on an explicit tensor-parallel group."""
from dataclasses import asdict, replace
from math import prod, sqrt

from .block_graph import BlockGraph, widths
from .config import SOURCES, load_model
from .decoder_kernels import DecoderKernels, KernelCall
from .projection import _column_ranges


class RankOperators(BlockGraph):
    def matrix(self, name, x, out_width):
        """The rank already owns K or N according to TP; within the chip split N."""
        b, k = self.tensors[x]["shape"]
        if self.tensors[x]["dtype"] != "INT8":
            raise ValueError(f"{name} projection requires INT8 input")
        ns = widths(out_width, self.u)
        storage = [self.k.gemm_bytes(b, k, n) if n else (0,0,0) for n in ns]
        a, w, c = zip(*storage)
        self.reserve(x, a)
        weight = self.tensor(name+".weight", (k, out_width), "INT8", w, "weight")
        out = self.tensor(name, (b, out_width), "INT32", c)
        runs = [self.k.kernel("gemm", b, k, n) if n else None for n in ns]
        self.emit(name, [x, weight], [out], [("local_gemm", runs, 1)], unit_output_columns=ns)
        return out

    def element(self, name, inputs, out_dtype, kind, *args):
        first = self.tensors[inputs[0]]
        if any(self.tensors[t]["shape"] != first["shape"] for t in inputs):
            raise ValueError(f"Elementwise shape mismatch in {name}")
        expected = {"residual": "INT8", "swiglu": "INT32", "scale": "INT32"}.get(kind)
        if expected and any(self.tensors[t]["dtype"] != expected for t in inputs):
            raise ValueError(f"{name} expects {expected} operands")
        sizes = widths(prod(first["shape"]), self.u)
        for t in inputs:
            item = self.tensors[t]
            self.reserve(t, [n * (item["logical_bytes"] // item["numel"]) for n in sizes])
        out = self.tensor(name, first["shape"], out_dtype, [n*(1 if out_dtype == "INT8" else 4) for n in sizes])
        runs = [self.k.kernel(kind, n, *args) if n else None for n in sizes]
        self.emit(name, inputs, [out], [(kind, runs, 1)])
        return out

    def cast(self, name, x, dtype):
        return self.element(name, [x], dtype, "cast", self.tensors[x]["dtype"], dtype)

    def normalize(self, name, x, kind):
        tensor = self.tensors[x]
        if tensor["dtype"] != "FP32":
            raise ValueError(f"{name} normalization expects FP32 input")
        width = tensor["shape"][-1]
        rows = tensor["numel"] // width
        splits = widths(width, self.u)
        sizes = [4*n*rows for n in splits]
        self.reserve(x, sizes)
        inputs = [x]
        if kind == "rmsnorm":
            inputs.append(self.tensor(name+".weight", (width,), "FP32", [4*n for n in splits], "weight"))
            order = ("sum_squares", "merge_fp32_add", "normalizer", "apply_scale_and_weight")
        else:
            order = ("row_max", "merge_fp32_max", "exponentials_and_sum", "merge_fp32_add", "reciprocal", "normalize")
        out = self.tensor(name, tensor["shape"], "FP32", sizes)
        participating = min(width, self.u)
        def local(i, phase):
            return KernelCall(kind, (splits[i], self.epsilon), phase)
        stages = []
        for phase in order:
            if phase.startswith("merge_"):
                active = list(range(participating))
                while len(active) > 1:
                    receivers = active[:-1:2]
                    runs = [local(i, phase) if i in receivers else None for i in range(self.u)]
                    stages.append((phase, runs, rows))
                    active = active[::2]
            else:
                active = 1 if phase in ("reciprocal", "normalizer") else participating
                stages.append((phase, [local(i, phase) if i < active else None for i in range(self.u)], rows))
        self.emit(name, inputs, [out], stages, row_width=width, row_count=rows, unit_widths=splits,
                  epsilon=self.epsilon if kind == "rmsnorm" else None,
                  normalizer_multiplier=1/width if kind == "rmsnorm" else None,
                  schedule="repeat_complete_phase_sequence_per_row")
        return out

    def rms(self, name, x):
        fp = self.cast(name+".to_fp32", x, "FP32")
        return self.cast(name+".to_int8", self.normalize(name, fp, "rmsnorm"), "INT8")


class RankBlock(RankOperators):
    def __init__(self, kernels, layout, config, request, length, tp):
        super().__init__(kernels, layout["chip_id"])
        self.b, self.h = request.batch, config["hidden_size"]
        self.d = self.h // config["num_attention_heads"]
        self.q = layout["query_heads"][1] - layout["query_heads"][0]
        self.kv = layout["kv_heads"][1] - layout["kv_heads"][0]
        self.i = layout["ffn_columns"][1] - layout["ffn_columns"][0]
        self.length, self.capacity_length = length, request.sequence_length + request.decode_steps
        self.epsilon, self.tp = config["rms_norm_eps"], tp
        self.u = kernels.units

    def collective(self, name, x):
        # All-reduce overwrites INT32 partials. Network arithmetic/time belongs
        # to the later system model; consumers explicitly wait for this event.
        tensor = self.tensors[x]
        self.emit(name, [x], [], updates=[x], collective="all_reduce_sum_int32" if self.tp > 1 else "identity",
                  participants=self.tp, payload_bytes_per_rank=tensor["logical_bytes"],
                  group_int32_adds=(self.tp-1)*tensor["numel"], communication_cycles=None)
        return self.cast(name+".to_int8", x, "INT8")

    def rotary(self, name, x, heads):
        if self.tensors[x]["shape"] != [self.b, heads*self.d] or self.tensors[x]["dtype"] != "INT8":
            raise ValueError(f"{name} has incompatible projection output")
        parts = [2*n for n in widths(self.d//2, self.u)]
        sizes = [self.b*heads*d for d in parts]
        self.reserve(x, sizes)
        out = self.tensor(name, (self.b, heads, 1, self.d), "INT8", sizes)
        runs = [self.k.kernel("rope", self.b, heads, d, self.length, self.capacity_length) if d else None for d in parts]
        self.emit(name, [x, "cos", "sin"], [out], [("rotate_pairs", runs, 1)], position=self.length-1)
        return out

    def attention(self, x):
        b, q, kv, d, length, cap = self.b, self.q, self.kv, self.d, self.length, self.capacity_length
        dims = widths(d, self.u)
        rows = b*q
        query, key, value = [self.cast(label, self.matrix(label+".projection", x, heads*d), "INT8")
                             for label, heads in (("query", q), ("key", kv), ("value", kv))]
        for table in ("cos", "sin"):
            self.tensor(table, (cap,d//2), "FP32", [4*cap*n for n in widths(d//2, self.u)], "rope_table")
        query = self.rotary("query.rope", query, q)
        key = self.rotary("key.rope", key, kv)
        for label in ("key", "value"):
            sizes = []
            for part in dims:
                shape = (1,part,cap) if label == "key" else (1,cap,part)
                size = self.k.gemm_bytes(*shape)[1] if part else 0
                sizes.append(b*kv*size)
            self.tensor(label+".cache", (b,kv,cap,d), "INT8", sizes, "kv_cache")
        self.reserve(value, [b*kv*n for n in dims])
        self.reserve(key, [b*kv*n for n in dims])
        runs = [self.k.kernel("append", b, kv, n, length, cap) if n else None for n in dims]
        caches = ["key.cache", "value.cache"]
        self.emit("kv.append", [key, value, *caches], [], [("append", runs, 1)], updates=caches,
                  token_interval=[length-1, length], appended_bytes=2*b*kv*d)

        # Split QK along channels. Each participating unit produces an INT32
        # score vector; binary merges finish before applying 1/sqrt(global D).
        local = [self.k.gemm_bytes(1,n,length) if n else (0,0,0) for n in dims]
        self.reserve(query, [rows*s[0] for s in local])
        partial = self.tensor("qk.partial", (b,q,length), "INT32", [rows*s[2] for s in local])
        runs = [self.k.kernel("gemm", 1,n,length) if n else None for n in dims]
        self.emit(partial, [query, caches[0]], [partial], [("head_gemm", runs, rows)], unit_k=dims,
                  valid_kv_length=length, cache_read_payload_bytes=rows*length*d)
        splits = widths(length, self.u)
        scores = self.tensor("qk.sum", (b,q,length), "INT32", [4*rows*n for n in splits])
        active = list(range(min(d,self.u)))
        stages = []
        while len(active) > 1:
            receivers = active[:-1:2]
            add = self.k.kernel("add_partial", length)
            # One peer input and a disjoint output reused for the next head.
            add = replace(add, extra_workspace=2*self.k.size(4*length))
            stages.append(("sum_partial_scores", [add if i in receivers else None for i in range(self.u)], rows))
            active = active[::2]
        self.emit(scores, [partial], [scores], stages)
        scaled = self.element("scores.scale", [scores], "FP32", "scale", 1/sqrt(d))
        prob = self.cast("probabilities", self.normalize("softmax", scaled, "softmax"), "INT8")
        local = [self.k.gemm_bytes(1,length,n) if n else (0,0,0) for n in dims]
        self.reserve(prob, [rows*s[0] for s in local])
        context = self.tensor("pv", (b,q*d), "INT32", [rows*s[2] for s in local])
        runs = [self.k.kernel("gemm", 1,length,n) if n else None for n in dims]
        self.emit(context, [prob, caches[1]], [context], [("head_gemm", runs, rows)],
                  valid_kv_length=length, cache_read_payload_bytes=rows*length*d, context_view=[b,q,1,d])
        return self.cast("context", context, "INT8")

    def build(self):
        hidden = self.tensor("input", (self.b,self.h), "INT8", widths(self.b*self.h,self.u), "input")
        context = self.attention(self.rms("attention_norm", hidden))
        proj = self.collective("attention.reduce", self.matrix("attention.output.partial", context, self.h))
        skip = self.element("attention.residual", [hidden,proj], "INT8", "residual")
        norm = self.rms("ffn_norm", skip)
        gate = self.matrix("ffn.gate", norm, self.i)
        up = self.matrix("ffn.up", norm, self.i)
        act = self.element("ffn.swiglu", [gate,up], "INT8", "swiglu")
        down = self.collective("ffn.reduce", self.matrix("ffn.down.partial", act, self.h))
        self.element("output", [skip,down], "INT8", "residual")
        return self

    def run(self):
        return dict(self.build().execute(), effective_length=self.length)


def tp_layout(config, chip_ids):
    tp = len(chip_ids)
    hidden = config["hidden_size"]
    query_heads = config["num_attention_heads"]
    kv_heads = config["num_key_value_heads"]
    intermediate = config["intermediate_size"]
    if min(hidden, query_heads, kv_heads) <= 0 or hidden % query_heads or query_heads % kv_heads:
        raise ValueError("GQA requires integral head dimensions and query/KV groups")
    head_dim = hidden // query_heads
    if head_dim % 2:
        raise ValueError("GQA requires integral query/KV groups and an even RoPE head dimension")
    group = query_heads // kv_heads
    chips = []
    partitions = zip(_column_ranges(query_heads, tp), _column_ranges(intermediate, tp), strict=True)
    for rank, (query_range, ffn_range) in enumerate(partitions):
        start, stop = query_range
        kv_first, kv_stop = start // group, (stop - 1) // group + 1
        key = f"q{stop-start}_kv{kv_stop-kv_first}_i{ffn_range[1]-ffn_range[0]}"
        chip = dict(chip_id=chip_ids[rank], query_heads=list(query_range), kv_heads=[kv_first, kv_stop],
                    ffn_columns=list(ffn_range), query_to_kv=[i // group for i in range(start, stop)], profile=key)
        chips.append(chip)
    return chips


def layer_work(config, batch, length, physical_kv_heads):
    hidden, kv_heads = config["hidden_size"], config["num_key_value_heads"]
    head_dim = hidden // config["num_attention_heads"]
    projection_weights = 2*hidden**2 + 2*hidden*kv_heads*head_dim + 3*hidden*config["intermediate_size"]
    logical = batch * (projection_weights + 2*hidden*length)
    replicated = batch * 2*hidden*head_dim*(physical_kv_heads - kv_heads)
    return dict(logical_useful_macs=logical, replica_projection_macs=replicated,
                executed_useful_macs=logical+replicated, logical_kv_append_bytes=2*batch*kv_heads*head_dim,
                physical_kv_append_bytes=2*batch*physical_kv_heads*head_dim)


def _run_block(config, request, arch, spec, tp, layer, mode):
    lengths = request.effective_lengths(config)
    if type(layer) is not int or not 0 <= layer < config["num_hidden_layers"] or mode not in ("count","timing"):
        raise ValueError("Invalid LLaMA layer or run mode")
    if type(tp) is not int or not 1 <= tp <= config["num_attention_heads"]:
        raise ValueError("Invalid TP group size")
    chips = tp_layout(config, range(tp))
    kernels = DecoderKernels(arch, spec, mode)
    profiles = {}
    for chip in chips:
        key = chip["profile"]
        if key not in profiles:
            profiles[key] = [RankBlock(kernels, chip, config, request, length, tp).run() for length in lengths]
    physical_kv_heads = sum(c["kv_heads"][1]-c["kv_heads"][0] for c in chips)
    totals = []
    for i, length in enumerate(lengths):
        executed = sum(profiles[c["profile"]][i]["useful_macs"] for c in chips)
        work = layer_work(config, request.batch, length, physical_kv_heads)
        if executed != work["executed_useful_macs"]:
            raise ValueError("Block MAC ledger disagrees with global weights, KV replicas and attention lengths")
        totals.append(dict(effective_length=length, **work))
    return dict(schema_version=1, kind="llama_decoder_block", mode=mode, architecture=arch, layer=layer,
                tp=tp, request=asdict(request), effective_lengths=list(lengths), resources=kernels.resources,
                chips=chips, profiles=profiles, step_totals=totals,
                logical_kv_heads=config["num_key_value_heads"], physical_kv_heads=physical_kv_heads,
                cache_capacity_tokens=lengths.stop-1, intra_chip_handoff_cycles=0,
                latency_scope="local compute per rank and step; both TP collective times pending")


def run_llama_block(request, architecture, spec, *, tp=4, layer=0, mode="count"):
    result = _run_block(load_model("llama3_70b"), request, architecture, spec, tp, layer, mode)
    return dict(result, model="llama3_70b", source=dict(SOURCES["llama3_70b"]))
