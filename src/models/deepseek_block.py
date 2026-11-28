"""Absorbed MLA with head-parallel projections and replicated compressed caches."""

from math import log, sqrt

from .block_graph import BlockGraph, widths
from .llama_block import RankBlock


class MLARank(RankBlock):
    def __init__(self, kernels, config, request, length, tp, heads, dense_width):
        BlockGraph.__init__(self, kernels, 0)
        self.u, self.b, self.h = kernels.units, request.batch, config["hidden_size"]
        self.q, self.i, self.tp = heads, dense_width, tp
        self.d = config["qk_rope_head_dim"]
        self.nope, self.value = config["qk_nope_head_dim"], config["v_head_dim"]
        self.qrank, self.latent = config["q_lora_rank"], config["kv_lora_rank"]
        self.length, self.capacity_length = length, request.sequence_length + request.decode_steps
        self.epsilon = config["rms_norm_eps"]
        yarn = config["rope_scaling"]
        multiplier = 1 + 0.1 * yarn["mscale_all_dim"] * log(max(1, yarn["factor"]))
        self.scale = multiplier**2 / sqrt(self.nope + self.d)

    def head_matrix(self, name, x, k, n, cache=None, *, flatten=False):
        # Per-head matrices have different weights. Cache matrices, in contrast,
        # share one backing across the heads of each batch item.
        rows = self.q * self.b
        ns = widths(n, self.u)
        storage = [self.k.gemm_bytes(1, k, w) if w else (0, 0, 0) for w in ns]
        a, weights, out = zip(*storage)
        self.reserve(x, [rows * v for v in a])
        if cache is None:
            rhs = self.tensor(name + ".weight", (self.q, k, n), "INT8", [self.q * v for v in weights], "weight")
        else:
            rhs = cache
            self.reserve(rhs, [self.b * v for v in weights])
        shape = (self.b, self.q * n) if flatten else (self.b, self.q, n)
        result = self.tensor(name, shape, "INT32", [rows * v for v in out])
        runs = [self.k.kernel("gemm", 1, k, w) if w else None for w in ns]
        self.emit(name, [x, rhs], [result], [("head_gemm", runs, rows)], unit_output_columns=ns)
        return result

    def cache(self, latent, pos):
        cap, length = self.capacity_length, self.length
        phases, caches = [], []
        for name, width, src in (("latent", self.latent, latent), ("position", self.d, pos)):
            channels, positions = widths(width, self.u), widths(cap, self.u)
            sizes = []
            for w, n in zip(channels, positions):
                key = self.k.gemm_bytes(1, width, n)[1] if n else 0
                value = self.k.gemm_bytes(1, cap, w)[1] if w and name == "latent" else 0
                sizes.append(self.b * max(key, value, cap * w))
            caches.append(self.tensor(name + ".cache", (self.b, 1, cap, width), "INT8", sizes, "kv_cache"))
            self.reserve(src, [self.b * n for n in channels])
            runs = [self.k.kernel("cache_component", self.b, n, length, cap) if n else None for n in channels]
            phases.append((name + "_append", runs, 1))
        self.emit(
            "kv.append",
            [latent, pos, *caches],
            [],
            phases,
            updates=caches,
            appended_bytes=self.b * (self.latent + self.d),
            token_interval=[length - 1, length],
        )
        return caches

    def attention(self, x):
        q = self.rms("q_norm", self.matrix("q.down", x, self.qrank))
        qc = self.cast("q.content", self.matrix("q.content.up", q, self.q * self.nope), "INT8")
        qp = self.cast("q.pos", self.matrix("q.pos.up", q, self.q * self.d), "INT8")
        latent = self.rms("kv_norm", self.matrix("kv.latent.down", x, self.latent))
        pos = self.cast("k.pos", self.matrix("kv.pos.down", x, self.d), "INT8")
        for name in ("cos", "sin"):
            sizes = [4 * self.capacity_length * n for n in widths(self.d // 2, self.u)]
            self.tensor(name, (self.capacity_length, self.d // 2), "FP32", sizes, "rope_table")
        qp = self.rotary("query.rope", qp, self.q)
        lc, pc = self.cache(latent, self.rotary("key.rope", pos, 1))
        ql = self.cast("q.latent", self.head_matrix("q.absorb", qc, self.nope, self.latent), "INT8")
        content = self.cast(
            "content.fp32", self.head_matrix("content.score", ql, self.latent, self.length, lc), "FP32"
        )
        position = self.cast(
            "position.fp32", self.head_matrix("position.score", qp, self.d, self.length, pc), "FP32"
        )
        score = self.element("scores", [content, position], "FP32", "fp32_add")
        scaled = self.element("scale", [score], "FP32", "scale_fp32", self.scale)
        self.nodes[-1]["scale"] = self.scale
        prob = self.cast("prob", self.normalize("softmax", scaled, "softmax"), "INT8")
        weighted = self.cast("weighted", self.head_matrix("pv", prob, self.length, self.latent, lc), "INT8")
        return self.cast(
            "context", self.head_matrix("value.up", weighted, self.latent, self.value, flatten=True), "INT8"
        )

    def build(self):
        if self.i:
            return super().build()
        x = self.tensor("input", (self.b, self.h), "INT8", widths(self.b * self.h, self.u), "input")
        ctx = self.attention(self.rms("attention_norm", x))
        projected = self.collective("attention.reduce", self.matrix("attention.output.partial", ctx, self.h))
        skip = self.element("attention.residual", [x, projected], "INT8", "residual")
        normalized = self.rms("ffn_norm", skip)
        self.emit("moe.inputs", [skip, normalized], [])
        return self
