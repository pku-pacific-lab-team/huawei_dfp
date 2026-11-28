"""Rank-local routed/shared FFNs, sigmoid group-limited router and weighted combine."""

from ..core import block_moe
from ..operators.adapters import adapt_native_result
from ..operators.moe_block import MoECombine, MoERouter, _RouterSelection
from ..operators.types import TensorSpec
from .block_graph import widths
from .decoder_kernels import DecoderKernels
from .llama_block import RankOperators


def router_spec(config, batch):
    h, e = config["hidden_size"], config["n_routed_experts"]
    gate = MoERouter(
        TensorSpec("x", (batch, h), "INT8"),
        TensorSpec("gate", (h, e), "INT8"),
        config["num_experts_per_tok"],
        config["n_group"],
        config["topk_group"],
        TensorSpec("bias", (e,), "FP32"),
        config["routed_scaling_factor"],
    )
    return _RouterSelection(gate, TensorSpec("logits", (batch, e), "INT32"))


class R1Kernels(DecoderKernels):
    def __init__(self, arch, spec, mode, config, *, config_dir=None):
        super().__init__(arch, spec, mode, config_dir=config_dir)
        self.router = router_spec(config, 1)

    def _kernel(self, kind, *dims, tile=None):
        if kind == "route_phase":
            operation, reads, writes, n, reduction = dims
            op = self.router
            path = block_moe.router_path(
                op, self.arch, self.spec, self.execution, self.mode, 0, self.capacity, self.allocation, tile,
                fragment_bytes=sum(reads) + sum(writes),
            )
            schedule = block_moe.Schedule(op, path, self.execution, self.block, self.mode)
            schedule.step(
                "local_selection", reads, writes, operation, n, external=True, reduction_width=reduction
            )
            raw = schedule.result("Distributed fragment of the native router scan")
        elif kind == "combine":
            b, k, h = dims
            op = MoECombine(
                TensorSpec("routed", (b, k, h), "FP32"),
                TensorSpec("weights", (b, k), "FP32"),
                TensorSpec("shared", (b, h), "FP32"),
                "INT8",
            )
            path = block_moe.combine_path(
                op, self.arch, self.spec, self.execution, self.mode, 0, self.capacity, self.allocation, tile
            )
            raw = block_moe.combine(op, path, self.execution, self.block, self.mode)
        else:
            return super()._kernel(kind, *dims, tile=tile)
        return self._cost(
            adapt_native_result(op.spec, self.arch, raw), path.memory["resident_bytes"] if self.resident else 0
        )


class ParallelSelection:
    """Translate the existing serial row policy into balanced waves and binary scalar merges."""

    def __init__(self, kernels):
        self.k, self.phases = kernels, []

    def step(self, label, reads, writes, operation=None, count=0, *, external=False, reduction_width=0):
        def call(op, r, w, n, red=0):
            return self.k.kernel("route_phase", op, tuple(r), tuple(w), n, red)

        # Native group scoring sums the two winners in a single instruction.
        # With distributed fragments that scalar sum belongs to one merge.
        if label.endswith(".score") and operation == "fp32_add":
            reduction_width = count + 1
        if reduction_width > 1:
            parts = widths(reduction_width, self.k.units)
            runs = [call(operation, [4 * n], [4], n - 1, n) if n else None for n in parts]
            self.phases.append((label, runs, self.repeats))
            active = list(range(min(reduction_width, self.k.units)))
            while len(active) > 1:
                receivers = active[:-1:2]
                runs = [
                    call(operation, [4, 4], [4], 1, 2) if i in receivers else None for i in range(self.k.units)
                ]
                self.phases.append((label + ".merge", runs, self.repeats))
                active = active[::2]
        else:
            rows = max([*reads, *writes]) // 4
            parts = widths(rows, self.k.units)
            runs = [
                call(
                    operation,
                    [4 * n if x == 4 * rows else x for x in reads],
                    [4 * n if x == 4 * rows else x for x in writes],
                    n if count else 0,
                )
                if n
                else None
                for n in parts
            ]
            self.phases.append((label, runs, self.repeats))


class ExpertOperators(RankOperators):
    def __init__(self, kernels, config, batch):
        super().__init__(kernels, 0)
        self.b, self.h, self.u = batch, config["hidden_size"], kernels.units
        self.config = config

    def hidden(self, name):
        return self.tensor(name, (self.b, self.h), "INT8", widths(self.b * self.h, self.u), "input")

    def expert(self, width, etp, leader):
        x = self.hidden("tokens")
        gate, up = self.matrix("gate", x, width), self.matrix("up", x, width)
        act = self.element("swiglu", [gate, up], "INT8", "swiglu")
        down = self.matrix("down.partial", act, self.h)
        if etp > 1:
            self.emit(
                "expert.reduce",
                [down],
                [],
                updates=[down],
                collective="all_reduce_sum_int32",
                participants=etp,
                payload_bytes_per_rank=4 * self.b * self.h,
                communication_cycles=None,
            )
        if leader:
            self.cast("expert.fp32", down, "FP32")
        return self

    def router(self):
        op = router_spec(self.config, self.b)
        e, k = op.weight.shape[1], op.top_k
        logits = self.matrix("router.gate", self.hidden("input"), e)
        bias = self.tensor("bias", (e,), "FP32", [4 * n for n in widths(e, self.u)], "weight")
        outputs = [
            self.tensor(name, (self.b, k), dtype, [4 * n for n in widths(self.b * k, self.u)])
            for name, dtype in (("indices", "INT32"), ("weights", "FP32"))
        ]
        # Intermediate score vectors, winner IDs and partial scalar reductions
        # persist during the complete selection node; reserve a scalar per unit.
        buffers = block_moe.router_workspace(op, lambda x: x)
        sizes = [0] * self.u
        for size in buffers.values():
            for i, n in enumerate(widths(size // 4, self.u)):
                sizes[i] += self.k.size(4 * n)
        scratch = self.tensor(
            "selection.workspace",
            (sum(buffers.values()) // 4 + self.u,),
            "FP32",
            [size + self.k.size(4) for size in sizes],
        )
        waves = ParallelSelection(self.k)
        block_moe.router_steps(op, waves)
        self.emit(
            "router.select", [logits, bias], [*outputs, scratch], waves.phases, route_scale=op.route_scale
        )
        return self

    def combine(self):
        b, h, k = self.b, self.h, self.config["num_experts_per_tok"]
        ns = widths(h, self.u)
        routed = self.tensor("routed", (b, k, h), "FP32", [4 * b * k * n for n in ns], "input")
        weights = self.tensor("weights", (b, k), "FP32", [4 * b * k if n else 0 for n in ns], "input")
        shared = self.tensor("shared", (b, h), "FP32", [4 * b * n for n in ns], "input")
        skip = self.hidden("skip")
        result = self.tensor("combine", (b, h), "INT8", [b * n for n in ns])
        runs = [self.k.kernel("combine", b, k, n) if n else None for n in ns]
        self.emit(result, [routed, weights, shared], [result], [("weighted_sum_plus_shared", runs, 1)])
        self.element("output", [skip, result], "INT8", "residual")
        return self
