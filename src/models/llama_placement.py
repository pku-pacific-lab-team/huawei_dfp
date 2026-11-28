"""Contiguous PP stages, rank layouts, and native decoder endpoint graphs."""
from .block_graph import widths
from .llama_block import RankOperators
from .projection import _column_ranges


def uniform_plan(layers, pp, tp):
    if type(tp) is not int or tp < 1:
        raise ValueError("TP must be a positive integer")
    return [{"layers": [start, end], "chips": list(range(stage*tp, (stage+1)*tp))}
            for stage, (start, end) in enumerate(_column_ranges(layers, pp))]


def recommended_plan(layers, architecture):
    if architecture not in ("tsp", "tsp_cim"):
        return uniform_plan(layers, 3, 1)
    stages = []
    start = offset = 0
    while start < layers:
        # The two wider endpoint groups each hold two layers plus embedding or lm_head.
        endpoint = start == 0 or start >= layers - 2
        tp = 16 if endpoint else 4
        stop = min(start + 2, layers) if endpoint else start + 1
        stages.append({"layers": [start, stop], "chips": list(range(offset, offset + tp))})
        offset += tp
        start = stop
    return stages


def validate_plan(stages, config):
    if not isinstance(stages, list) or not stages:
        raise ValueError("Expected a nonempty list of PP stages")
    next_layer, used = 0, set()
    for stage in stages:
        if not isinstance(stage, dict) or set(stage) != {"layers", "chips"}:
            raise ValueError("Each stage requires layers and chips")
        interval, chips = stage["layers"], stage["chips"]
        if (not isinstance(interval, list) or len(interval) != 2 or any(type(i) is not int for i in interval)
                or interval[0] != next_layer or not next_layer < interval[1] <= config["num_hidden_layers"]):
            raise ValueError("PP layers must cover ordered, contiguous, nonempty half-open intervals")
        if (not isinstance(chips, list) or not 1 <= len(chips) <= min(config["num_attention_heads"],
                config["intermediate_size"], config["vocab_size"]) or any(type(i) is not int or i < 0 for i in chips)
                or len(set(chips)) != len(chips) or used.intersection(chips)):
            raise ValueError("Invalid or reused chip in TP group")
        next_layer = interval[1]
        used.update(chips)
    if next_layer != config["num_hidden_layers"]:
        raise ValueError("PP stages must cover every decoder layer")


class Endpoints(RankOperators):
    def __init__(self, kernels, chip_id, config, request):
        super().__init__(kernels, chip_id)
        self.u, self.b = kernels.units, request.batch
        self.h, self.v = config["hidden_size"], config["vocab_size"]
        self.epsilon = config["rms_norm_eps"]

    def embedding(self, width):
        # Hidden-column sharding works for every symbolic token ID. Each active
        # unit owns whole vocabulary rows for its own columns.
        ns = widths(width, self.u)
        ids = self.tensor("token_ids", (self.b,), "INT32", [4*self.b if n else 0 for n in ns], "input")
        weight = self.tensor("embedding.weight", (self.v,width), "INT8", [self.v*n for n in ns], "weight")
        shard = self.tensor("hidden.shard", (self.b,width), "INT8", [self.b*n for n in ns])
        runs = [self.k.kernel("embedding", self.b, self.v, n) if n else None for n in ns]
        self.emit("lookup", [ids, weight], [shard], [("row_lookup", runs, 1)], unit_hidden_columns=ns)
        if width != self.h:
            full = self.tensor("hidden", (self.b,self.h), "INT8", widths(self.b*self.h, self.u))
            self.emit("hidden.all_gather", [shard], [full])
        return self

    def head(self, vocab_width, leader, select_token):
        x = self.tensor("input", (self.b,self.h), "INT8", widths(self.b*self.h, self.u), "input")
        logits = self.cast("logits.fp32", self.matrix("lm_head", self.rms("final_norm", x), vocab_width), "FP32")
        if vocab_width != self.v:
            outputs = []
            if leader:
                full = self.tensor("logits.full", (self.b,self.v), "FP32", [4*n for n in widths(self.b*self.v, self.u)])
                outputs.append(full)
            self.emit("logits.gather", [logits], outputs)
            if leader:
                logits = full
        if leader and select_token:
            ids = self.tensor("next_ids", (self.b,), "INT32", [4*self.b] + [0]*(self.u-1))
            self.emit("select_token", [logits], [ids], service="external_token_selection", service_cycles=None)
        return self
