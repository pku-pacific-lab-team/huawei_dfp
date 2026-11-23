"""Fixed-routing MoE experts; routing arithmetic and weighted combine are separate."""
from dataclasses import dataclass

from .blocks import checked_inputs, SwiGLUFFN
from .types import OperatorSpec, TensorSpec, output_name, require_text


@dataclass(frozen=True)
class MoERoutingPlan:
    expert_count: int
    assignments: tuple
    name: str = "route_indices"

    def __post_init__(self):
        if type(self.expert_count) is not int or self.expert_count <= 0:
            raise ValueError("MoE expert_count must be a positive integer")
        require_text(self.name, "routing name")
        rows = self.assignments
        if not isinstance(rows, (tuple, list)) or not rows:
            raise ValueError("MoE assignments require a nonempty batch")
        if any(not isinstance(row, (tuple, list)) or not row for row in rows):
            raise ValueError("Each token requires a nonempty expert selection")
        k = len(rows[0])
        if k > self.expert_count or any(len(row) != k for row in rows):
            raise ValueError("MoE selections must have uniform top_k <= expert_count")
        if any(type(e) is not int or not 0 <= e < self.expert_count for row in rows for e in row):
            raise ValueError("MoE expert indices must be integers in range")
        if any(len(set(row)) != k for row in rows):
            raise ValueError("A token cannot select the same expert twice")
        object.__setattr__(self, "assignments", tuple(tuple(row) for row in rows))

    @property
    def batch(self):
        return len(self.assignments)

    @property
    def top_k(self):
        return len(self.assignments[0])

    @property
    def indices(self):
        return TensorSpec(self.name, (self.batch, self.top_k), "INT32")

    @property
    def slots_by_expert(self):
        slots = [[] for _ in range(self.expert_count)]
        for token, row in enumerate(self.assignments):
            for slot, expert in enumerate(row):
                slots[expert].append((token, slot))
        return tuple(tuple(row) for row in slots)

    @property
    def counts(self):
        return tuple(len(row) for row in self.slots_by_expert)

    def to_dict(self):
        return {"expert_count": self.expert_count, "assignments": [list(row) for row in self.assignments],
                "counts": list(self.counts), "total_assignments": self.batch * self.top_k,
                "slots_by_expert": [[list(slot) for slot in row] for row in self.slots_by_expert],
                "indices": self.indices.to_dict(), "source": "caller_supplied_fixed_routing"}


@dataclass(frozen=True)
class MoEExperts:
    x: TensorSpec
    gate_weight: TensorSpec
    up_weight: TensorSpec
    down_weight: TensorSpec
    routing: MoERoutingPlan
    shared_experts: int = 0
    shared_gate_weight: TensorSpec | None = None
    shared_up_weight: TensorSpec | None = None
    shared_down_weight: TensorSpec | None = None
    output_names: tuple | None = None
    operation = "moe_experts"

    def __post_init__(self):
        if not isinstance(self.routing, MoERoutingPlan):
            raise ValueError("Expected MoERoutingPlan")
        if type(self.shared_experts) is not int or self.shared_experts < 0:
            raise ValueError("shared_experts must be a nonnegative integer")
        shared = (self.shared_gate_weight, self.shared_up_weight, self.shared_down_weight)
        if (self.shared_experts and any(t is None for t in shared)) or (not self.shared_experts and any(t is not None for t in shared)):
            raise ValueError("Shared expert count and all three shared weights must be supplied together")
        checked_inputs(self.inputs)
        if self.x.dtype != "INT8" or len(self.x.shape) != 2 or self.x.shape[0] != self.routing.batch:
            raise ValueError("MoE x must be INT8 [routing batch, hidden]")
        e, h = self.routing.expert_count, self.x.shape[1]
        if len(self.gate_weight.shape) != 3:
            raise ValueError("MoE routed weights must have an expert axis")
        i = self.gate_weight.shape[-1]
        for t, shape in zip((self.gate_weight, self.up_weight, self.down_weight), ((e, h, i), (e, h, i), (e, i, h))):
            if t.dtype != "INT8" or t.shape != shape:
                raise ValueError("MoE routed weights must be INT8 [E,H,I], [E,H,I], [E,I,H]")
        if self.shared_experts:
            width = self.shared_experts * i
            for t, shape in zip(shared, ((h, width), (h, width), (width, h))):
                if t.dtype != "INT8" or t.shape != shape:
                    raise ValueError("Shared FFN width must equal shared_experts * routed intermediate size")
        count = 2 if self.shared_experts else 1
        names = self.output_names
        if names is not None and (not isinstance(names, (tuple, list)) or len(names) != count):
            raise ValueError("MoE requires a routed output name and, if enabled, a shared output name")
        if names is not None:
            for name in names:
                require_text(name, "MoE output name")
        outputs = []
        for n, label in enumerate(("expert_outputs", "shared_output")[:count]):
            selected = output_name(None if names is None else names[n], self.inputs + tuple(outputs), label)
            outputs.append(TensorSpec(selected, (1,), "FP32"))
        object.__setattr__(self, "output_names", tuple(t.name for t in outputs))

    @property
    def inputs(self):
        shared = (self.shared_gate_weight, self.shared_up_weight, self.shared_down_weight) if self.shared_experts else ()
        return (self.x, self.gate_weight, self.up_weight, self.down_weight, self.routing.indices) + shared

    @property
    def hidden(self):
        return self.x.shape[1]

    @property
    def intermediate(self):
        return self.gate_weight.shape[-1]

    @property
    def spec(self):
        outputs = (TensorSpec(self.output_names[0], (self.routing.batch, self.routing.top_k, self.hidden), "FP32"),)
        if self.shared_experts:
            outputs += (TensorSpec(self.output_names[1], self.x.shape, "FP32"),)
        return OperatorSpec(self.operation, self.inputs, outputs, "INT32",
            {"expert_count": self.routing.expert_count, "active_experts": sum(n > 0 for n in self.routing.counts),
             "top_k": self.routing.top_k, "intermediate_size": self.intermediate, "shared_experts": self.shared_experts,
             "routing_source": "caller_supplied_fixed_routing", "routing_compute_included": False,
             "weighted_combine_included": False, "output_order": "batch, selection_slot, hidden",
             "weight_storage": "all_declared_experts_resident", "expert_output_dtype": "FP32"})

    def expert(self, index):
        """Shape-only native FFN; empty experts deliberately have no kernel."""
        if type(index) is not int or not 0 <= index < self.routing.expert_count:
            raise ValueError("Expert index is outside this routing plan")
        n = self.routing.counts[index]
        if n == 0:
            raise ValueError("Empty expert has no FFN invocation")
        prefix, h, i = f"expert_{index}.", self.hidden, self.intermediate
        tensor = lambda label, shape: TensorSpec(prefix + label, shape, "INT8")
        return SwiGLUFFN(tensor("tokens", (n, h)), tensor("gate", (h, i)), tensor("up", (h, i)),
                         tensor("down", (i, h)), prefix + "output", "FP32")

    def shared(self):
        if not self.shared_experts:
            return None
        return SwiGLUFFN(self.x, self.shared_gate_weight, self.shared_up_weight, self.shared_down_weight,
                         self.output_names[1], "FP32")


@dataclass(frozen=True)
class _MoEPayload:
    source: TensorSpec
    target: TensorSpec | None
    row_bytes: int
    rows: int
    operation: str

    @property
    def spec(self):
        return OperatorSpec(self.operation, (self.source,), (self.target or self.source,), None,
            {"row_bytes": self.row_bytes, "rows": self.rows, "copy": self.target is not None,
             "read_only_alias": self.target is None})


def simulate_moe_experts(operator, architecture, spec, **options):
    from ..runtime.moe import simulate_moe_experts as run
    return run(operator, architecture, spec, **options)
