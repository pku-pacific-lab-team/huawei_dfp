"""Shape-only DeepSeek sigmoid router, weighted combine and complete MoE."""
from dataclasses import dataclass
from math import isfinite

from .blocks import checked_inputs
from .moe import MoEExperts
from .types import OperatorSpec, TensorSpec, output_name, require_text


@dataclass(frozen=True)
class MoECombine:
    experts: TensorSpec
    weights: TensorSpec
    shared: TensorSpec | None = None
    output_dtype: str = "FP32"
    output_name: str | None = None
    operation = "moe_combine"

    def __post_init__(self):
        checked_inputs(self.inputs)
        if self.experts.dtype != "FP32" or len(self.experts.shape) != 3:
            raise ValueError("Combine experts must be FP32 [B,k,H]")
        b, k, h = self.experts.shape
        if self.weights.dtype != "FP32" or self.weights.shape != (b, k):
            raise ValueError("Combine weights must be FP32 [B,k]")
        if self.shared is not None and (self.shared.dtype != "FP32" or self.shared.shape != (b, h)):
            raise ValueError("Combine shared result must be FP32 [B,H]")
        if self.output_dtype not in ("FP32", "INT8"):
            raise ValueError("Combine output must be FP32 or INT8")
        object.__setattr__(self, "output_name", output_name(self.output_name, self.inputs))

    @property
    def inputs(self):
        return (self.experts, self.weights) + (() if self.shared is None else (self.shared,))

    @property
    def spec(self):
        b, _, h = self.experts.shape
        return OperatorSpec(self.operation, self.inputs, (TensorSpec(self.output_name, (b, h), self.output_dtype),), "FP32",
            {"formula": "sum_k(experts * weights[...,None]) + optional_shared", "broadcast_materialized": False})


@dataclass(frozen=True)
class MoERouter:
    x: TensorSpec
    weight: TensorSpec
    top_k: int
    groups: int = 1
    selected_groups: int = 1
    bias: TensorSpec | None = None
    route_scale: float = 1.0
    output_names: tuple = ("route_indices", "route_weights")
    operation = "moe_router"

    def __post_init__(self):
        checked_inputs(self.inputs)
        if self.x.dtype != "INT8" or len(self.x.shape) != 2 or self.weight.dtype != "INT8" or len(self.weight.shape) != 2 or self.weight.shape[0] != self.x.shape[1]:
            raise ValueError("Router requires INT8 x[B,H] and gate weight[H,E]")
        e = self.weight.shape[1]
        if any(type(n) is not int or n < 1 for n in (self.top_k, self.groups, self.selected_groups)):
            raise ValueError("Router k and group counts must be positive integers")
        if e % self.groups or self.selected_groups > self.groups or self.top_k > self.selected_groups * (e // self.groups):
            raise ValueError("Router requires equal groups and enough selected candidates for top_k")
        if self.bias is not None and (self.bias.dtype != "FP32" or self.bias.shape != (e,)):
            raise ValueError("Router selection bias must be FP32 [E]")
        if self.groups > 1 and self.bias is not None and e // self.groups < 2:
            raise ValueError("Biased group scoring requires at least two experts per group")
        if type(self.route_scale) not in (int, float) or not isfinite(self.route_scale) or self.route_scale <= 0:
            raise ValueError("route_scale must be finite and positive")
        if not isinstance(self.output_names, (tuple, list)) or len(self.output_names) != 2:
            raise ValueError("Router requires indices and weights output names")
        taken = self.inputs
        for name in self.output_names:
            require_text(name, "Router output name")
            output_name(name, taken)
            taken += (TensorSpec(name, (1,), "FP32"),)
        object.__setattr__(self, "output_names", tuple(self.output_names))

    @property
    def inputs(self):
        return (self.x, self.weight) + (() if self.bias is None else (self.bias,))

    @property
    def spec(self):
        shape = (self.x.shape[0], self.top_k)
        return OperatorSpec(self.operation, self.inputs, tuple(TensorSpec(n, shape, d) for n, d in zip(self.output_names, ("INT32", "FP32"))), "INT32",
            {"score_function": "sigmoid", "top_k": self.top_k, "groups": self.groups,
             "selected_groups": self.selected_groups, "route_scale": self.route_scale,
             "bias_affects": "selection_only", "weight_source": "original_sigmoid_scores",
             "numeric_indices_generated": False})


@dataclass(frozen=True)
class MoE:
    router: MoERouter
    experts: MoEExperts
    output_dtype: str = "INT8"
    output_name: str | None = None
    operation = "moe"

    def __post_init__(self):
        if not isinstance(self.router, MoERouter) or not isinstance(self.experts, MoEExperts):
            raise ValueError("MoE requires a router and expert block")
        r, e = self.router, self.experts
        if r.x != e.x or r.weight.shape[1] != e.routing.expert_count or r.top_k != e.routing.top_k or r.spec.outputs[0] != e.routing.indices:
            raise ValueError("MoE router and expert input/routing contracts differ")
        width = e.routing.expert_count // r.groups
        if any(len({i // width for i in row}) > r.selected_groups for row in e.routing.assignments):
            raise ValueError("Scenario assignments exceed selected router groups")
        checked_inputs(self.inputs)
        names = [t.name for t in (*self.inputs, *r.spec.outputs, *e.spec.outputs)]
        if len(names) != len(set(names)):
            raise ValueError("MoE intermediate and external tensors cannot alias")
        if self.output_dtype not in ("INT8", "FP32"):
            raise ValueError("MoE output must be INT8 or FP32")
        object.__setattr__(self, "output_name", output_name(self.output_name, (*self.inputs, *r.spec.outputs, *e.spec.outputs)))

    @property
    def inputs(self):
        return tuple(t for t in self.experts.inputs if t != self.experts.routing.indices) + self.router.inputs[1:]

    def combine(self):
        out = self.experts.spec.outputs
        return MoECombine(out[0], self.router.spec.outputs[1], out[1] if len(out) > 1 else None, self.output_dtype, self.output_name)

    @property
    def spec(self):
        return OperatorSpec(self.operation, self.inputs, self.combine().spec.outputs, "INT32",
            {"routing_compute_included": True, "weighted_combine_included": True,
             "routing_distribution": "caller_supplied_scenario", "numeric_values_simulated": False})


def simulate_moe(operator, architecture, spec, **options):
    from ..runtime.moe_block import simulate_moe as run
    return run(operator, architecture, spec, **options)


def simulate_moe_router(operator, architecture, spec, **options):
    from ..runtime.moe_block import simulate_moe_router as run
    return run(operator, architecture, spec, **options)


def simulate_moe_combine(operator, architecture, spec, **options):
    from ..runtime.moe_block import simulate_moe_combine as run
    return run(operator, architecture, spec, **options)


@dataclass(frozen=True)
class _RouterSelection:
    router: MoERouter
    logits: TensorSpec
    operation = "moe_route_selection"

    def __getattr__(self, name):
        return getattr(self.router, name)

    @property
    def inputs(self):
        return (self.logits,) + (() if self.router.bias is None else (self.router.bias,))

    @property
    def spec(self):
        return OperatorSpec(self.operation, self.inputs, self.router.spec.outputs, "FP32", dict(self.router.spec.attributes))
