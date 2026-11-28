"""Local services used by the column- and row-parallel LLaMA block."""
from collections import Counter
from copy import copy
from dataclasses import dataclass
from functools import lru_cache

from ..core import block_elementwise, block_embedding, block_rope
from ..core.composite import RowSchedule, rmsnorm, softmax
from ..core.kv_cache import make_path as cache_path, simulate as cache_kernel, simulate_payloads
from ..core.vector_resources import resident_allocation
from ..hardware.blocks import load_block_execution
from ..hardware.vector import load_vector_execution
from ..operators.adapters import adapt_native_result
from ..operators.attention import KVCacheAccess, KVCacheSpec, _AttentionStage
from ..operators.blocks import Residual, SwiGLU
from ..operators.composite import RMSNorm, Softmax
from ..operators.embedding import Embedding
from ..operators.primitives import Gemm
from ..operators.projections import _StorageBoundary
from ..operators.resources import memory_requirements
from ..operators.rope import _RotaryTensor
from ..operators.types import TensorSpec
from ..operators.vector import Vector
from ..runtime.backends import get_backend
from ..runtime.blocks import align, gemm_storage
from ..runtime.simulation import simulate_gemm, simulate_vector
from .chip import chip_resources


@dataclass
class LocalCost:
    cycles: int | None
    actions: dict
    scratch: int
    private: dict
    macs: int = 0


@dataclass(frozen=True)
class KernelCall:
    kind: str
    args: tuple
    phase: str | None = None
    extra_workspace: int = 0


class DecoderKernels:
    def __init__(self, arch, spec, mode, *, config_dir=None):
        self.arch, self.spec, self.mode = arch, spec, mode
        self.resources = chip_resources(arch, spec)
        self.units = self.resources["units_per_chip"]
        self.backend = get_backend(arch, spec)
        self.execution = load_vector_execution(arch, config_dir=config_dir)
        self.block = load_block_execution(config_dir=config_dir)
        self.resident = arch in ("tsp", "tsp_cim")
        self.allocation = None
        self.stripe = 1
        if self.resident:
            native = spec.native_reference_spec if arch == "tsp_cim" else spec
            self.allocation = resident_allocation(native, None, None, 0, 0, label="Decoder")
            self.stripe = len(self.allocation.slices)
            self.capacity = native.mem_slice_capacity_bytes * self.stripe
            self.clock = native.clock_freq_ghz and native.clock_freq_ghz * 1e9
        elif arch == "ascend":
            self.capacity = spec.chip["hbm_usable_bytes"]
            self.clock = spec.clock_ghz and spec.clock_ghz * 1e9
        else:
            self.capacity = spec.values["hbm_usable_bytes"]
            self.clock = spec.values["clock_hz"]
        if not self.capacity:
            raise ValueError("Missing usable block storage capacity")
        self._init_run_caches()

    def _init_run_caches(self):
        self.local_run = lru_cache(maxsize=256)(self._kernel)
        self.norm_run = lru_cache(maxsize=128)(self._norm_phases)

    def for_timing(self):
        # Share the request's loaded profiles, but never reuse count-only results.
        timing = copy(self)
        timing.mode = "timing"
        timing._init_run_caches()
        return timing

    def kernel(self, kind, *args):
        return KernelCall(kind, args)

    def resolve(self, call, tile=None):
        if call.phase is not None:
            return self.norm_run(call.kind, *call.args, tile=tile)[call.phase]
        return self.local_run(call.kind, *call.args, tile=tile)

    def size(self, n):
        return align(n, self.stripe)

    def gemm_bytes(self, m, k, n):
        sizes = gemm_storage(Gemm(m, k, n), self.arch, self.spec)
        return tuple(self.size(sizes[key]) for key in ("A", "B", "C"))

    def _cost(self, result, scratch=0, macs=0):
        private = {k: v for k, v in result.resource_requirements.items() if k not in ("mem", "hbm")}
        return LocalCost(result.cycle_count, result.action_ledger.counts_dict(), scratch, private, macs)

    def _kernel(self, kind, *dims, tile=None):
        arch, spec, mode = self.arch, self.spec, self.mode
        if kind == "gemm":
            op = Gemm(*dims)
            policy = {}
            if tile is not None:
                policy = {"tile_m": tile[0]} if arch == "tsp_cim" else {"tile": tile}
            if arch == "tsp":
                policy["report_weight_buffer"] = True
            return self._cost(simulate_gemm(op, arch, spec, mode=mode, **policy), macs=op.useful_macs)
        if kind == "embedding":
            b, vocab, width = dims
            op = Embedding(TensorSpec("ids", (b,), "INT32"), TensorSpec("weight", (vocab,width), "INT8"))
            live = sum(self.size(t.nbytes) for t in (*op.inputs, *op.spec.outputs))
            path = block_embedding.make_path(op, arch, spec, self.execution, mode, live, self.capacity,
                                             self.allocation, tile)
            raw = block_embedding.simulate(op, path, self.execution, mode)
            return self._cost(adapt_native_result(op.spec, arch, raw),
                              path.memory["resident_bytes"] - live if self.resident else 0)
        if kind == "append":
            b, heads, d, length, capacity = dims
            op = KVCacheAccess(KVCacheSpec(b, heads, d, length-1, capacity), read_cache=False)
            # The graph owns the padded cache and both new-token payloads.
            path = cache_path(arch, spec, self.execution, mode, 0, self.capacity, self.allocation, tile, d)
            raw = cache_kernel(op, path, self.execution, mode)
            return self._cost(adapt_native_result(op.spec, arch, raw))
        if kind == "cache_component":
            b, width, length, cap = dims
            op = KVCacheAccess(KVCacheSpec(b, 1, width, length-1, cap), read_cache=False)
            path = cache_path(arch, spec, self.execution, mode, 0, self.capacity, self.allocation, tile, width)
            raw = simulate_payloads(op, path, self.execution, mode, phases=[("append", width, True, b)],
                                    appended_bytes=b*width, cache_read_bytes=0, model="mla_component_append",
                                    policy="Append one compressed component per batch item")
            return self._cost(adapt_native_result(op.spec, arch, raw))
        if kind == "rope":
            b, heads, d, length, capacity = dims
            x = TensorSpec("x", (b, heads, 1, d), "INT8")
            cos = TensorSpec("cos", (capacity, d//2), "FP32")
            op = _RotaryTensor(x, cos, TensorSpec("sin", cos.shape, "FP32"), length-1, "out")
            kernel = block_rope
        else:
            count = dims[0]
            def tensor(name, dtype):
                return TensorSpec(name, (count,), dtype)
            if kind == "add_partial":
                op = Vector("int32_add", (tensor("a", "INT32"), tensor("b", "INT32")))
                return self._cost(simulate_vector(op, arch, spec, mode=mode, execution=self.execution,
                                                  tile_elements=tile))
            if kind in ("residual", "fp32_add"):
                dtype = "FP32" if kind == "fp32_add" else "INT8"
                op = Residual(tensor("x", dtype), tensor("skip", dtype))
            elif kind == "swiglu":
                op = SwiGLU(tensor("gate", "INT32"), tensor("up", "INT32"))
            elif kind == "scale":
                op = _AttentionStage(tensor("x", "INT32"), "attention_scale", "FP32", "out", dims[1])
            elif kind == "scale_fp32":
                op = _AttentionStage(tensor("x", "FP32"), "attention_scale", "FP32", "out", dims[1])
            elif kind == "cast":
                op = _StorageBoundary(tensor("x", dims[1]), tensor("out", dims[2]))
            else:
                raise ValueError(f"Unknown local block kernel {kind}")
            kernel = block_elementwise
        live = sum(self.size(t.nbytes) for t in (*op.inputs, *op.spec.outputs))
        path = kernel.make_path(op, arch, spec, self.execution, mode, live, self.capacity, self.allocation, tile)
        raw = kernel.simulate(op, path, self.execution, self.block, mode)
        scratch = path.memory["resident_bytes"] - live if self.resident else 0
        return self._cost(adapt_native_result(op.spec, arch, raw), scratch)

    def _norm_phases(self, kind, width, epsilon, *, tile=None):
        """One row fragment. The caller schedules the scalar phase once per global row."""
        x = TensorSpec("x", (width,), "FP32")
        op = RMSNorm(x, TensorSpec("w", x.shape, "FP32"), epsilon) if kind == "rmsnorm" else Softmax(x)
        path = self.backend.composite_path(op, self.spec, self.execution, mode=self.mode, tile_elements=tile)
        schedule = RowSchedule(op, path, self.execution, self.mode)
        (rmsnorm if kind == "rmsnorm" else softmax)(schedule)
        # A balanced tree between units uses the same local read/compute/write
        # and dependency cost as one binary merge. Inter-unit transport is free.
        for action in (("fp32_add",) if kind == "rmsnorm" else ("fp32_max", "fp32_add")):
            schedule.group("merge_" + action)
            schedule.step("merge", [("local", 1), ("peer", 1)], [("state", 1)], action, 1, reduction=True)
        counts, cycles = {}, Counter()
        for group in schedule.groups:
            phase = group["phase"]
            amounts = counts.setdefault(phase, Counter())
            for stage in group["stages"]:
                repeats = group["repetitions_per_row"]
                amounts.update({k: v*repeats for k, v in stage["counts_per_execution"].items()})
                if self.mode == "timing":
                    cycles[phase] += repeats * sum(stage["cycles_per_execution"].values())
        private = memory_requirements(path.memory)
        private.pop("hbm", None)
        private.pop("mem", None)
        scratch = 0
        if self.resident:
            scratch = self.size(path.memory["scratch_tile_bytes"]) + self.size(path.memory["row_state_bytes"])
        phases = {}
        for phase, amounts in counts.items():
            actions = {}
            for key, value in amounts.items():
                if key in path.action_specs:
                    component, name, category, basis = path.action_specs[key]
                    unit = "byte"
                else:
                    component, name, basis, unit = ("vxm" if self.resident else "vector"), key, "logical", "op"
                    category = "nonlinear" if key in ("fp32_exp", "fp32_rcp", "fp32_rsqrt") else "vector"
                actions[key] = dict(component=component, name=name, category=category, basis=basis,
                                    unit=unit, count=value, dtype="FP32", role="action", status="available")
            phases[phase] = LocalCost(cycles[phase] if self.mode == "timing" else None, actions, scratch, private)
        return phases
