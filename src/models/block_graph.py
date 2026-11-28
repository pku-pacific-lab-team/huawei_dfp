"""Materialized tensor backings and serial operator dependencies on one TP rank."""
from dataclasses import asdict, dataclass

from ..core.capacity import CapacityExceeded
from ..operators.types import TensorSpec


@dataclass
class BlockCapacityError(ValueError):
    chip: int
    unit: int | None
    level: str
    node: str
    required_bytes: int
    capacity_bytes: int
    tensors: dict

    def __str__(self):
        return (f"chip {self.chip}, unit {self.unit}, {self.level} at {self.node}: "
                f"requires {self.required_bytes} B, available {self.capacity_bytes} B")

    def to_dict(self):
        return asdict(self)


def widths(n, units):
    base, tail = divmod(n, units)
    return [base + int(i < tail) for i in range(units)]


class BlockGraph:
    def __init__(self, kernels, rank):
        self.k, self.rank = kernels, rank
        self.tensors, self.producers, self.nodes = {}, {}, []
        self.program = []
        self.reserved = [0] * kernels.units  # Other model tensors resident on this chip.

    def tensor(self, name, shape, dtype, per_unit, role="activation"):
        spec = TensorSpec(name, tuple(shape), dtype)
        sizes = [self.k.size(n) for n in per_unit]
        if name in self.tensors or len(sizes) != self.k.units or min(sizes) < 0:
            raise ValueError(f"Invalid block tensor allocation: {name}")
        self.tensors[name] = dict(**spec.to_dict(), role=role, bytes_per_unit=sizes)
        if role != "activation":
            self.producers[name] = None
        return name

    def reserve(self, name, sizes):
        current = self.tensors[name]["bytes_per_unit"]
        self.tensors[name]["bytes_per_unit"] = [max(a, self.k.size(b)) for a, b in zip(current, sizes, strict=True)]

    def emit(self, name, inputs, outputs, phases=(), *, updates=(), **details):
        # Each phase is a concurrent wave. Whole waves (and repeated rows) are serial.
        if not all(t in self.producers for t in inputs) or any(t in self.producers for t in outputs):
            raise ValueError(f"Invalid block dependency at {name}")
        deps = list(dict.fromkeys(self.producers[t] for t in inputs if self.producers[t] is not None))
        self.nodes.append(dict(id=name, inputs=list(inputs), outputs=list(outputs), updates=list(updates),
                               depends_on=deps, scratch_per_unit=[0]*self.k.units, **details))
        self.program.append(phases)
        for t in (*outputs, *updates):
            self.producers[t] = name

    def execute_node(self, node, phases, tile=None):
        actions, private, stages = {}, {}, []
        scratch = [0] * self.k.units
        live_bytes = [self.reserved[i] + sum(self.tensors[t]["bytes_per_unit"][i] for t in node["live_tensors"])
                      for i in range(self.k.units)]
        cycles, macs = 0, 0
        for label, runs, repeats in phases:
            elapsed = 0
            for i, call in enumerate(runs):
                if call is None:
                    continue
                try:
                    run = self.k.resolve(call, tile)
                    if tile is None and self.k.resident and run.scratch:
                        available = self.k.capacity - live_bytes[i] - call.extra_workspace
                        fallback = self.k.execution.policy["tile_elements"]
                        minimum = 2 if call.kind == "rope" else 1
                        while run.scratch > available and fallback > minimum:
                            fallback = max(minimum, fallback // 2)
                            run = self.k.resolve(call, fallback)
                except CapacityExceeded as exc:
                    raise BlockCapacityError(self.rank, i, exc.level, node["id"], exc.required_bytes,
                                             exc.capacity_bytes, {node["id"]+".workspace": exc.required_bytes}) from exc
                except ValueError as exc:
                    raise ValueError(f"chip {self.rank}, unit {i}, {node['id']}, {call.kind}{call.args}: {exc}") from exc
                if self.k.mode == "timing" and run.cycles is None:
                    raise ValueError(f"Missing local timing for {node['id']} on {self.k.arch}")
                elapsed = max(elapsed, run.cycles or 0)
                macs += run.macs * repeats
                scratch[i] = max(scratch[i], run.scratch + call.extra_workspace)
                for level, req in run.private.items():
                    if req["capacity_bytes"] is not None and req["required_bytes"] > req["capacity_bytes"]:
                        raise BlockCapacityError(self.rank, i, level, node["id"], req["required_bytes"],
                                                 req["capacity_bytes"], {"workspace": req["required_bytes"]})
                    if level not in private or req["required_bytes"] > private[level]["required_bytes"]:
                        private[level] = dict(req, unit_id=i)
                for key, row in run.actions.items():
                    entry = actions.setdefault(key, {**row, "scope": "chip", "count": 0})
                    if row["count"] is None or entry["count"] is None:
                        entry.update(count=None, status="unavailable")
                    else:
                        entry["count"] += row["count"] * repeats
            cycles += elapsed * repeats
            stages.append({"phase": label, "active_units": sum(r is not None for r in runs),
                           "repetitions": repeats, "cycles_per_wave": elapsed if self.k.mode == "timing" else None})
        self.check_node_capacity(node, [a + b for a, b in zip(live_bytes, scratch)], scratch)
        node.update(phases=stages, useful_macs=macs, action_counts=actions,
                    local_compute_cycles=cycles if self.k.mode == "timing" else None,
                    scratch_per_unit=scratch, private_memory=private)

    def check_node_capacity(self, node, lanes, scratch):
        unit = lanes.index(max(lanes)) if self.k.resident else None
        def charged(values):
            return sum(values) if unit is None else values[unit]
        required = charged(lanes)
        if required > self.k.capacity:
            contributions = {t: charged(self.tensors[t]["bytes_per_unit"]) for t in node["live_tensors"]}
            contributions["workspace"] = charged(scratch)
            if any(self.reserved):
                contributions["other_resident_tensors"] = charged(self.reserved)
            raise BlockCapacityError(self.rank, unit, "HBM" if unit is None else "MEM",
                                     node["id"], required, self.k.capacity, contributions)

    def capacity(self):
        intervals = {}
        for index, node in enumerate(self.nodes):
            for t in (*node["inputs"], *node["outputs"], *node["updates"]):
                intervals.setdefault(t, [index, index])[1] = index
        for t, tensor in self.tensors.items():
            if tensor["role"] in ("weight", "kv_cache", "rope_table"):
                intervals[t] = [0, len(self.nodes)-1]
            elif tensor["role"] == "input":
                intervals[t][0] = 0
        peak = None
        unit_peaks = [0] * self.k.units
        for index, node in enumerate(self.nodes):
            live = [t for t, (start, stop) in intervals.items() if start <= index <= stop]
            lanes = [a+b for a, b in zip(self.reserved, node["scratch_per_unit"], strict=True)]
            for t in live:
                lanes = [a+b for a, b in zip(lanes, self.tensors[t]["bytes_per_unit"])]
            unit_peaks = [max(a,b) for a,b in zip(unit_peaks, lanes)]
            node["live_tensors"] = live
            node["required_bytes"] = sum(lanes)
            self.check_node_capacity(node, lanes, node["scratch_per_unit"])
            if peak is None or sum(lanes) > peak["required_bytes"]:
                peak = {"node": node["id"], "required_bytes": sum(lanes), "live_tensors": live,
                        "workspace_bytes": sum(node["scratch_per_unit"])}
        return {"level": "MEM" if self.k.resident else "HBM", "peak": peak,
                "capacity_bytes": self.k.capacity * (self.k.units if self.k.resident else 1),
                "unit_peak_bytes": unit_peaks, "unit_capacity_bytes": self.k.capacity if self.k.resident else None}

    def execute(self, *, search_tiles=False):
        if search_tiles:
            from .software_tiles import select_node_tile
            # Detach the baseline ledger and preflight live backings before
            # assigning workspace for this run, including after a failed trial.
            self.nodes = [dict(n, scratch_per_unit=[0] * self.k.units) for n in self.nodes]
        self.capacity()
        for node, phases in zip(self.nodes, self.program, strict=True):
            if search_tiles:
                select_node_tile(self, node, phases)
            else:
                self.execute_node(node, phases)
        capacity = self.capacity()
        cycles = None if self.k.mode == "count" else sum(n["local_compute_cycles"] for n in self.nodes)
        return dict(nodes=self.nodes, tensors=self.tensors, capacity=capacity,
                    useful_macs=sum(n["useful_macs"] for n in self.nodes), local_compute_cycles=cycles,
                    local_compute_seconds=cycles/self.k.clock if cycles is not None and self.k.clock else None)
