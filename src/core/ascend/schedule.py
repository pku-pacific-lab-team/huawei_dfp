"""Tile-level FIFO queues, shared services and explicit buffer lifetimes.

This is an analytical list schedule, not a cycle-accurate instruction simulator.
The L0C export queue is a model assumption, not an asserted hardware MTE mapping.
"""
from dataclasses import dataclass
import heapq


# The first resource is the FIFO queue. HBM read/write queues also contend for
# one non-preemptive payload service; A/B loads share MTE1 rather than two ports.
RESOURCES = {
    "hbm_to_l2": ("mte2", "hbm"),
    "l2_fill": ("mte2", "l2"),
    "l2_to_l1": ("mte2", "l2"),
    "hbm_to_l1": ("mte2", "hbm"),
    "l1_l0a": ("mte1",),
    "l1_l0b": ("mte1",),
    "compute": ("cube",),
    "l0c_ub": ("l0c_export",),
    "ub_to_hbm": ("mte3", "hbm"),
}


@dataclass(frozen=True)
class Task:
    stage: str
    cycles: int
    dependencies: tuple
    coordinates: dict


class Schedule:
    def __init__(self, resources=None):
        self.resources = RESOURCES if resources is None else resources
        self.tasks = []
        self.queue_tail = {}
        self.lifetimes = []

    def add(self, stage, cycles, dependencies=(), **coordinates):
        if type(cycles) is not int or cycles <= 0:
            raise ValueError("Task service must be a positive integer number of cycles")
        queue = self.resources[stage][0]
        deps = {d for d in dependencies if d is not None}
        if queue in self.queue_tail:
            deps.add(self.queue_tail[queue])
        index = len(self.tasks)
        if any(type(d) is not int or not 0 <= d < index for d in deps):
            raise ValueError("Dependencies must refer to earlier tasks")
        self.tasks.append(Task(stage, cycles, tuple(sorted(deps)), coordinates))
        self.queue_tail[queue] = index
        return index

    def hold(self, buffer, slot, size, acquire, *releases):
        self.lifetimes.append((buffer, slot, size, acquire, releases))

    def run(self, trace=False):
        count = len(self.tasks)
        successors = [[] for _ in self.tasks]
        pending = [len(t.dependencies) for t in self.tasks]
        ready_at = [0] * count
        starts, ends = [0] * count, [0] * count
        available = {r: 0 for rs in self.resources.values() for r in rs}
        busy = dict.fromkeys(available, 0)
        ready = []
        for index, task in enumerate(self.tasks):
            for dep in task.dependencies:
                successors[dep].append(index)
            if not pending[index]:
                heapq.heappush(ready, (0, index))
        scheduled = 0
        while ready:
            earliest, index = heapq.heappop(ready)
            task = self.tasks[index]
            resources = self.resources[task.stage]
            start = max(ready_at[index], *(available[r] for r in resources))
            # Queue keys are lower bounds. Reinsert after resource availability
            # changes, so a future output cannot block an earlier ready prefetch.
            if start != earliest:
                heapq.heappush(ready, (start, index))
                continue
            end = start + task.cycles
            starts[index], ends[index] = start, end
            for resource in resources:
                available[resource] = end
                busy[resource] += task.cycles
            for child in successors[index]:
                ready_at[child] = max(ready_at[child], end)
                pending[child] -= 1
                if not pending[child]:
                    heapq.heappush(ready, (ready_at[child], child))
            scheduled += 1
        if scheduled != count:
            raise ValueError("Cyclic Ascend task dependencies")
        total = max(ends, default=0)
        live_changes = {}
        lifetimes = []
        for buffer, slot, size, acquire, releases in self.lifetimes:
            release = max(releases, key=ends.__getitem__)
            start, end = starts[acquire], ends[release]
            changes = live_changes.setdefault(buffer, [])
            changes.extend(((start, size), (end, -size)))
            if trace:
                lifetimes.append({"buffer": buffer, "slot": slot, "bytes": size,
                                  "acquire_task": acquire, "release_task": release,
                                  "start_cycle": start, "end_cycle": end})
        peaks = {}
        for buffer, changes in live_changes.items():
            live = peak = 0
            # Release before acquire at the same cycle (half-open intervals).
            for _, delta in sorted(changes):
                live += delta
                peak = max(peak, live)
            peaks[buffer] = peak
        result = {
            "total_cycles": total, "task_count": count,
            "dispatch": "FIFO per queue; earliest feasible start, task-id tie break; non-preemptive services",
            "resource_busy_cycles": busy,
            "resource_utilization": {r: value / total if total else 0 for r, value in busy.items()},
            "buffer_live_peak_bytes": peaks,
        }
        if trace:
            result["events"] = [
                {"id": i, "stage": task.stage, "resources": list(self.resources[task.stage]),
                 "dependencies": list(task.dependencies), "start_cycle": starts[i],
                 "end_cycle": ends[i], **task.coordinates}
                for i, task in enumerate(self.tasks)
            ]
            result["buffer_lifetimes"] = lifetimes
        return result
