"""Explicit local-DMA and resident-MEM paths for row-kernel accounting."""
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from typing import Callable, ClassVar

from .vector import cycles


class MemoryKind(Enum):
    LOCAL_DMA = "local_dma"
    RESIDENT = "resident_mem"


@dataclass(kw_only=True)
class MemoryPath(ABC):
    spec: object
    timed: bool
    tile: int
    requested_tile: int
    clock: Fraction | int | float | None
    startup: int
    compute: Callable[[dict], int]
    budget: int | float | None
    compute_prefix: str
    memory: dict
    boundary: dict
    provenance: dict
    references: dict
    services: dict
    required: list[str]
    read_bpc: Fraction | None
    write_bpc: Fraction | None
    kind: ClassVar[MemoryKind]

    @abstractmethod
    def load(self, size): ...

    @abstractmethod
    def export(self, size): ...

    @abstractmethod
    def access(self, sizes, direction): ...

    @property
    @abstractmethod
    def action_specs(self): ...

    @property
    @abstractmethod
    def temporary_capacity_bytes(self): ...


@dataclass(kw_only=True)
class LocalMemoryPath(MemoryPath):
    local: str
    hbm_bpc: Fraction | None
    kind: ClassVar[MemoryKind] = MemoryKind.LOCAL_DMA

    @property
    def temporary_capacity_bytes(self):
        return self.memory[self.local]["allocated_bytes"]

    def load(self, size):
        counts = Counter(hbm_read_bytes=size, **{f"{self.local}_dma_write_bytes": size})
        time = cycles(size, min(self.hbm_bpc, self.write_bpc), self.startup) if self.timed else 0
        return counts, time

    def export(self, size):
        counts = Counter(hbm_write_bytes=size, **{f"{self.local}_dma_read_bytes": size})
        time = cycles(size, min(self.hbm_bpc, self.read_bpc), self.startup) if self.timed else 0
        return counts, time

    def access(self, sizes, direction):
        size = sum(sizes)
        if not size:
            return Counter(), 0
        rate = self.read_bpc if direction == "read" else self.write_bpc
        return Counter({f"{self.local}_{direction}_bytes": size}), cycles(size, rate) if self.timed else 0

    @property
    def action_specs(self):
        return {"hbm_read_bytes": ("hbm", "read", "offchip_memory", "logical"),
                "hbm_write_bytes": ("hbm", "write", "offchip_memory", "logical"),
                **{f"{self.local}_{phase}{direction}_bytes": (self.local, direction, "onchip_memory", "logical")
                   for phase in ("", "dma_") for direction in ("read", "write")}}


@dataclass(kw_only=True)
class ResidentMemoryPath(MemoryPath):
    width: int
    stream_bpc: Fraction | None
    kind: ClassVar[MemoryKind] = MemoryKind.RESIDENT

    @property
    def temporary_capacity_bytes(self):
        return self.memory["scratch_tile_bytes"] + self.memory["row_state_bytes"]

    def load(self, size):
        # The source already resides in MEM; its consumer accounts the read.
        return Counter(), 0

    def export(self, size):
        # The producer wrote the MEM destination directly.
        return Counter(), 0

    def access(self, sizes, direction):
        counts, time = Counter(), 0
        for size in sizes:
            if not size:
                continue
            padded = (size + self.width - 1) // self.width * self.width
            counts[f"mem_{direction}_bytes"] += size
            counts["input_stream_bytes" if direction == "read" else "output_stream_bytes"] += padded
            if self.timed:
                rate = self.read_bpc if direction == "read" else self.write_bpc
                time += max(cycles(size, rate), cycles(padded, self.stream_bpc)) + self.startup
        return counts, time

    @property
    def action_specs(self):
        return {"mem_read_bytes": ("mem", "read", "onchip_memory", "logical"),
                "mem_write_bytes": ("mem", "write", "onchip_memory", "logical"),
                "input_stream_bytes": ("stream_to_vxm", "transfer", "interconnect", "padded"),
                "output_stream_bytes": ("stream_from_vxm", "transfer", "interconnect", "padded")}
