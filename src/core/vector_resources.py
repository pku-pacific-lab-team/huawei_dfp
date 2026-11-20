"""Shared Vector resource assembly, independent of any operator schedule."""
from dataclasses import dataclass
from fractions import Fraction

from .vector import positive


@dataclass(frozen=True)
class LocalServices:
    hbm_bpc: Fraction | None
    read_bpc: Fraction | None
    write_bpc: Fraction | None


def ascend_services(spec, execution, timed):
    if not timed:
        return LocalServices(None, None, None)
    clock = positive(spec.clock_ghz, "clock_freq_ghz") * 10**9
    hbm = positive(spec.chip["hbm_bandwidth_bytes_per_second"], "chip.hbm_bandwidth_bytes_per_second") / spec.architectural_cores / clock
    return LocalServices(hbm, positive(execution.values["ascend_ub_read_bpc"], "ascend_ub_read_bpc"),
                         positive(execution.values["ascend_ub_write_bpc"], "ascend_ub_write_bpc"))


def tpu_services(spec, timed):
    if not timed:
        return LocalServices(None, None, None)
    v = spec.values
    clock = positive(v["clock_hz"], "clock_hz")
    hbm = positive(v["hbm_bandwidth_bytes_per_second"], "hbm_bandwidth_bytes_per_second") / positive(v["tensorcores_per_chip"], "tensorcores_per_chip") / clock
    local = positive(v["vmem_bandwidth_bytes_per_second"], "vmem_bandwidth_bytes_per_second") / clock
    return LocalServices(hbm, local, local)


@dataclass(frozen=True)
class ResidentAllocation:
    slices: tuple[int, ...]
    streams: int
    reserved_bytes: int
    hemisphere: int


def resident_allocation(spec, mem_slices, streams, reserved_bytes, hemisphere, *, label):
    slices = tuple(range(spec.n_mem_slices_per_hemisphere)) if mem_slices is None else mem_slices
    if not isinstance(slices, (tuple, list)) or not slices or any(type(s) is not int or not 0 <= s < spec.n_mem_slices_per_hemisphere for s in slices) or len(set(slices)) != len(slices):
        raise ValueError(f"{label} MEM slices must be distinct valid local indices")
    streams = spec.n_streams_per_direction if streams is None else streams
    if type(streams) is not int or not 1 <= streams <= spec.n_streams_per_direction:
        raise ValueError(f"{label} stream allocation exceeds directional inventory")
    if type(reserved_bytes) is not int or reserved_bytes < 0 or type(hemisphere) is not int or not 0 <= hemisphere < spec.n_hemispheres:
        raise ValueError("Invalid reserved bytes or hemisphere")
    return ResidentAllocation(tuple(slices), streams, reserved_bytes, hemisphere)


@dataclass(frozen=True)
class ResidentServices:
    read_bpc: Fraction | None
    write_bpc: Fraction | None
    stream_bpc: Fraction | None


def resident_services(spec, allocation, timed):
    if not timed:
        return ResidentServices(None, None, None)
    return ResidentServices(positive(spec.values["mem_read_bpc"], "mem_read_bpc") * len(allocation.slices),
                            positive(spec.values["mem_write_bpc"], "mem_write_bpc") * len(allocation.slices),
                            Fraction(allocation.streams * spec.stream_width_bytes))
