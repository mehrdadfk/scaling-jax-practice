"""Roofline classification: is an op/workload compute-bound or memory-bound?

Core idea from the scaling book: arithmetic intensity (FLOPs/byte moved)
compared against a chip's ridge point (peak_flops / hbm_bandwidth) tells you
which resource is the bottleneck. Below the ridge point, you're memory-bound
and more FLOPs are "free"; above it, you're compute-bound and the chip is
fully utilized (assuming perfect overlap).
"""

from dataclasses import dataclass
from enum import Enum

from scalecalc.chips import ChipSpec
from scalecalc.result import Err, Ok, Result


class Bound(Enum):
    COMPUTE = "compute-bound"
    MEMORY = "memory-bound"


@dataclass(frozen=True)
class RooflineVerdict:
    bound: Bound
    arithmetic_intensity: float  # FLOPs/byte
    ridge_point: float
    achievable_flops: float  # FLOP/s ceiling given this workload's intensity
    time_seconds: float  # wall-clock lower bound to execute flops/bytes on this chip


def classify(chip: ChipSpec, flops: float, bytes_moved: float) -> RooflineVerdict:
    if bytes_moved <= 0:
        raise ValueError("bytes_moved must be positive")

    intensity = flops / bytes_moved
    ridge = chip.ridge_point

    if intensity >= ridge:
        bound = Bound.COMPUTE
        achievable_flops = chip.peak_flops_bf16
        time_seconds = flops / chip.peak_flops_bf16
    else:
        bound = Bound.MEMORY
        achievable_flops = intensity * chip.hbm_bandwidth_bytes_per_sec
        time_seconds = bytes_moved / chip.hbm_bandwidth_bytes_per_sec

    return RooflineVerdict(
        bound=bound,
        arithmetic_intensity=intensity,
        ridge_point=ridge,
        achievable_flops=achievable_flops,
        time_seconds=time_seconds,
    )


def mfu(chip: ChipSpec, flops: float, time_seconds: float) -> float:
    """Model FLOPs Utilization: achieved FLOP/s as a fraction of peak."""
    if time_seconds <= 0:
        raise ValueError("time_seconds must be positive")
    return (flops / time_seconds) / chip.peak_flops_bf16


def try_classify(chip: ChipSpec, flops: float, bytes_moved: float) -> Result[RooflineVerdict, ValueError]:
    """Non-raising variant of classify(): callers handle the invalid-input case explicitly."""
    if bytes_moved <= 0:
        return Err(ValueError("bytes_moved must be positive"))
    return Ok(classify(chip, flops, bytes_moved))
