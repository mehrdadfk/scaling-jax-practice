"""Hardware specs for common accelerators.

Figures are per-chip, peak (not sustained), for the bf16/fp16 numerics used
in most large-model training. Sources: vendor spec sheets and the scaling
book's own hardware appendix. Treat these as approximate ceilings, not
guarantees — real kernels rarely hit peak FLOPs or peak bandwidth.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ChipSpec:
    name: str
    peak_flops_bf16: float  # FLOP/s, dense (no sparsity)
    hbm_bytes: float  # total HBM capacity per chip
    hbm_bandwidth_bytes_per_sec: float
    interconnect_bandwidth_bytes_per_sec: float  # per-link, one direction

    @property
    def ridge_point(self) -> float:
        """Arithmetic intensity (FLOPs/byte) where compute and HBM bandwidth balance.

        Below this intensity, an op is memory-bound; above it, compute-bound.
        """
        return self.peak_flops_bf16 / self.hbm_bandwidth_bytes_per_sec


CHIPS: dict[str, ChipSpec] = {
    "tpu-v5e": ChipSpec(
        name="TPU v5e",
        peak_flops_bf16=197e12,
        hbm_bytes=16e9,
        hbm_bandwidth_bytes_per_sec=819e9,
        interconnect_bandwidth_bytes_per_sec=200e9,
    ),
    "tpu-v5p": ChipSpec(
        name="TPU v5p",
        peak_flops_bf16=459e12,
        hbm_bytes=95e9,
        hbm_bandwidth_bytes_per_sec=2765e9,
        interconnect_bandwidth_bytes_per_sec=400e9,
    ),
    "tpu-v6e": ChipSpec(
        name="TPU v6e (Trillium)",
        peak_flops_bf16=918e12,
        hbm_bytes=32e9,
        hbm_bandwidth_bytes_per_sec=1640e9,
        interconnect_bandwidth_bytes_per_sec=400e9,
    ),
    "h100": ChipSpec(
        name="NVIDIA H100 SXM",
        peak_flops_bf16=989e12,
        hbm_bytes=80e9,
        hbm_bandwidth_bytes_per_sec=3350e9,
        interconnect_bandwidth_bytes_per_sec=450e9,  # NVLink, per-direction share
    ),
    "h200": ChipSpec(
        name="NVIDIA H200 SXM",
        peak_flops_bf16=989e12,
        hbm_bytes=141e9,
        hbm_bandwidth_bytes_per_sec=4800e9,
        interconnect_bandwidth_bytes_per_sec=450e9,
    ),
    "a100": ChipSpec(
        name="NVIDIA A100 SXM",
        peak_flops_bf16=312e12,
        hbm_bytes=80e9,
        hbm_bandwidth_bytes_per_sec=2039e9,
        interconnect_bandwidth_bytes_per_sec=300e9,
    ),
}


def get_chip(name: str) -> ChipSpec:
    key = name.strip().lower()
    if key not in CHIPS:
        available = ", ".join(sorted(CHIPS))
        raise KeyError(f"Unknown chip {name!r}. Available: {available}")
    return CHIPS[key]
