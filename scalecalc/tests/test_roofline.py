import pytest

from scalecalc.chips import CHIPS
from scalecalc.roofline import Bound, classify, mfu


def test_high_intensity_workload_is_compute_bound():
    chip = CHIPS["tpu-v5e"]
    # Way above ridge point: huge FLOPs, tiny bytes moved.
    verdict = classify(chip, flops=1e15, bytes_moved=1e6)
    assert verdict.bound is Bound.COMPUTE
    assert verdict.achievable_flops == chip.peak_flops_bf16


def test_low_intensity_workload_is_memory_bound():
    chip = CHIPS["tpu-v5e"]
    # Way below ridge point: tiny FLOPs, huge bytes moved.
    verdict = classify(chip, flops=1e6, bytes_moved=1e12)
    assert verdict.bound is Bound.MEMORY
    assert verdict.achievable_flops < chip.peak_flops_bf16


def test_ridge_point_is_boundary():
    chip = CHIPS["tpu-v5e"]
    bytes_moved = 1e9
    flops_at_ridge = chip.ridge_point * bytes_moved
    verdict = classify(chip, flops=flops_at_ridge, bytes_moved=bytes_moved)
    assert verdict.bound is Bound.COMPUTE  # tie-break toward compute-bound at exact ridge point


def test_classify_rejects_zero_bytes():
    chip = CHIPS["tpu-v5e"]
    with pytest.raises(ValueError):
        classify(chip, flops=1e9, bytes_moved=0)


def test_mfu_is_one_at_peak_flops():
    chip = CHIPS["h100"]
    flops = 1e15
    time_seconds = flops / chip.peak_flops_bf16
    assert mfu(chip, flops, time_seconds) == pytest.approx(1.0)


def test_mfu_below_peak_is_less_than_one():
    chip = CHIPS["h100"]
    flops = 1e15
    time_seconds = 2 * (flops / chip.peak_flops_bf16)
    assert mfu(chip, flops, time_seconds) == pytest.approx(0.5)
