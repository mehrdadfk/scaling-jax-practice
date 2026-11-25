import argparse

from scalecalc.chips import CHIPS, get_chip
from scalecalc.model_math import ModelConfig, model_flops_per_step, model_memory_bytes, param_count
from scalecalc.roofline import classify, mfu


def _human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PB"


def _human_flops(n: float) -> str:
    for unit in ("FLOP/s", "KFLOP/s", "MFLOP/s", "GFLOP/s", "TFLOP/s"):
        if n < 1000:
            return f"{n:.2f} {unit}"
        n /= 1000
    return f"{n:.2f} PFLOP/s"


def cmd_describe(args: argparse.Namespace) -> None:
    cfg = ModelConfig(
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        d_ff=args.d_ff,
    )
    chip = get_chip(args.chip)

    n = param_count(cfg)
    flops = model_flops_per_step(cfg)
    mem = model_memory_bytes(cfg)

    print(f"Model: {n / 1e9:.2f}B params")
    print(f"  FLOPs/step (fwd+bwd): {flops:.3e}")
    print("  Memory breakdown (single device, unsharded):")
    for k, v in mem.items():
        print(f"    {k:15s} {_human_bytes(v)}")

    print(f"\nChip: {chip.name}")
    print(f"  Peak bf16: {_human_flops(chip.peak_flops_bf16)}")
    print(f"  HBM: {_human_bytes(chip.hbm_bytes)} @ {_human_bytes(chip.hbm_bandwidth_bytes_per_sec)}/s")
    print(f"  Ridge point: {chip.ridge_point:.1f} FLOPs/byte")

    fits = mem["total"] <= chip.hbm_bytes
    print(f"\nFits on single chip: {'yes' if fits else 'no'}")
    if not fits:
        n_chips = mem["total"] / chip.hbm_bytes
        print(f"  Needs >= {n_chips:.1f} chips just for memory (before considering compute time)")

    verdict = classify(chip, flops=flops, bytes_moved=mem["total"])
    step_time = flops / chip.peak_flops_bf16
    print(f"\nRoofline verdict: {verdict.bound.value}")
    print(f"  Arithmetic intensity: {verdict.arithmetic_intensity:.1f} FLOPs/byte")
    print(f"  Lower-bound step time: {step_time * 1e3:.3f} ms")
    print(f"  MFU at that step time: {mfu(chip, flops, step_time) * 100:.1f}%")


def cmd_chips(args: argparse.Namespace) -> None:
    for key, chip in sorted(CHIPS.items()):
        print(f"{key:10s} {chip.name:24s} peak={chip.peak_flops_bf16 / 1e12:.0f} TFLOP/s  "
              f"hbm={chip.hbm_bytes / 1e9:.0f} GB  bw={chip.hbm_bandwidth_bytes_per_sec / 1e9:.0f} GB/s")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="scalecalc")
    sub = parser.add_subparsers(dest="command", required=True)

    p_chips = sub.add_parser("chips", help="list known hardware specs")
    p_chips.set_defaults(func=cmd_chips)

    p_describe = sub.add_parser("describe", help="describe a model's FLOPs/memory/roofline on a chip")
    p_describe.add_argument("--d-model", type=int, required=True)
    p_describe.add_argument("--num-layers", type=int, required=True)
    p_describe.add_argument("--num-heads", type=int, required=True)
    p_describe.add_argument("--vocab-size", type=int, default=128_000)
    p_describe.add_argument("--seq-len", type=int, default=4096)
    p_describe.add_argument("--batch-size", type=int, default=1)
    p_describe.add_argument("--d-ff", type=int, default=None)
    p_describe.add_argument("--chip", type=str, default="tpu-v5e")
    p_describe.set_defaults(func=cmd_describe)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
