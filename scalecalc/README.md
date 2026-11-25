# scalecalc

An executable companion to the [scaling book](https://jax-ml.github.io/scaling-book/)'s
roofline math: model FLOPs/memory, communication cost, and parallelism
strategy recommendations, so you can plug in a real model + cluster config
and get concrete numbers instead of doing the arithmetic by hand.

## Status

Phase 1, milestone 1: chip database, dense-transformer parameter/FLOPs/memory
math, and single-device roofline classification (compute-bound vs
memory-bound, MFU).

Planned next:
- Sharding cost model (DP/FSDP/TP/PP/EP) and a strategy recommender that
  sweeps mesh shapes for a Pareto frontier of memory vs. step time.
- An interactive web UI reproducing the book's key figures, parametrized by
  your own model.
- A Phase 2 profiler that captures real JAX/XLA traces and overlays them on
  this same roofline model to flag underutilized ops.

## Install

```bash
pip install -e ".[dev]"
```

## Usage

```bash
scalecalc chips

scalecalc describe \
  --d-model 4096 --num-layers 32 --num-heads 32 \
  --vocab-size 128000 --seq-len 4096 --batch-size 8 \
  --chip tpu-v5p
```

```python
from scalecalc import ModelConfig, model_flops_per_step, model_memory_bytes, classify, CHIPS

cfg = ModelConfig(d_model=4096, num_layers=32, num_heads=32, vocab_size=128_000, seq_len=4096, batch_size=8)
flops = model_flops_per_step(cfg)
mem = model_memory_bytes(cfg)
verdict = classify(CHIPS["tpu-v5p"], flops=flops, bytes_moved=mem["total"])
print(verdict.bound, verdict.arithmetic_intensity)
```

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT





Phase 1: "ScaleCalc" — interactive scaling-book companion

What it does: you plug in a model config (d_model, layers, heads, vocab, seq_len, batch) and a cluster config (chip type, count, mesh shape), it tells you memory footprint, FLOPs/step, communication volume per parallelism strategy (DP/FSDP/TP/PP/EP), and which one wins under your constraints — basically an executable version of the book's roofline math.

Structure:
scalecalc/
  chips.py       # known HW specs: peak FLOPs, HBM BW, interconnect BW (TPU v5e/v5p/v6e, H100, etc.)
  model_math.py  # param count, activation memory, FLOPs/step
  comm_cost.py   # all-reduce/all-gather/reduce-scatter/all-to-all volume + time formulas per strategy
  roofline.py    # arithmetic intensity, compute-vs-comm-bound verdict, MFU estimate
  recommend.py   # sweep mesh shapes, return Pareto frontier (memory vs step time)
  cli.py
  webapp/        # marimo or streamlit, sliders like the book's interactive figures

Milestones:
1. Chip DB + single-device memory/FLOPs calc + CLI
2. Sharding cost model (DP/FSDP/TP/PP) + strategy comparison table + recommender
3. Interactive UI reproducing the book's key plots, parametrized by your own model

Phase 2: Roofline profiler on real traces

What it does: wraps an actual JAX training/inference run, captures a real XLA trace, pulls per-HLO-op FLOPs/bytes via cost_analysis(), plots measured ops against the roofline ceiling from Phase 1's chip DB, and flags which ops are leaving MFU on the table — then cross-checks against Phase 1's predicted comm cost to suggest concrete sharding changes.

Structure:
roofline/
  capture.py   # jax.profiler.trace() wrapper, CLI: jaxroof profile -- python train.py
  hlo_cost.py  # jax.jit(...).lower(...).compile().cost_analysis() per-op FLOPs/bytes
  classify.py  # compute-bound / memory-bound / comm-bound per op, using scalecalc.chips
  report.py    # interactive HTML report (plotly), diffs measured vs scalecalc's predicted comm cost

Milestones:
4. Trace capture + HLO cost extraction, single-op roofline
5. Aggregate-by-op-type report + comm-bound flagging
6. Link to Phase 1 predictions ("measured all-gather matchey TP instead")

Cross-cutting

- Regression tests: assert calculator output matches known inchilla-style FLOPs, known TPU pod comm costs) so the mathis trustworthy before anyone builds on it.
- Docs/examples reproduce specific scaling-book figures — g launch.
