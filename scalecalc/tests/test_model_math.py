from scalecalc.model_math import ModelConfig, model_flops_per_step, model_memory_bytes, param_count


def test_param_count_matches_gpt3_175b():
    # GPT-3 175B config: d_model=12288, 96 layers, 96 heads, d_ff=4*d_model, vocab=50257.
    cfg = ModelConfig(
        d_model=12288,
        num_layers=96,
        num_heads=96,
        vocab_size=50257,
        seq_len=2048,
        batch_size=1,
    )
    n = param_count(cfg)
    # Published param count is ~175B; our dense approximation should land within 5%.
    assert 0.95 * 175e9 <= n <= 1.05 * 175e9


def test_flops_per_step_scales_with_batch_and_tokens():
    base = ModelConfig(d_model=512, num_layers=4, num_heads=8, vocab_size=1000, seq_len=128, batch_size=1)
    doubled_batch = ModelConfig(**{**base.__dict__, "batch_size": 2})

    flops_base = model_flops_per_step(base, include_attention_quadratic=False)
    flops_doubled = model_flops_per_step(doubled_batch, include_attention_quadratic=False)

    # Dense 6N*tokens term is linear in batch size when attention quadratic term is excluded.
    assert flops_doubled == 2 * flops_base


def test_attention_quadratic_term_grows_with_seq_len():
    short = ModelConfig(d_model=512, num_layers=4, num_heads=8, vocab_size=1000, seq_len=128, batch_size=1)
    long = ModelConfig(**{**short.__dict__, "seq_len": 4096})

    short_with_attn = model_flops_per_step(short)
    short_without_attn = model_flops_per_step(short, include_attention_quadratic=False)
    long_with_attn = model_flops_per_step(long)
    long_without_attn = model_flops_per_step(long, include_attention_quadratic=False)

    short_attn_frac = (short_with_attn - short_without_attn) / short_with_attn
    long_attn_frac = (long_with_attn - long_without_attn) / long_with_attn

    # Quadratic attention cost should be a much larger fraction of total FLOPs at longer seq_len.
    assert long_attn_frac > short_attn_frac


def test_memory_breakdown_totals_sum_correctly():
    cfg = ModelConfig(d_model=1024, num_layers=8, num_heads=16, vocab_size=32000, seq_len=1024, batch_size=4)
    mem = model_memory_bytes(cfg)
    assert mem["total"] == mem["params"] + mem["optimizer_state"] + mem["activations"]


def test_memory_without_optimizer_state_is_smaller():
    cfg = ModelConfig(d_model=1024, num_layers=8, num_heads=16, vocab_size=32000, seq_len=1024, batch_size=4)
    with_opt = model_memory_bytes(cfg, include_optimizer_state=True)
    without_opt = model_memory_bytes(cfg, include_optimizer_state=False)
    assert without_opt["total"] < with_opt["total"]
    assert without_opt["optimizer_state"] == 0.0
