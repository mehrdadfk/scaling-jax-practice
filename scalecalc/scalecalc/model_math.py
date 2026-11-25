"""Transformer parameter count, memory footprint, and FLOPs/step.

Formulas follow the standard dense-transformer approximations used in the
scaling book (and the Chinchilla/Kaplan scaling-law papers): 6N FLOPs per
token for the forward+backward pass, ignoring attention's O(seq^2) term
except where it's explicitly broken out for long-context regimes.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    d_model: int
    num_layers: int
    num_heads: int
    vocab_size: int
    seq_len: int
    batch_size: int
    d_ff: int | None = None  # defaults to 4 * d_model if unset
    bytes_per_param: int = 2  # bf16 params
    optimizer_bytes_per_param: int = 8  # Adam-style fp32 moments (2 * 4 bytes)

    @property
    def ffn_hidden(self) -> int:
        return self.d_ff if self.d_ff is not None else 4 * self.d_model

    @property
    def tokens_per_step(self) -> int:
        return self.batch_size * self.seq_len


def param_count(cfg: ModelConfig) -> int:
    """Approximate dense-transformer parameter count (excludes embeddings tying edge cases)."""
    d, ff, l, v = cfg.d_model, cfg.ffn_hidden, cfg.num_layers, cfg.vocab_size

    attn_params_per_layer = 4 * d * d  # Q, K, V, O projections
    ffn_params_per_layer = 2 * d * ff  # up and down projections
    per_layer = attn_params_per_layer + ffn_params_per_layer

    embedding_params = v * d
    unembedding_params = v * d  # separate head; set vocab_size=0 to model tied weights externally

    return l * per_layer + embedding_params + unembedding_params


def model_flops_per_step(cfg: ModelConfig, include_attention_quadratic: bool = True) -> float:
    """Total FLOPs for one forward+backward pass over the whole batch.

    Uses the 6*N*tokens rule of thumb (2 forward + 4 backward FLOPs per
    parameter per token), plus the O(seq^2) attention score/weighted-sum
    term when include_attention_quadratic is set.
    """
    n = param_count(cfg)
    tokens = cfg.tokens_per_step
    dense_flops = 6 * n * tokens

    if not include_attention_quadratic:
        return dense_flops

    # QK^T and attn @ V, each ~2 * seq^2 * d_model FLOPs per layer, per sequence in the batch.
    attn_quadratic_flops = (
        2 * (2 * cfg.seq_len**2 * cfg.d_model) * cfg.num_layers * cfg.batch_size
    )
    return dense_flops + attn_quadratic_flops


def model_memory_bytes(cfg: ModelConfig, include_optimizer_state: bool = True) -> dict[str, float]:
    """Breakdown of per-device memory if the whole model fit on one chip (no sharding)."""
    n = param_count(cfg)
    params = n * cfg.bytes_per_param
    optimizer = n * cfg.optimizer_bytes_per_param if include_optimizer_state else 0.0
    # Rough activation memory: one set of layer activations per token, checkpointed per layer.
    activations = cfg.tokens_per_step * cfg.d_model * cfg.num_layers * 2  # bf16 activations

    return {
        "params": float(params),
        "optimizer_state": float(optimizer),
        "activations": float(activations),
        "total": float(params + optimizer + activations),
    }
