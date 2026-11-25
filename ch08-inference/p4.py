import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import treescope
from safetensors import safe_open

checkpoint_dir = Path("/Users/m89251/projects/scaling-book/checkpoints/Llama-3.1-8B-Instruct")


@dataclass(frozen=True)
class LlamaConfig:
    hidden_size: int  # D
    intermediate_size: int  # F
    num_attention_heads: int  # N
    num_key_value_heads: int  # K
    num_hidden_layers: int  # L
    vocab_size: int  # V
    rms_norm_eps: float
    rope_theta: float
    rope_scaling: dict[str, Any] | None = None

    @property
    def head_dim(self) -> int:  # H
        return self.hidden_size // self.num_attention_heads


def load_config(checkpoint_dir: Path) -> LlamaConfig:
    with (checkpoint_dir / "config.json").open() as config_file:
        raw = json.load(config_file)
    return LlamaConfig(
        hidden_size=raw["hidden_size"],
        intermediate_size=raw["intermediate_size"],
        num_attention_heads=raw["num_attention_heads"],
        num_key_value_heads=raw["num_key_value_heads"],
        num_hidden_layers=raw["num_hidden_layers"],
        vocab_size=raw["vocab_size"],
        rms_norm_eps=raw["rms_norm_eps"],
        rope_theta=raw["rope_theta"],
        rope_scaling=raw.get("rope_scaling"),
    )


def load_weights(checkpoint_dir: Path) -> dict[str, jax.Array]:
    weights: dict[str, jax.Array] = {}
    shard_paths = sorted(checkpoint_dir.glob("*.safetensors"))
    if not shard_paths:
        raise FileNotFoundError(f"No .safetensors shards in {checkpoint_dir}")
    for shard_path in shard_paths:
        with safe_open(shard_path, framework="np") as shard:
            for name in shard.keys():
                weights[name] = jnp.asarray(shard.get_tensor(name))
    return weights


def nest_weights(weights: dict[str, jax.Array]) -> dict[str, Any]:
    """Turn HF's dotted flat names into a nested dict so treescope.show() renders a real tree."""
    nested: dict[str, Any] = {}
    for name, tensor in weights.items():
        node = nested
        *path, leaf = name.split(".")
        for key in path:
            node = node.setdefault(key, {})
        node[leaf] = tensor
    return nested


def actual_param_counts(weights: dict[str, jax.Array]) -> dict[str, int]:
    totals = {"attention": 0, "mlp": 0, "embedding/output": 0, "norm": 0, "other": 0}
    for name, tensor in weights.items():
        if ".self_attn." in name:
            group = "attention"
        elif ".mlp." in name:
            group = "mlp"
        elif "embed_tokens" in name or name == "lm_head.weight":
            group = "embedding/output"
        elif "norm" in name:
            group = "norm"
        else:
            group = "other"
        totals[group] += tensor.size
    return totals


def expected_param_counts(config: LlamaConfig) -> dict[str, int]:
    """Hand-derived counts from the book's GQA and SwiGLU formulas (transformers.md)."""
    D, F, L, V = config.hidden_size, config.intermediate_size, config.num_hidden_layers, config.vocab_size
    N, K, H = config.num_attention_heads, config.num_key_value_heads, config.head_dim
    return {
        "attention": L * 2 * D * (N + K) * H,  # W_Q, W_O: DNH each; W_K, W_V: DKH each
        "mlp": L * 3 * D * F,  # SwiGLU gate/up/down
        "norm": L * 2 * D + D,  # 2 RMSNorms/layer + final norm
        "embedding/output": 2 * D * V,  # embed_tokens + untied lm_head
    }


def print_param_report(config: LlamaConfig, weights: dict[str, jax.Array]) -> None:
    actual = actual_param_counts(weights)
    expected = expected_param_counts(config)
    print(f"{'group':18} {'actual':>14} {'expected':>14}")
    for group in ("attention", "mlp", "norm", "embedding/output"):
        print(f"{group:18} {actual[group]:>14,} {expected[group]:>14,}")
    if actual["other"]:
        print(f"{'other':18} {actual['other']:>14,}")
    total_actual = sum(actual.values())
    print(f"{'total':18} {total_actual:>14,} {sum(expected.values()):>14,}  ({total_actual / 1e9:.2f}B)")


def rms_norm(x: jax.Array, weight: jax.Array, eps: float) -> jax.Array:
    variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    return x * jax.lax.rsqrt(variance + eps) * weight


def linear(x: jax.Array, weight: jax.Array) -> jax.Array:
    """HF Linear weights are stored as [out_features, in_features]."""
    return x @ weight.T


def rotate_half(x: jax.Array) -> jax.Array:
    first_half, second_half = jnp.split(x, 2, axis=-1)
    return jnp.concatenate((-second_half, first_half), axis=-1)


def rope_frequencies(config: LlamaConfig, positions: jax.Array) -> jax.Array:
    """RoPE angles at base theta=500,000 (Llama 3 paper sec. 3.2), plus Llama 3.1's
    NTK-by-parts long-context scaling (config.json's rope_scaling, not in the original
    Llama 3 paper's Table 3 since it's part of the later 128K-context extension)."""
    freqs = 1.0 / (config.rope_theta ** (jnp.arange(0, config.head_dim, 2) / config.head_dim))
    scaling = config.rope_scaling or {}
    if scaling.get("rope_type") == "llama3":
        low_freq_wavelen = scaling["original_max_position_embeddings"] / scaling["low_freq_factor"]
        high_freq_wavelen = scaling["original_max_position_embeddings"] / scaling["high_freq_factor"]
        wavelen = 2 * jnp.pi / freqs
        scaled = freqs / scaling["factor"]
        smooth = (scaling["original_max_position_embeddings"] / wavelen - scaling["low_freq_factor"]) / (
            scaling["high_freq_factor"] - scaling["low_freq_factor"]
        )
        freqs = jnp.where(wavelen > low_freq_wavelen, scaled, freqs)
        is_medium = (wavelen <= low_freq_wavelen) & (wavelen >= high_freq_wavelen)
        freqs = jnp.where(is_medium, (1 - smooth) * scaled + smooth * freqs, freqs)
    angles = positions[:, None] * freqs[None, :]
    return jnp.concatenate((angles, angles), axis=-1)


def apply_rope(x: jax.Array, angles: jax.Array) -> jax.Array:
    """x has its sequence axis at position 1 (both q's [B,T,K,G,H] and k's [B,S,K,H] do)."""
    broadcast_shape = [1] * x.ndim
    broadcast_shape[1] = angles.shape[0]
    broadcast_shape[-1] = angles.shape[1]
    angles = angles.reshape(broadcast_shape)
    return x * jnp.cos(angles) + rotate_half(x) * jnp.sin(angles)


def attention(x: jax.Array, layer: dict[str, jax.Array], config: LlamaConfig, positions: jax.Array) -> jax.Array:
    """GQA per transformers.md's Q[B,T,K,G,H]*K[B,S,K,H] batched formulation (Note 2), no
    repeat of K/V onto N heads."""
    B, T, D = x.shape
    N, K, H = config.num_attention_heads, config.num_key_value_heads, config.head_dim
    G = N // K
    q = linear(x, layer["q_proj.weight"]).reshape(B, T, K, G, H)
    k = linear(x, layer["k_proj.weight"]).reshape(B, T, K, H)
    v = linear(x, layer["v_proj.weight"]).reshape(B, T, K, H)
    angles = rope_frequencies(config, positions)
    q, k = apply_rope(q, angles), apply_rope(k, angles)
    scores = jnp.einsum("btkgh,bskh->btskg", q, k) / (H**0.5)
    causal_mask = jnp.triu(jnp.ones((T, T), dtype=bool), k=1)
    scores = jnp.where(causal_mask[None, :, :, None, None], -jnp.inf, scores)
    probs = jax.nn.softmax(scores.astype(jnp.float32), axis=2).astype(x.dtype)
    out = jnp.einsum("btskg,bskh->btkgh", probs, v).reshape(B, T, D)
    return linear(out, layer["o_proj.weight"])


def mlp(x: jax.Array, layer: dict[str, jax.Array]) -> jax.Array:
    """Gating einsum (transformers.md Note [gating einsum]): silu(W_in1 x) * (W_in2 x), then W_out."""
    gated = jax.nn.silu(linear(x, layer["gate_proj.weight"]))
    return linear(gated * linear(x, layer["up_proj.weight"]), layer["down_proj.weight"])


def layer_weights(weights: dict[str, jax.Array], layer_index: int, section: str) -> dict[str, jax.Array]:
    prefix = f"model.layers.{layer_index}.{section}."
    return {name.removeprefix(prefix): tensor for name, tensor in weights.items() if name.startswith(prefix)}


def forward(token_ids: jax.Array, weights: dict[str, jax.Array], config: LlamaConfig) -> jax.Array:
    """Pre-norm architecture (transformers.md Note 3): x + attn(norm(x)), x + mlp(norm(x))."""
    hidden = weights["model.embed_tokens.weight"][token_ids]
    positions = jnp.arange(token_ids.shape[1])
    for index in range(config.num_hidden_layers):
        prefix = f"model.layers.{index}."
        normed = rms_norm(hidden, weights[prefix + "input_layernorm.weight"], config.rms_norm_eps)
        hidden = hidden + attention(normed, layer_weights(weights, index, "self_attn"), config, positions)
        normed = rms_norm(hidden, weights[prefix + "post_attention_layernorm.weight"], config.rms_norm_eps)
        hidden = hidden + mlp(normed, layer_weights(weights, index, "mlp"))
    hidden = rms_norm(hidden, weights["model.norm.weight"], config.rms_norm_eps)
    return linear(hidden, weights["lm_head.weight"])


def greedy_generate(
    token_ids: jax.Array, weights: dict[str, jax.Array], config: LlamaConfig, max_new_tokens: int
) -> jax.Array:
    
    for _ in range(max_new_tokens):
        next_token = jnp.argmax(forward(token_ids, weights, config)[:, -1, :], axis=-1)[:, None]
        token_ids = jnp.concatenate((token_ids, next_token), axis=1)
    return token_ids


if __name__ == "__main__":
    config = load_config(checkpoint_dir)
    weights = load_weights(checkpoint_dir)
    print_param_report(config, weights)
    treescope.show(nest_weights(weights))
