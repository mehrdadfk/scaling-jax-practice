import os

import jax
import jax.numpy as jnp

jax.config.update("jax_num_cpu_devices", 4)
Auto = jax.sharding.AxisType.Auto
Explicit = jax.sharding.AxisType.Explicit

TRACE_DIR = os.environ.get("JAX_TRACE_DIR")
TRACE_ITERS = int(os.environ.get("JAX_TRACE_ITERS", "3"))

"""
W: float32[EX, D, F]
A: float32[SX, D]
B: int32[SX]
Out[i] = A[i] @ W[B[i]].
"""
def moe_naive_reference(A, W, B):
    """O(S*D*F) but materializes [S, D, F] -- only for checking correctness"""
    return jnp.einsum('sd,sdf->sf', A, W[B])  # A: [S, D], W[B]: [S, D, F]

def moe_local_sorted(A, W, B):
    E, D, F = W.shape
    S = A.shape[0]
    """
scan:
    E × S × D × F

ragged:
    S × D × F

in worst case all to token to one expert, so each bucket needs the capacity of all S.

B:[2, 0, 1, 2, 0, 1]
sorted_B:[0, 0, 1, 1, 2, 2]
sorted_indices:[1, 4, 2, 5, 0, 3]
position = [0, 1, 0, 1, 0, 1]
position within expert=global sorted position−start position of expert​
global position:        0  1  2  3  4  5
expert group starts:    0  0  2  2  4  4
                        -----------------
position within group:  0  1  0  1  0  1

token 1 -> (expert 0, position 0)
token 4 -> (expert 0, position 1)

token 2 -> (expert 1, position 0)
token 5 -> (expert 1, position 1)

token 0 -> (expert 2, position 0)
token 3 -> (expert 2, position 1)
group the tokens by expert.
buffer.shape = [E, S, D]
buffer[0] =
[
 [3,4],     token 1
 [7,8],     token 4
 [0,0],     padding
 [0,0]
]

buffer[1] =
[
 [1,2],     token 2
 [5,6],     token 3
 [0,0],
 [0,0]
]

Memory ≈ E*S*D + E*S*F
    """
    sorted_B, sorted_indices = jax.lax.sort_key_val(B, jnp.arange(S))
    sorted_A = A[sorted_indices]

    boundaries = jnp.concatenate([jnp.array([True]), jnp.diff(sorted_B) != 0])
    indices = jnp.arange(len(sorted_B))
    # boundaries:       [True, False, True, False, False, True, False]
    # indices:          [0,    1,     2,    3,     4,     5,    6    ]
    # where             [0,   -1,     2,   -1,    -1,     5,   -1    ]
    # jnp.maximum.accumulate carries the largest value seen so far left-to-right. Because -1 is smaller than any real index 
    group_starts = jnp.maximum.accumulate(jnp.where(boundaries, indices, -1))

    expert_positions = indices - group_starts
    sorted_buffer = jnp.zeros((E, S, D),dtype=A.dtype)
    sorted_buffer = sorted_buffer.at[sorted_B, expert_positions].set(sorted_A)

    expert_out = jnp.einsum("esd,edf->esf", sorted_buffer,W)
    sorted_out = expert_out[sorted_B,expert_positions]
    
    out = jnp.zeros((S, F),dtype=expert_out.dtype)
    
    out = out.at[sorted_indices].set(sorted_out)

    return out

def moe_local_scan(A, W, B):
    def body(output, e):
        mask = (B == e) [:, None] # [S, 1]
        expert_out = jnp.einsum("sd, df -> sf", A, W[e])
        return output + expert_out * mask, None

    output = jnp.zeros((A.shape[0], W.shape[2]), dtype=A.dtype)
    output, _ = jax.lax.scan(body, output, jnp.arange(W.shape[0]))
    return output

def moe_local_ragged(A, W, B, device_index=None):

    local_experts = W.shape[0]
    tokens = A.shape[0]

    if device_index is not None:
        owns_expert = B // local_experts == device_index
        A = jnp.where(owns_expert[:, None], A, 0)
        B = jnp.where(owns_expert, B % local_experts, 0)

    sorted_B, sorted_indices = jax.lax.sort_key_val(B, jnp.arange(tokens))
    sorted_A = A[sorted_indices]

    group_sizes = jnp.bincount(sorted_B, length=local_experts)

    sorted_out = jax.lax.ragged_dot(sorted_A, W, group_sizes)  # [S, F]
    
    return jnp.zeros((tokens, W.shape[2]), sorted_out.dtype).at[sorted_indices].set(sorted_out)

def moe_all_gather_ragged(A, W, B):
    device_index = jax.lax.axis_index("X")

    A_full = jax.lax.all_gather(A, axis_name="X", axis=0, tiled=True)
    B_full = jax.lax.all_gather(B, axis_name="X", axis=0, tiled=True)
    out_full = moe_local_ragged(A_full, W, B_full, device_index)

    return jax.lax.psum_scatter(out_full,axis_name="X",scatter_dimension=0,tiled=True,)


def moe_all_to_all_ragged(A, W, B):
    chunk_size = 128
    tokens = A.shape[0]
    num_chunks = (tokens + chunk_size - 1) // chunk_size
    N = jax.lax.axis_size("X")
    C = num_chunks * chunk_size 
    dst = B // W.shape[0] 
    e_local = B % W.shape[0]  
    
    onehot_B = jax.nn.one_hot(dst, N, dtype=jnp.int32)
    slot = jnp.sum(onehot_B * (jnp.cumsum(onehot_B, axis=0) - 1), axis=1).astype(jnp.int32)
    

# token:       0  1  2  3  4  5
# dst:         2  0  2  1  2  0
#
# onehot_B =
# [[0, 0, 1, 0],  # token 0 -> device 2
#  [1, 0, 0, 0],  # token 1 -> device 0
#  [0, 0, 1, 0],  # token 2 -> device 2
#  [0, 1, 0, 0],  # token 3 -> device 1
#  [0, 0, 1, 0],  # token 4 -> device 2
#  [1, 0, 0, 0]]  # token 5 -> device 0

# [[0, 0, 1, 0],
#  [1, 0, 1, 0],
#  [1, 0, 2, 0],
#  [1, 1, 2, 0],
#  [1, 1, 3, 0],
#  [2, 1, 3, 0]]
    send_A = jnp.zeros((N, C, D), A.dtype).at[dst, slot].set(A)
    send_B = jnp.zeros((N, C), jnp.int32).at[dst, slot].set(e_local)
    
    def body(i, output):
        start = i * chunk_size
        chunk_A = jax.lax.dynamic_slice_in_dim(send_A, start, chunk_size, axis=1) # send_A[:, 0:128, :]= [4, 128, 64]
        chunk_B = jax.lax.dynamic_slice_in_dim(send_B, start, chunk_size, axis=1)
        rec_A = jax.lax.all_to_all(chunk_A, axis_name="X", split_axis=0, concat_axis=0, tiled=True)
        rec_B = jax.lax.all_to_all(chunk_B, axis_name="X", split_axis=0, concat_axis=0, tiled=True)

        
        #row 0 -> tokens device 0 wants to send to device 0
        #row 1 -> tokens device 0 wants to send to device 1
        #row 2 -> tokens device 0 wants to send to device 2
        #row 3 -> tokens device 0 wants to send to device 3
        
        #rec_A[0] -> tokens sent from device 0 to device 3
        #rec_A[1] -> tokens sent from device 1 to device 3
        #rec_A[2] -> tokens sent from device 2 to device 3
        #rec_A[3] -> tokens sent from device 3 to device 3
        
        received_out = moe_local_ragged(
            rec_A.reshape(N * chunk_size, D),
            W,
            rec_B.reshape(N * chunk_size),
        ).reshape(N, chunk_size, F)
        
        returned_out = jax.lax.all_to_all(
            received_out, axis_name="X", split_axis=0, concat_axis=0
        )
        output = jax.lax.dynamic_update_slice_in_dim(output, returned_out, start, axis=1)
        return output
    
    out_acc0 = jax.lax.pcast(jnp.zeros((N, C, F), A.dtype), ("X",), to="varying")
    output_by_bucket = jax.lax.fori_loop(0, num_chunks, body, out_acc0)
    return output_by_bucket[dst, slot]

def moe_k_expert(A, W, BK):
    experts_per_token = BK.shape[1]

    def body(expert_index, accumulated_out):
        expert_out = moe_all_to_all_ragged(A, W, BK[:, expert_index])
        return accumulated_out + expert_out

    initial_out = jax.lax.pcast(jnp.zeros((A.shape[0], W.shape[2]), dtype=A.dtype), ("X",), to="varying")
    total_out = jax.lax.fori_loop(0, experts_per_token, body, initial_out)
    return total_out / experts_per_token
   
   
def moe_k_expert2(A, W, BK):
    tokens, experts_per_token = BK.shape
    repeated_A = jnp.repeat(A, experts_per_token, axis=0)
    routed_out = moe_all_to_all_ragged(repeated_A, W, BK.reshape(-1))
    return routed_out.reshape(tokens, experts_per_token, W.shape[2]).mean(axis=1)
       
   
mesh = jax.make_mesh((4,), ("X",), (Auto,))
jax.set_mesh(mesh)

S, D, F, E , K = 4, 64, 128, 8, 3
key = jax.random.PRNGKey(0)
W = jax.random.normal(key, (E, D, F), jnp.float32)
A = jax.random.normal(key, (S, D), jnp.float32)
B = jax.random.randint(key, (S,), 0, E)
BK = jax.random.randint(key, (S, K), 0, E)

W_sharding = jax.NamedSharding(mesh, jax.P("X", None, None))
A_sharding = jax.NamedSharding(mesh, jax.P("X", None))
B_sharding = jax.NamedSharding(mesh, jax.P("X"))
BK_sharding = jax.NamedSharding(mesh, jax.P("X", None))
out_sharding = jax.NamedSharding(mesh, jax.P("X", None))

W_sharded = jax.device_put(W, W_sharding)
A_sharded = jax.device_put(A, A_sharding)
B_sharded = jax.device_put(B, B_sharding)
BK_sharded = jax.device_put(BK, BK_sharding)

explicit_mesh = jax.make_mesh((4,), ("X",), (Explicit,))

W_explicit_sharded = jax.device_put(W, jax.NamedSharding(explicit_mesh, jax.P("X", None, None)))
A_explicit_sharded = jax.device_put(A, jax.NamedSharding(explicit_mesh, jax.P("X", None)))
B_explicit_sharded = jax.device_put(B, jax.NamedSharding(explicit_mesh, jax.P("X")))
BK_explicit_sharded = jax.device_put(BK, jax.NamedSharding(explicit_mesh, jax.P("X", None)))

moe_naive_reference_jit = jax.jit(moe_naive_reference, out_shardings=out_sharding)
moe_sorted_jit = jax.jit(moe_local_sorted, out_shardings=out_sharding)
moe_scan_jit = jax.jit(moe_local_scan, out_shardings=out_sharding)
moe_ragged_jit = jax.jit(moe_local_ragged, out_shardings=out_sharding)

with jax.set_mesh(explicit_mesh):
    explicit_specs = (jax.P("X", None), jax.P("X", None, None), jax.P("X"))
    moe_all_gather_ragged_jit = jax.jit(jax.shard_map(moe_all_gather_ragged, in_specs = explicit_specs, out_specs = jax.P("X", None)))
    moe_all_to_all_ragged_jit = jax.jit(jax.shard_map(moe_all_to_all_ragged, in_specs = explicit_specs, out_specs = jax.P("X", None)))
    moe_k_expert_jit = jax.jit(jax.shard_map(moe_k_expert, in_specs = explicit_specs, out_specs = jax.P("X", None)))
    

for name, fn in [
   # ("moe_sorted_jit", moe_sorted_jit),
    # ("moe_naive_reference_jit", moe_naive_reference_jit),
    # ("moe_scan_jit", moe_scan_jit),
    # ("moe_ragged_jit", moe_ragged_jit),
]:
    print(fn.lower(A_sharded, W_sharded, B_sharded).compile().as_text())
# 
if __name__ == "__main__":
    want = moe_naive_reference(A, W, B)
    got = moe_local_sorted(A, W, B)
    want_k = jnp.einsum("sd,skdf->skf", A, W[BK]).mean(axis=1)

    print(f"moe_local matches reference: {jnp.allclose(got, want)}")
    
    with jax.set_mesh(explicit_mesh):
        for name, fn in [
            #("moe_all_gather (pass 1: all_gather)", moe_all_gather_jit),
            #("moe_shard_map_alltoall (pass 2: all_to_all + while_loop)", moe_all_gather_jit),
            #("moe_all_gather_sorted_buffer", moe_all_gather_sorted_buffer_jit),
            # ("moe_all_gather_ragged", moe_all_gather_ragged_jit),
            ("moe_all_to_all_ragged", moe_all_to_all_ragged_jit),
        ]:
            out = fn(A_explicit_sharded, W_explicit_sharded, B_explicit_sharded)
            print (jax.device_get(want))
            print (jax.device_get(out) )
            assert jnp.allclose(jax.device_get(want), jax.device_get(out), atol=1e-2), f"{name} result doesn't match reference"

        got_k = moe_k_expert_jit(A_explicit_sharded, W_explicit_sharded, BK_explicit_sharded)
        assert jnp.allclose(jax.device_get(want_k), jax.device_get(got_k), atol=1e-2), "moe_k_expert result doesn't match reference"
        print("moe_k_expert matches reference")

    if TRACE_DIR:
        trace_calls = [
            ("moe_sorted", moe_sorted_jit, (A_sharded, W_sharded, B_sharded)),
            ("moe_scan", moe_scan_jit, (A_sharded, W_sharded, B_sharded)),
            ("moe_ragged", moe_ragged_jit, (A_sharded, W_sharded, B_sharded)),
            ("moe_all_gather_ragged", moe_all_gather_ragged_jit,
             (A_explicit_sharded, W_explicit_sharded, B_explicit_sharded)),
            ("moe_all_to_all_ragged", moe_all_to_all_ragged_jit,
             (A_explicit_sharded, W_explicit_sharded, B_explicit_sharded)),
            ("moe_k_expert", moe_k_expert_jit,
             (A_explicit_sharded, W_explicit_sharded, BK_explicit_sharded)),
        ]
        for _, fn, args in trace_calls:
            fn(*args).block_until_ready()

        os.makedirs(TRACE_DIR, exist_ok=True)
        print(f"Capturing profiler trace to {TRACE_DIR}...")
        with jax.profiler.trace(TRACE_DIR):
            for name, fn, args in trace_calls:
                with jax.profiler.TraceAnnotation(name):
                    for _ in range(TRACE_ITERS):
                        output = fn(*args)
                    output.block_until_ready()
        print(f"Trace written to {TRACE_DIR}")

"""
footnotes :
1. in moe_all_to_all_ragged this time instead of sorting to determine the position of token, i will use hot one method
The cumulative sum counts how many tokens have already been assigned to each bucket.

2. pcast(..., to="varying") is needed for scan because shard_map initially treats output as replicated across mesh axis X, but the scan makes it different on each device.

3. because the number of tokens per expert can be uneven, we will add padding to make it same shape. 
"""