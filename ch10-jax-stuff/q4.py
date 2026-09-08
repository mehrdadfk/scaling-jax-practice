import functools
import time

import jax
import jax.numpy as jnp

jax.config.update("jax_num_cpu_devices", 16)
Auto = jax.sharding.AxisType.Auto
Explicit = jax.sharding.AxisType.Explicit

#--------------------------------------------
# Part 1  A[BX, DY] *D W[DY, F] -> Out[BX, F]
#--------------------------------------------
def bidirectional_ring_psum(local_value, axis_name="Y"):
    if local_value.shape[-1] % 2:
        raise ValueError("bidirectional psum requires an even final dimension")

    N = jax.lax.axis_size(axis_name)
    half_width = local_value.shape[-1] // 2
    clockwise = [(i, (i + 1) % N) for i in range(N)]
    counterclockwise = [(i, (i - 1) % N) for i in range(N)]

    right = local_value[..., :half_width]
    left = local_value[..., half_width:]
    right_total = right
    left_total = left

    for _ in range(N - 1):
        right = jax.lax.ppermute(right, axis_name, clockwise)
        left = jax.lax.ppermute(left, axis_name, counterclockwise)
        right_total = right_total + right
        left_total = left_total + left

    return jnp.concatenate((right_total, left_total), axis=-1)
    
def baseline_allreduce_matmul(A, W):
    out = A @ W
    return jax.lax.psum(out, axis_name="Y")


def collective_all_reduce_matmul(A, W, tile_size=1024):
    _, F = W.shape
    numchunks = (F + tile_size - 1) // tile_size
    def body(i, carry):
        out, previous = carry
        partial = bidirectional_ring_psum(previous, axis_name="Y")
        W_chunk = jax.lax.dynamic_slice_in_dim(W, i * tile_size, tile_size, axis=1)
        next_mat_mul = A @ W_chunk
        out = jax.lax.dynamic_update_slice_in_dim(out, partial, (i-1) * tile_size, axis=1)
        return out, next_mat_mul
    
    first_chunk = jax.lax.dynamic_slice_in_dim(W, 0, tile_size, axis=1)
    next_mat_mul = A @ first_chunk
    out0 = jnp.zeros((A.shape[0], F), dtype=A.dtype)
    
    out, last_mat_mul = jax.lax.fori_loop(1, numchunks, body, (out0, next_mat_mul))
    
    last_partial =jax.lax.psum(last_mat_mul, axis_name="Y")
    out = jax.lax.dynamic_update_slice_in_dim(out, last_partial, (numchunks-1) * tile_size, axis=1)
    
    return out

#--------------------------------------------
# Part 2  # Tmp[BX, FY] *F W2[FY, D] -> Out[BX, DY]
#--------------------------------------------

def collective_reduce_scatter_matmul(tmp, w):
    N = jax.lax.axis_size("Y")
    D_total = w.shape[1]
    chunks = D_total // N
    index = jax.lax.axis_index("Y")
    half_chunk = chunks // 2
    right_perm = [(i, (i + 1) % N) for i in range(N)]
    left_perm = [(i, (i - 1) % N) for i in range(N)]
        
    def body(i, acc):
        right, left = acc
        right = jax.lax.ppermute(right, axis_name="Y", perm=right_perm)
        left = jax.lax.ppermute(left, axis_name="Y", perm=left_perm)
        right_index = (index - i - 1) % N
        left_index = (index + i + 1) % N

        w_chunk_right = jax.lax.dynamic_slice_in_dim(w, right_index * chunks, half_chunk, axis=1)
        w_chunk_left = jax.lax.dynamic_slice_in_dim(w, left_index * chunks + half_chunk, half_chunk, axis=1)
        
        partial_right = right + tmp @ w_chunk_right
        partial_left = left + tmp @ w_chunk_left
        return partial_right, partial_left
    
    right_index = (index - 1) % N
    left_index = (index + 1) % N
    right_w_chunk = jax.lax.dynamic_slice_in_dim(
        w, right_index * chunks, half_chunk, axis=1
    )
    left_w_chunk = jax.lax.dynamic_slice_in_dim(
        w, left_index * chunks + half_chunk, half_chunk, axis=1
    )
    right = tmp @ right_w_chunk
    left = tmp @ left_w_chunk

    right, left = jax.lax.fori_loop(1, N, body, (right, left))
    return jnp.concatenate((right, left), axis=1)    

def baseline_reduce_scatter_matmul(tmp, w):
    return jax.lax.psum_scatter(tmp @ w, axis_name="Y", scatter_dimension=1, tiled=True)

#--------------------------------------------
# Part 3  # In[BX, DY] *D Win[D, FY] *F Wout[FY, D] -> Out[BX, DY] 
#--------------------------------------------
 
def collective_mlp(input, w_in, w_out):
    temp = collective_all_gather_matmul(input, w_in)
    out = collective_reduce_scatter_matmul(temp, w_out)
    return out
    
# In[BX, DY] *D Win[D, FY] -> Tmp[BX, FY]
def collective_all_gather_matmul(input, w_in):
    N = jax.lax.axis_size("Y")
    D_total = w_in.shape[0]
    chunks = D_total // N
    if chunks % 2:
        raise ValueError("bidirectional all-gather requires an even input chunk size")
    index = jax.lax.axis_index("Y")
    half_chunk = chunks // 2
    right_perm = [(i, (i + 1) % N) for i in range(N)]
    left_perm = [(i, (i - 1) % N) for i in range(N)]

    def partial_for(in_chunk, chunk_index, offset):
        w_chunk = jax.lax.dynamic_slice_in_dim(
            w_in, chunk_index * chunks + offset, half_chunk, axis=0
        )
        return in_chunk @ w_chunk

    right = input[:, :half_chunk]
    left = input[:, half_chunk:]
    right_total = partial_for(right, index, 0)
    left_total = partial_for(left, index, half_chunk)

    def body(i, carry):
        right, left, right_total, left_total = carry
        right = jax.lax.ppermute(right, axis_name="Y", perm=right_perm)
        left = jax.lax.ppermute(left, axis_name="Y", perm=left_perm)
        right_index = (index - i) % N
        left_index = (index + i) % N
        return (
            right,
            left,
            right_total + partial_for(right, right_index, 0),
            left_total + partial_for(left, left_index, half_chunk),
        )

    _, _, right_total, left_total = jax.lax.fori_loop(
        1, N, body, (right, left, right_total, left_total)
    )
    return right_total + left_total

def baseline_mlp(input, w_in, w_out):
    in_full = jax.lax.all_gather(input, axis_name="Y", axis=1, tiled=True)
    tmp_local = in_full @ w_in  
    out_partial = tmp_local @ w_out 
    return jax.lax.psum_scatter(out_partial, axis_name="Y", scatter_dimension=1, tiled=True)

# In[BX, DY] *D Win[D, FY] -> Tmp[BX, FY]
def baseline_all_gather_matmul(input, w_in):
    in_full = jax.lax.all_gather(input, axis_name="Y", axis=1, tiled=True)  # [B_local, D]
    out = in_full @ w_in  # [B_local, F_local]
    return out 

B, D, F = 64, 64, 256
key = jax.random.PRNGKey(0)
W = jax.random.normal(key, (D, F), jnp.float32)
W_out = jax.random.normal(key, (F, D), jnp.float32)
A = jax.random.normal(key, (B, D), jnp.float32)
TMP = jax.random.normal(key, (B, D), jnp.float32)
explicit_mesh = jax.make_mesh((4,4), ("X","Y"), (Explicit, Explicit))

W_explicit_sharded = jax.device_put(W, jax.NamedSharding(explicit_mesh, jax.P("Y", None)))
A_explicit_sharded = jax.device_put(A, jax.NamedSharding(explicit_mesh, jax.P("X", "Y")))
TMP_explicit_sharded = jax.device_put(TMP, jax.NamedSharding(explicit_mesh, jax.P("X", "Y")))
W_in_explicit_sharded = jax.device_put(W, jax.NamedSharding(explicit_mesh, jax.P(None, "Y")))
W_out_explicit_sharded = jax.device_put(W_out, jax.NamedSharding(explicit_mesh, jax.P("Y", None)))


for name, fn in [
   # ("moe_sorted_jit", moe_sorted_jit),
    # ("moe_naive_reference_jit", moe_naive_reference_jit),
    # ("moe_scan_jit", moe_scan_jit),
    # ("moe_ragged_jit", moe_ragged_jit),
]:
    print(fn.lower(A_explicit_sharded, W_explicit_sharded).compile().as_text())
# 
if __name__ == "__main__":
   

    # print("Profiling moe_sorted_jit (the version that AllGathers A)...")
    # with jax.profiler.trace("/tmp/jax-trace-q2-f"):
    #     with jax.named_scope("moe_sorted_jit"):
    #         out = moe_sorted_jit(A_sharded, W_sharded, B_sharded)
    #         out.block_until_ready()
    #     with jax.named_scope("moe_ragged_jit"):
    #         out = moe_ragged_jit(A_sharded, W_sharded, B_sharded)
    #         out.block_until_ready()
    # print("Trace written to /tmp/jax-trace-q2-f")
    # print(f"moe_local matches reference: {jnp.allclose(got, out)}")

    with jax.set_mesh(explicit_mesh):
        explicit_specs = (jax.P("X", "Y"), jax.P("Y", None))
        explicit_specs_mlp = (jax.P("X", "Y"), jax.P(None, "Y"), jax.P("Y", None))
        explicit_specs_all_gather = (jax.P("X", "Y"), jax.P(None, "Y"))
        
        baseline_allreduce_matmul_jit = jax.jit(
            jax.shard_map(baseline_allreduce_matmul, in_specs=explicit_specs, out_specs=jax.P("X", None))
        )
        
        collective_all_reduce_matmul_jit = jax.jit(
            jax.shard_map(functools.partial(collective_all_reduce_matmul, tile_size=64)
                                            , in_specs = explicit_specs, out_specs = jax.P("X", None), check_vma=False)
        )
        collective_scatter_reduce_matmul_jit = jax.jit(
            jax.shard_map(functools.partial(collective_reduce_scatter_matmul)
                                            , in_specs = explicit_specs, out_specs = jax.P("X", "Y"))
        )
        baseline_reduce_scatter_matmul_jit = jax.jit(
            jax.shard_map(baseline_reduce_scatter_matmul, in_specs=explicit_specs, out_specs=jax.P("X", "Y"))
        )
       
        baseline_all_gather_matmul_jit = jax.jit(
            jax.shard_map(functools.partial(baseline_all_gather_matmul)
                                                    , in_specs = explicit_specs_all_gather, out_specs = jax.P("X", "Y"))
        )
        collective_all_gather_matmul_jit = jax.jit(
            jax.shard_map(functools.partial(collective_all_gather_matmul)
                                            , in_specs = explicit_specs_all_gather, out_specs = jax.P("X", "Y"))
        )

        baseline_mlp_jit = jax.jit(
            jax.shard_map(functools.partial(baseline_mlp)
                                            , in_specs = explicit_specs_mlp, out_specs = jax.P("X", "Y"))
        )
        collective_mlp_jit = jax.jit(
            jax.shard_map(functools.partial(collective_mlp)
                                            , in_specs = explicit_specs_mlp, out_specs = jax.P("X", "Y"))
        )
        
        want = baseline_allreduce_matmul_jit(A_explicit_sharded, W_explicit_sharded)
        for name, fn in [
            ("collective_all_reduce_matmul", collective_all_reduce_matmul_jit)
        ]:
            out = fn(A_explicit_sharded, W_explicit_sharded)
            assert jnp.allclose(jax.device_get(want), jax.device_get(out), atol=1e-2), f"{name} result doesn't match reference"

        want_reduce_scatter = baseline_reduce_scatter_matmul_jit(TMP_explicit_sharded, W_explicit_sharded)
        for name, fn in [
            ("collective_scatter_reduce_matmul", collective_scatter_reduce_matmul_jit)
        ]:
            out = fn(TMP_explicit_sharded, W_explicit_sharded)
            assert jnp.allclose(jax.device_get(want_reduce_scatter), jax.device_get(out), atol=1e-2), f"{name} result doesn't match reference"

        print("All-reduce and reduce-scatter variants match their references.")
        
        for name, fn in [
                    ("baseline_all_gather_matmul", baseline_all_gather_matmul_jit),
                    ("collective_all_gather_matmul", collective_all_gather_matmul_jit)
                ]:
                    out = fn(TMP_explicit_sharded, W_in_explicit_sharded)
                    assert jnp.allclose(jax.device_get(want), jax.device_get(out), atol=1e-2), f"{name} result doesn't match reference"


        for name, fn in [
                    ("baseline_mlp", baseline_mlp_jit),
                    ("collective_mlp", collective_mlp_jit)
                ]:
                    out = fn(A_explicit_sharded, W_in_explicit_sharded, W_out_explicit_sharded)

                    want_mlp = (A @ W) @ W_out
                    assert jnp.allclose(jax.device_get(want_mlp), jax.device_get(out), rtol=1e-4, atol=1.0,), f"{name} result doesn't match reference"
        

        def benchmark(name, fn, *args, iterations=20):
            fn(*args).block_until_ready()
            start = time.perf_counter()
            for _ in range(iterations):
                fn(*args).block_until_ready()
            elapsed_ms = (time.perf_counter() - start) * 1_000 / iterations
            print(f"{name:40} {elapsed_ms:8.3f} ms")

        print("\nAverage runtime after warm-up:")
        benchmark("baseline_allreduce_matmul", baseline_allreduce_matmul_jit,
                  A_explicit_sharded, W_explicit_sharded)
        benchmark("collective_all_reduce_matmul", collective_all_reduce_matmul_jit,
                  A_explicit_sharded, W_explicit_sharded)
        benchmark("baseline_reduce_scatter_matmul", baseline_reduce_scatter_matmul_jit,
                  TMP_explicit_sharded, W_explicit_sharded)
        benchmark("collective_scatter_reduce_matmul", collective_scatter_reduce_matmul_jit,
                  TMP_explicit_sharded, W_explicit_sharded)
        benchmark("baseline_all_gather_matmul", baseline_all_gather_matmul_jit,
                  TMP_explicit_sharded, W_in_explicit_sharded)
        benchmark("collective_all_gather_matmul", collective_all_gather_matmul_jit,
                  TMP_explicit_sharded, W_in_explicit_sharded)
        benchmark("baseline_mlp", baseline_mlp_jit,
                  A_explicit_sharded, W_in_explicit_sharded, W_out_explicit_sharded)
        benchmark("collective_mlp", collective_mlp_jit,
                  A_explicit_sharded, W_in_explicit_sharded, W_out_explicit_sharded)
        
        
"""
footnote :

1. ppermute is not a per device list. Every device runs the same ppermute instruction with the same complete permutation list:
[(0, 1), (1, 2), (2, 3), (3, 0)]
    
2. in part 2 each device initializes a contribution for the chunk one device behind it in the ring.

3. pcast is only needed when the initial loop carry is a replicated constant but will become a Y-varying value during the loop. so if there is no teplicated buffer it is not needed

"""
