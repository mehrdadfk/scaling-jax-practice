import functools

import jax
import jax.numpy as jnp

jax.config.update("jax_num_cpu_devices", 8)
Auto = jax.sharding.AxisType.Auto
Explicit = jax.sharding.AxisType.Explicit

#--------------------------------------------
# Part 1  A[BX, DY] *D W[DY, F] -> Out[BX, F]
#--------------------------------------------
def baseline_allreduce_matmul(A, W):
    out = A @ W
    return jax.lax.psum(out, axis_name="Y")

def collective_all_reduce_matmul(A, W, tile_size=1024):
    _, F = W.shape
    numchunks = (F + tile_size - 1) // tile_size
    
    def body(i, carry):
        out, previous = carry
        partial = jax.lax.psum(previous, axis_name="Y")
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

def collective_scatter_reduce_matmul(tmp, w):
    N = jax.lax.axis_size("Y")
    D_total = w.shape[1]
    chunks = D_total // N
    index = jax.lax.axis_index("Y")
    
    def body(i, acc):
        acc = jax.lax.ppermute(acc, axis_name="Y", perm=[(j, (j + 1) % N) for j in range(N)])
        idx = (index - i - 1) % N
        w_chunk = jax.lax.dynamic_slice_in_dim(w, idx * chunks, chunks, axis=1)
        partial = tmp @ w_chunk
        return acc + partial
    
    idx = (index - 1) % N
    first_chunk = jax.lax.dynamic_slice_in_dim(w, idx * chunks, chunks, axis=1)
    first_mat_mul = tmp @ first_chunk
    
    out = jax.lax.fori_loop(1, N, body, first_mat_mul)
    return out    

B, D, F = 64, 64, 256
key = jax.random.PRNGKey(0)
W = jax.random.normal(key, (D, F), jnp.float32)
A = jax.random.normal(key, (B, D), jnp.float32)

explicit_mesh = jax.make_mesh((2,4), ("X","Y"), (Explicit, Explicit))

W_explicit_sharded = jax.device_put(W, jax.NamedSharding(explicit_mesh, jax.P("Y", None)))
A_explicit_sharded = jax.device_put(A, jax.NamedSharding(explicit_mesh, jax.P("X", "Y")))

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
            jax.shard_map(functools.partial(collective_scatter_reduce_matmul)
                                            , in_specs = explicit_specs, out_specs = jax.P("X", "Y"))
        )

        
        want = baseline_allreduce_matmul_jit(A_explicit_sharded, W_explicit_sharded)
        for name, fn in [
            ("collective_all_reduce_matmul", collective_all_reduce_matmul_jit),
            ("collective_scatter_reduce_matmul", collective_scatter_reduce_matmul_jit)
        ]:
            out = fn(A_explicit_sharded, W_explicit_sharded)
            #print (jax.device_get(want))
            #print (jax.device_get(out) )
            assert jnp.allclose(jax.device_get(want), jax.device_get(out), atol=1e-2), f"{name} result doesn't match reference"

        print(" matches reference")
        
        
"""
footnote :

1. ppermute is not a per device list. Every device runs the same ppermute instruction with the same complete permutation list:
[(0, 1), (1, 2), (2, 3), (3, 0)]
    
2. in part 2 each device initializes a contribution for the chunk one device behind it in the ring.

3. pcast is only needed when the initial loop carry is a replicated constant but will become a Y-varying value during the loop. so if there is no teplicated buffer it is not needed

"""
