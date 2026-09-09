
import jax
import jax.numpy as jnp
import time

jax.config.update("jax_num_cpu_devices", 8) 
Auto = jax.sharding.AxisType.Auto
Explicit = jax.sharding.AxisType.Explicit

def average_local(local_xy):
    
    return  jnp.mean(local_xy, keepdims=True)

def average_jit(arr):
    X, Y = mesh_auto.axis_sizes
    return arr.reshape(X, arr.shape[0] // X, Y, arr.shape[1] // Y).mean(axis=(1, 3))

def roll(x, shift):
    out = jnp.roll(x, shift, 0) - x
    return out


S, D = 8, 8 
N = jax.device_count
mesh_auto = jax.make_mesh((2, 4), ("X", "Y"), (Auto, Auto))
mesh_explicit = jax.make_mesh((2, 4), ("X", "Y"), (Explicit, Explicit))

key = jax.random.PRNGKey(0)
input = jax.random.normal(key, (S, D), jnp.float32)

XY_sharded_explicit = jax.device_put(input, jax.NamedSharding(mesh_explicit, jax.P("X", "Y")))
XY = jax.device_put(input, jax.NamedSharding(mesh_auto, jax.P("X", "Y")))

if __name__ == "__main__":

    with jax.set_mesh(mesh_explicit):
        average_shard_map_jit = jax.jit(
            jax.shard_map(average_local, in_specs=jax.P("X", "Y"), out_specs= jax.P("X", "Y"), mesh=mesh_explicit))
        
        want = average_shard_map_jit(XY_sharded_explicit)
        print(want)
        
        want = roll_shard_map_jit = jax.jit(
                    jax.shard_map(roll, in_specs=jax.P("X", "Y"), out_specs= jax.P("X", "Y"), mesh=mesh_explicit))
        
        jnp.allclose(want, 0, atol=1e-2)
        
    with jax.set_mesh(mesh_auto):
            average_jit = jax.jit(
                        average_jit, out_shardings = jax.NamedSharding(mesh_auto, jax.P("X", "Y")))
            
            want = average_jit(XY)
            print(want)
       
    def benchmark(name, fn, *args, iterations = 20):
        fn(*args).block_until_ready()
        start = time.perf_counter()
        for _ in range(iterations):
            fn(*args).block_until_ready()
        elapsed_ms = (time.perf_counter() - start) * 1_000 / iterations
        print(f"{name:40} {elapsed_ms:8.3f} ms")
        
    print("\nAverage runtime after warm-up:")
    benchmark("average_shard_map_jit", average_shard_map_jit,
                      XY_sharded_explicit)
    benchmark("average_jit", average_jit,
                          XY)
    

"""
it is important to have keepdims as true so each device returns a 1*1 array like mean: [[-0.27315563]] and shard_map at the end recreate the whole array
"""
