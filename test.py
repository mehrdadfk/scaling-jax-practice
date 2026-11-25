import time

import jax
import jax.numpy as jnp
from jax import lax


AXIS_NAME = "dp"


def allreduce_matmul_tiled(
    a_local: jax.Array,
    w_local: jax.Array,
    *,
    output_tile_size: int,
) -> jax.Array:
    """A[BX, DY] * W[DY, F] -> Out[BX, FY].

    a_local has shape [BX, DY / P].
    w_local has shape [DY / P, FY / P].

    The result is sharded over FY: every device returns [BX, FY / P].
    """
    bx, dy_local = a_local.shape
    _, fy_local = w_local.shape

    if fy_local % output_tile_size != 0:
        raise ValueError("FY / number_of_devices must be divisible by tile size")

    num_tiles = fy_local // output_tile_size
    w_tiles = w_local.reshape(dy_local, num_tiles, output_tile_size)
    w_tiles = jnp.transpose(w_tiles, (1, 0, 2))

    def body(_, w_tile):
        partial = a_local @ w_tile
        reduced = lax.psum(partial, AXIS_NAME)
        return None, reduced

    _, output_tiles = lax.scan(body, None, w_tiles)

    # [num_tiles, BX, tile] -> [BX, FY / P]
    return jnp.transpose(output_tiles, (1, 0, 2)).reshape(bx, fy_local)


def reducescatter_matmul_tiled(
    tmp_local: jax.Array,
    w_local: jax.Array,
    *,
    input_tile_size: int,
) -> jax.Array:
    """Tmp[BX, FY] * W[FY, D] -> Out[BX, DY].

    tmp_local has shape [BX, FY / P].
    w_local has shape [FY / P, D].

    psum_scatter communicates only the output shard, producing
    [BX, D / P] on each device.
    """
    bx, fy_local = tmp_local.shape
    _, d = w_local.shape

    if fy_local % input_tile_size != 0:
        raise ValueError("FY / number_of_devices must be divisible by tile size")
    if d % jax.device_count() != 0:
        raise ValueError("D must be divisible by the number of devices")

    num_tiles = fy_local // input_tile_size
    tmp_tiles = tmp_local.reshape(bx, num_tiles, input_tile_size)
    tmp_tiles = jnp.transpose(tmp_tiles, (1, 0, 2))

    w_tiles = w_local.reshape(num_tiles, input_tile_size, d)

    def body(_, values):
        tmp_tile, w_tile = values

        # Each device computes all output columns for this FY tile.
        partial = tmp_tile @ w_tile

        # Reduce over devices and retain only each device's D shard.
        output_shard = lax.psum_scatter(
            partial,
            AXIS_NAME,
            scatter_dimension=1,
            tiled=True,
        )
        return None, output_shard

    _, output_tiles = lax.scan(body, None, (tmp_tiles, w_tiles))

    # The tiles are contributions to the same output, so accumulate them.
    return jnp.sum(output_tiles, axis=0)


def transformer_block_overlap(
    x_local: jax.Array,
    win_local: jax.Array,
    wout_local: jax.Array,
    *,
    tile_size: int,
) -> jax.Array:
    """In[BX, DY] * Win[DY, FY] * Wout[FY, D] -> Out[BX, DY]."""
    hidden_local = allreduce_matmul_tiled(
        x_local,
        win_local,
        output_tile_size=tile_size,
    )
    hidden_local = jax.nn.gelu(hidden_local)

    return reducescatter_matmul_tiled(
        hidden_local,
        wout_local,
        input_tile_size=tile_size,
    )


def transformer_block_baseline(
    x_local: jax.Array,
    win_local: jax.Array,
    wout_local: jax.Array,
) -> jax.Array:
    """Untiled baseline using one collective per matmul."""
    hidden_partial = x_local @ win_local
    hidden_local = lax.psum(hidden_partial, AXIS_NAME)
    hidden_local = jax.nn.gelu(hidden_local)

    output_partial = hidden_local @ wout_local
    return lax.psum_scatter(
        output_partial,
        AXIS_NAME,
        scatter_dimension=1,
        tiled=True,
    )


def benchmark(
    *,
    bx: int = 1024,
    d: int = 4096,
    f: int = 11008,
    tile_size: int = 256,
    warmup: int = 3,
    iterations: int = 10,
):
    num_devices = jax.device_count()

    if d % num_devices != 0 or f % num_devices != 0:
        raise ValueError("D and F must be divisible by the number of devices")

    d_local = d // num_devices
    f_local = f // num_devices

    key = jax.random.key(0)
    keys = jax.random.split(key, 4)

    # Per-device input shard: [P, BX, D / P].
    x = jax.random.normal(keys[0], (num_devices, bx, d_local))

    # Win is sharded over D and F:
    # [P, D / P, F / P].
    win = jax.random.normal(keys[1], (num_devices, d_local, f_local))

    # Wout is sharded over F and replicated over D:
    # [P, F / P, D].
    wout_one = jax.random.normal(keys[2], (f_local, d))
    wout = jnp.broadcast_to(wout_one, (num_devices, f_local, d))

    overlap = jax.pmap(
        lambda x, win, wout: transformer_block_overlap(
            x,
            win,
            wout,
            tile_size=tile_size,
        ),
        axis_name=AXIS_NAME,
    )

    baseline = jax.pmap(
        transformer_block_baseline,
        axis_name=AXIS_NAME,
    )

    # Compile and warm up.
    overlap(x, win, wout).block_until_ready()
    baseline(x, win, wout).block_until_ready()

    for _ in range(warmup):
        overlap(x, win, wout).block_until_ready()
        baseline(x, win, wout).block_until_ready()

    print("\n=== Overlap implementation ===")
    inspect_collectives(overlap, x, win, wout)

    print("\n=== Baseline implementation ===")
    inspect_collectives(baseline, x, win, wout)

    start = time.perf_counter()
    for _ in range(iterations):
        overlap(x, win, wout).block_until_ready()
    overlap_time = (time.perf_counter() - start) / iterations

    start = time.perf_counter()
    for _ in range(iterations):
        baseline(x, win, wout).block_until_ready()
    baseline_time = (time.perf_counter() - start) / iterations

    print(f"devices:       {num_devices}")
    print(f"overlap:       {overlap_time * 1e3:.3f} ms")
    print(f"baseline:      {baseline_time * 1e3:.3f} ms")
    print(f"speedup:       {baseline_time / overlap_time:.3f}x")


    print(f"\nlogical tiles per matmul: {f_local // tile_size}")
    print("logical collectives:")
    print(f"  overlap AllReduce:       {f_local // tile_size}")
    print(f"  overlap ReduceScatter:   {f_local // tile_size}")
    print("  baseline AllReduce:      1")
    print("  baseline ReduceScatter:  1")
    # The two implementations should produce numerically equivalent results.
    y_overlap = overlap(x, win, wout)
    y_baseline = baseline(x, win, wout)
    print(
        "max error:     ",
        float(jnp.max(jnp.abs(y_overlap - y_baseline))),
    )

def inspect_collectives(compiled_fn, *args):
    """Print collective operations in the generated StableHLO IR."""
    ir = compiled_fn.lower(*args).compiler_ir(dialect="stablehlo")
    ir_text = str(ir)

    collective_names = (
        "stablehlo.all_reduce",
        "stablehlo.reduce_scatter",
        "stablehlo.all_gather",
        "stablehlo.collective_permute",
    )

    print("StableHLO collective operations:")
    total = 0

    for name in collective_names:
        count = ir_text.count(name)
        if count:
            print(f"  {name}: {count}")
            total += count

    print(f"  total textual collective ops: {total}")

    # Print the surrounding IR for manual inspection.
    print("\nCollective-related IR lines:")
    for line in ir_text.splitlines():
        if any(name in line for name in collective_names):
            print(line.strip())

if __name__ == "__main__":
    benchmark()
    