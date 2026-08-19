from max.gpu.host.dim import Dim
from max.gpu.host import DeviceContext
from std.gpu import grid_dim, block_dim, block_idx, thread_idx
from std.math import exp
from std.sys import exit, has_accelerator

from lake import VSQR, TSCALE, tpdt

comptime BUFFER_TYPE = Pointer[Float64, MutUntrackedOrigin]



# GPU Kernel
def evolve(
    d_un: BUFFER_TYPE,
    d_uc: BUFFER_TYPE,
    d_uo: BUFFER_TYPE,
    pebbles: BUFFER_TYPE,
    n: Int,
    h: Float64,
    dt: Float64,
    t: Float64,
):
    var row: Int = grid_dim.x * block_dim.x
    var p_row: Int = grid_dim.x * block_dim.x + 4
    var p_offset: Int = 2 * p_row + 2
    var i: Int = (block_idx.y * block_dim.y + thread_idx.y) * row + (
        block_idx.x * block_dim.x + thread_idx.x
    )
    var idx: Int = (
        (block_idx.y * block_dim.y + thread_idx.y) * p_row
        + (block_idx.x * block_dim.x + thread_idx.x)
        + p_offset
    )

    def d_f(p: Float64, t: Float64) -> Float64:
        return -exp(-TSCALE * t) * p

    d_un[unsafe_offset=idx] = (
        2 * d_uc[unsafe_offset=idx]
        - d_uo[unsafe_offset=idx]
        + VSQR
        * (dt * dt)
        * (
            (
                1
                * (
                    d_uc[unsafe_offset=idx - 1]
                    + d_uc[unsafe_offset=idx + 1]
                    + d_uc[unsafe_offset=idx + p_row]
                    + d_uc[unsafe_offset=idx - p_row]
                )
                + 0.25
                * (
                    d_uc[unsafe_offset=idx + p_row - 1]
                    + d_uc[unsafe_offset=idx + p_row + 1]
                    + d_uc[unsafe_offset=idx - p_row - 1]
                    + d_uc[unsafe_offset=idx - p_row + 1]
                )
                + 0.125
                * (
                    d_uc[unsafe_offset=idx - 2]
                    + d_uc[unsafe_offset=idx + 2]
                    + d_uc[unsafe_offset=idx + p_row + p_row]
                    + d_uc[unsafe_offset=idx - p_row - p_row]
                )
                - 5.5 * d_uc[unsafe_offset=idx]
            )
            / (h * h)
            + d_f(pebbles[unsafe_offset=i], t)
        )
    )


def run_gpu(
    mut u: List[Float64],
    u0: List[Float64],
    u1: List[Float64],
    pebbles: List[Float64],
    n: UInt32,
    h: Float64,
    end_time: Float64,
    nthreads: UInt32,
) raises:
    var t: Float64 = 0.0
    var dt: Float64 = h / 2.0
    var nblocks: UInt32 = n / nthreads

    var buffer_size = Int(((n + 4) * (n + 4)))

    var ctx: DeviceContext = DeviceContext()

    # Alloc device buffers
    var d_un = ctx.enqueue_create_buffer[DType.float64](buffer_size)
    var d_uc = ctx.enqueue_create_buffer[DType.float64](buffer_size)
    var d_uo = ctx.enqueue_create_buffer[DType.float64](buffer_size)
    var d_pebbles = ctx.enqueue_create_buffer[DType.float64](Int(n * n))
    ctx.synchronize()

    # Memset
    ctx.enqueue_memset[DType.float64](d_un, 0)
    ctx.enqueue_memset[DType.float64](d_un, 0)
    ctx.enqueue_memset[DType.float64](d_un, 0)
    ctx.enqueue_memset[DType.float64](d_un, 0)
    ctx.synchronize()

    # Copy host to device
    for i in range(n):
        ctx.enqueue_copy[DType.float64](
            d_uo.unsafe_ptr() + (n + 4) * (i + 2) + 2,
            u0.unsafe_ptr() + n * i,
            Int(n),
        )
        ctx.enqueue_copy[DType.float64](
            d_uc.unsafe_ptr() + (n + 4) * (i + 2) + 2,
            u1.unsafe_ptr() + n * i,
            Int(n),
        )
    ctx.enqueue_copy[DType.float64](d_pebbles, pebbles.unsafe_ptr())
    ctx.synchronize()

    var grid_size: Dim = (nblocks, nblocks)
    var block_size: Dim = (nthreads, nthreads)

    var compiled_evolve_kernel = ctx.compile_function[evolve]()

    # Core kernel loop
    while True:
        ctx.enqueue_function(
            compiled_evolve_kernel,
            d_un,
            d_uc,
            d_uo,
            d_pebbles,
            n,
            h,
            dt,
            t,
            grid_dim=grid_size,
            block_dim=block_size,
        )
        ctx.synchronize()

        var temp = d_uo
        d_uo = d_uc
        d_uc = d_un
        d_un = temp

        def tpdt(mut t: Float64, dt: Float64, tf: Float64) -> Int:
            if (t + dt) > tf:
                return 0
            t += dt
            return 1

        if not tpdt(t, dt, end_time):
            break

    # Copy back from device to host.
    for i in range(n):
        ctx.enqueue_copy(
            u.unsafe_ptr() + n * i,
            d_un.unsafe_ptr() + (n + 4) * (i + 2) + 2,
            Int(n),
        )
    ctx.synchronize()
