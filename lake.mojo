from collections import List
from collections.string import atof
from collections.string import atol
from math import exp
from memory import AddressSpace
from memory import memcpy
from memory import memset_zero
from random import seed
from random import random_ui64
from sys.terminate import exit
from sys import argv
from time import time_function


alias TSCALE = 1.0
alias VSQR = 0.1


fn init_pebbles(mut p: List[Float32], pn: UInt32, n: UInt32) -> None:

    memset_zero(p.data, p.capacity)

    # var i64_casted: UInt64 = n.cast[DType.uint64]()

    p[32*128 + 32] = 2
    p[32*128 + 96] = 2
    p[96*128 + 32] = 2
    p[96*128 + 96] = 2

     # for _ in range(pn):
     #     var i: UInt64 = random_ui64(2, i64_casted - 3)
     #     var j: UInt64 = random_ui64(2, i64_casted - 3)
     #     var sz: UInt64 = random_ui64(1, 10)
     #     p[Int(j + i * i64_casted)] = sz.cast[DType.float32]()


fn f(p: Float32, t: Float32) -> Float32:
    return -exp(-TSCALE * t) * p


fn tpdt(mut t: Float32, dt: Float32, tf: Float32) -> UInt32:
    if (t + dt) > tf:
        return 0
    t += dt
    return 1


fn init(mut u: List[Float32], pebbles: List[Float32]) -> None:
    for idx in range(u.size):
        u[idx] = f(pebbles[idx], 0.0)


fn print_heatmap(
    filename: String, u: List[Float32], n: UInt32, h: Float32
) -> None:
    try:
        with open(filename, "w") as out_file:
            for i in range(n):
                for j in range(n):
                    var idx: UInt32 = j + i * n
                    var out_str: String = "{} {} {}\n".format(
                        i.cast[DType.float32]() * h,
                        j.cast[DType.float32]() * h,
                        u[Int(idx)],
                    )
                    out_file.write(out_str)
    except e:
        print("Error writing results to file: ", e)


fn evolve(
    mut un: List[Float32],
    uc: List[Float32],
    uo: List[Float32],
    pebbles: List[Float32],
    n: UInt32,
    h: Float32,
    dt: Float32,
    t: Float32,
) -> None:
    var idx: Int
    for i in range(n):
        for j in range(n):
            idx = Int(j + i * n)
            if (
                i == 0
                or i == 1
                or i == n - 1
                or i == n - 2
                or j == 0
                or j == 1
                or j == n - 1
                or j == n - 2
            ):
                un[idx] = 0.0
            else:
                un[idx] =
                    2 * uc[idx] - uo[idx] +
                    VSQR * (dt * dt) *
                        ((1 * (uc[idx - 1] + uc[idx + 1] + uc[idx + n] +
                               uc[idx - n]) + # 1st degree cardinals
                          0.25 * (uc[idx + n - 1] + uc[idx + n + 1] + uc[idx - n - 1] +
                                  uc[idx - n + 1]) + # 1st degree ordinals
                          0.125 * (uc[idx - 2] + uc[idx + 2] + uc[idx + n + n] +
                                   uc[idx - n - n]) - # 2nd degree cardinals
                          5.5 * uc[idx]) /
                             (h * h) + # normalization
                         f(pebbles[idx], t));




fn run_cpu(
    mut u: List[Float32],
    u0: List[Float32],
    u1: List[Float32],
    pebbles: List[Float32],
    n: UInt32,
    h: Float32,
    end_time: Float32,
) -> None:
    var un: List[Float32] = List[Float32](capacity=Int(n * n))
    var uc: List[Float32] = List[Float32](capacity=Int(n * n))
    var uo: List[Float32] = List[Float32](capacity=Int(n * n))

    memcpy(uo.data, u0.data, u0.capacity)
    memcpy(uc.data, u1.data, u1.capacity)

    var t: Float32 = 0.0
    var dt: Float32 = h / 2.0

    while 1:
        evolve(un, uc, uo, pebbles, n, h, dt, t)
        var temp = uo
        uo = uc
        uc = un
        un = temp

        if not tpdt(t, dt, end_time):
            break

    memcpy(u.data, un.data, un.capacity)


fn main() raises:
    print("Running mojo lake simulation...")

    var args = argv()
    if len(args) != 5:
        print("Usage: mojo run lake.mojo npoints npebs time_finish nthreads")
        exit(1)

    var npoints: UInt32 = 0
    var npebs: UInt32 = 0
    var end_time: Float32 = 0.0
    var nthreads: UInt32 = 0

    try:
        npoints = atol(args[1])
        npebs = atol(args[2])
        end_time = atof(args[3]).cast[DType.float32]()
        nthreads = atol(args[4])
    except e:
        print("Error converting argument to int or float: ", e)

    var narea: UInt32 = npoints * npoints

    var u_i0: List[Float32] = List[Float32](capacity=Int(narea))
    var u_i1: List[Float32] = List[Float32](capacity=Int(narea))
    var pebs: List[Float32] = List[Float32](capacity=Int(narea))
    var u_cpu: List[Float32] = List[Float32](capacity=Int(narea))
    # var u_gpu: List[Float32] = List[Float32](capacity=Int(narea))

    try:
        print(
            "Running lake.mojo with ({} x {}) grid, until {}, with {} threads."
            .format(npoints, npoints, end_time, nthreads)
        )
    except:
        pass

    var h: Float32 = 1.0 / npoints.cast[DType.float32]()

    init_pebbles(pebs, npebs, npoints)
    init(u_i0, pebs)
    init(u_i1, pebs)

    print_heatmap("lake_i_mojo.dat", u_i0, npoints, h)

    run_cpu(u_cpu, u_i0, u_i1, pebs, npoints, h, end_time)

    print_heatmap("lake_f_mojo.dat", u_cpu, npoints, h)
