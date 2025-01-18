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


fn zero_init_list(mut lst: List[Float64]) -> None:
    for _ in range(lst.capacity):
        lst.append(0.0)


fn init_pebbles(mut p: List[Float64], pn: UInt32, n: UInt32) -> None:
    seed()

    memset_zero(p.data, p.capacity)

    var i64_casted: UInt64 = n.cast[DType.uint64]()

    for _ in range(pn):
        var i: UInt64 = random_ui64(2, i64_casted - 3)
        var j: UInt64 = random_ui64(2, i64_casted - 3)
        var sz: UInt64 = random_ui64(1, 10)
        p[int(j + i * i64_casted)] = sz.cast[DType.float64]()


fn f(p: Float64, t: Float64) -> Float64:
    return -exp(-TSCALE * t) * p


fn tpdt(mut t: Float64, dt: Float64, tf: Float64) -> UInt32:
    if (t + dt) > tf:
        return 0
    t += dt
    return 1


fn init(mut u: List[Float64], pebbles: List[Float64]) -> None:
    for idx in range(u.size):
        u[idx] = f(pebbles[idx], 0.0)


fn print_heatmap(
    filename: String, u: List[Float64], n: UInt32, h: Float64
) -> None:
    try:
        with open(filename, "w") as out_file:
            for i in range(n):
                for j in range(n):
                    var idx: UInt32 = j + i * n
                    var out_str: String = "{} {} {}\n".format(
                        i.cast[DType.float64]() * h,
                        j.cast[DType.float64]() * h,
                        u[int(idx)],
                    )
                    out_file.write(out_str)
    except e:
        print("Error writing results to file: ", e)


fn evolve(
    mut un: List[Float64],
    uc: List[Float64],
    uo: List[Float64],
    pebbles: List[Float64],
    n: UInt32,
    h: Float64,
    dt: Float64,
    t: Float64,
) -> None:
    var idx: Int
    var n1 = int(n)
    var n2 = int(n * 2)
    for i in range(n):
        for j in range(n):
            idx = int(j + i * n)
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
                un[int(idx)] = 0.0
            else:
                un[idx] = (
                    2 * uc[idx]
                    - uo[idx]
                    + VSQR
                    * (dt * dt)
                    * (
                        (
                            1
                            * (
                                uc[idx - 1]
                                + uc[idx + 1]
                                + uc[idx + n1]
                                + uc[idx - n1]
                            )
                            + 0.25
                            * (
                                uc[idx + n1 - 1]
                                + uc[idx + n1 + 1]
                                + uc[idx - n1 - 1]
                                + uc[idx - n1 + 1]
                            )
                            + 0.125
                            * (
                                uc[idx - 2]
                                + uc[idx + 2]
                                + uc[idx + n2]
                                + uc[idx - n2]
                            )
                            - 5.5 * uc[idx]
                        )
                        / (h * h)
                        + f(pebbles[idx], t)
                    )
                )


fn run_cpu(
    mut u: List[Float64],
    mut u0: List[Float64],
    mut u1: List[Float64],
    mut pebbles: List[Float64],
    n: UInt32,
    h: Float64,
    end_time: Float64,
) -> None:
    var un: List[Float64] = List[Float64](capacity=int(n * n))
    var uc: List[Float64] = List[Float64](capacity=int(n * n))
    var uo: List[Float64] = List[Float64](capacity=int(n * n))

    memcpy(uo.data, u0.data, u0.capacity)
    memcpy(uc.data, u1.data, u1.capacity)

    var t: Float64 = 0.0
    var dt: Float64 = h / 2.0

    var count = 0
    while 1:
        evolve(un, uc, uo, pebbles, n, h, dt, t)
        var temp: List[Float64] = uo
        uo = uc
        uc = un
        un = temp

        if not tpdt(t, dt, end_time):
            break

    memcpy(u.data, un.data, un.capacity)


fn main() -> None:
    print("Running mojo lake simulation...")

    var args = argv()
    if len(args) != 5:
        print("Usage: mojo run lake.mojo npoints npebs time_finish nthreads")
        exit(1)

    var npoints: UInt32 = 0
    var npebs: UInt32 = 0
    var end_time: Float64 = 0.0
    var nthreads: UInt32 = 0

    try:
        npoints = atol(args[1])
        npebs = atol(args[2])
        end_time = atof(args[3])
        nthreads = atol(args[4])
    except e:
        print("Error converting argument to int or float: ", e)

    var narea: UInt32 = npoints * npoints

    var u_i0: List[Float64] = List[Float64](capacity=int(narea))
    var u_i1: List[Float64] = List[Float64](capacity=int(narea))
    var pebs: List[Float64] = List[Float64](capacity=int(narea))
    var u_cpu: List[Float64] = List[Float64](capacity=int(narea))
    # var u_gpu: List[Float64] = List[Float64](capacity=int(narea))

    try:
        print(
            "Running lake.mojo with ({} x {}) grid, until {}, with {} threads."
            .format(npoints, npoints, end_time, nthreads)
        )
    except:
        pass

    var h: Float64 = 1.0 / npoints.cast[DType.float64]()

    init_pebbles(pebs, npebs, npoints)
    init(u_i0, pebs)
    init(u_i1, pebs)

    print_heatmap("lake_i.dat", u_i0, npoints, h)

    run_cpu(u_cpu, u_i0, u_i1, pebs, npoints, h, end_time)

    print_heatmap("lake_f.dat", u_cpu, npoints, h)
