# Simulation parameters
NPOINTS = 256
NPEBS = 4
TIME = 1.0
NTHREADS = 8
ARGS = $(NPOINTS) $(NPEBS) $(TIME) $(NTHREADS)

# Compiler settings
CC = gcc
NVCC = nvcc
CFLAGS = -O3 -lm
NVCCFLAGS = -O3 -lm
MPIFLAGS = -lmpi

# Target executables
TARGETS = lake_cuda lake_mojo lake_openmp lake_openacc lake_mpi
MP_ACC_FILES = lake_mp_acc.c lake.h lake_util.h

all: $(TARGETS)

# CUDA Implementation
lake_cuda: lakegpu.cu lake.cu
	$(NVCC) $^ -o $@ -O0 -lm

# Mojo Implementation
lake_mojo: lake.mojo lakegpu.mojo
	pixi run mojo build -g $< -o $@

# OpenMP/ACC implementations
lake_openacc: $(MP_ACC_FILES)
	$(CC) -fopenacc $< -o $@ $(CFLAGS)

lake_openmp: $(MP_ACC_FILES)
	$(CC) -fopenmp $< -o $@ $(CFLAGS)

# MPI CPU & GPU implementation
lake_mpi: mpi/lake_mpi.cu mpi/lakegpu_mpi.cu
	$(NVCC) $^ -o $@ $(NVCCFLAGS) $(MPIFLAGS)

# Run targets
run_cuda: lake_cuda
	./$< $(ARGS)

run_mojo: lake_mojo
	./$< $(ARGS)

run_acc: lake_openacc
	./$< $(ARGS)

run_openmp: lake_openmp
	./$< $(ARGS)

run_mpi: lake_mpi
	mpirun -np 4 ./$< $(ARGS)

# Plotting
plot:
	gnuplot heatmap.gnu

# Run all implementations
run_all: run_mojo run_cuda run_openmp run_mpi plot

# Cleanup
clean:
	rm -f $(TARGETS) *.dat *.png lake.log

.PHONY: all run_cuda run_mojo run_acc run_openmp run_mpi plot run_all clean
