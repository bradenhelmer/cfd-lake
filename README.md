# CFD Lake: Ripple Simulation

A high-performance computational fluid dynamics (CFD) simulation that models ripples propagating across a lake surface using centralized finite difference methods with a 13-point stencil. This project implements the same simulation in multiple languages and parallel computing paradigms to demonstrate performance comparisons.

## Overview

The simulation models pebbles dropped into a lake, creating ripples that propagate across the water surface. The physics are computed using the 2D wave equation solved with finite difference methods, providing an accurate representation of wave propagation dynamics.

## Features

- **Multiple Implementation Languages**: Mojo, CUDA C/C++
- **Parallel Computing Support**: 
  - GPU acceleration (CUDA)
  - Multi-threading
  - MPI distributed computing
  - OpenMP/OpenACC parallelization
- **Visualization**: Automatic heatmap generation using gnuplot
- **Performance Comparison**: Side-by-side execution of different implementations

## Project Structure

```
cfd-lake/
├── lake.mojo           # Mojo CPU implementation
├── lakegpu.mojo        # Mojo GPU implementation
├── lake.cu             # CUDA C++ CPU implementation
├── lakegpu.cu          # CUDA C++ GPU implementation
├── lake_mp_acc.c       # OpenMP/OpenACC implementation
├── lake.h              # Common function declarations and constants
├── lake_util.h         # Utility functions for logging and file I/O
├── mpi/                # MPI distributed implementations
│   ├── lake_mpi.cu     # MPI CPU + GPU implementation
│   └── lakegpu_mpi.cu  # MPI GPU kernels
├── heatmap.gnu         # Gnuplot visualization script
├── pixi.toml           # Pixi package configuration
├── pixi.lock           # Pixi lock file
└── Makefile            # Build configuration
```

## Quick Start

### Prerequisites

- **CUDA Toolkit** (for GPU implementations)
- **Pixi** package manager (for Mojo environment management)
- **Mojo** compiler (installed via Pixi)
- **GCC** (for C/C++ implementations with OpenMP/OpenACC support)
- **MPI** library (MPICH or OpenMPI for distributed implementations)
- **gnuplot** (for visualization)

### Build and Run

```bash
# Build all implementations
make all

# Run both Mojo and CUDA implementations with visualization
make run

# Run individual implementations
make run_mojo      # Mojo implementation
make run_cuda      # CUDA implementation
make run_openmp    # OpenMP implementation
make run_acc       # OpenACC implementation
make run_mpi       # MPI CPU and GPU implementations (4 processes)

# Run all implementations with visualization
make run_all

# Clean build artifacts
make clean
```

### Individual Build Targets

```bash
# Build specific implementations
make lake_mojo     # Build Mojo implementation
make lake_cuda     # Build CUDA implementation  
make lake_openmp   # Build OpenMP implementation
make lake_openacc  # Build OpenACC implementation
make lake_mpi      # Build MPI CPU and GPU implementation
```

### Debug

```bash
# Debug Mojo implementation with cuda-gdb
make debug_mojo
```

## Configuration

Default simulation parameters (configurable in Makefile):

- **Grid Size**: 256×256 points
- **Pebbles**: 4 initial disturbances
- **Simulation Time**: 1.0 time units
- **Threads**: 8 (for multi-threaded implementations)

Modify these values in the Makefile:
```makefile
NPOINTS = 256    # Grid resolution
NPEBS = 4        # Number of pebbles
TIME = 1.0       # Simulation duration
NTHREADS = 8     # Thread count
```

## Physics Model

The simulation solves the 2D wave equation:

```
∂²u/∂t² = c²(∂²u/∂x² + ∂²u/∂y²) + f(x,y,t)
```

Where:
- `u(x,y,t)` represents the water surface height
- `c` is the wave propagation speed
- `f(x,y,t)` represents external forces (pebble impacts)

The finite difference discretization uses a 13-point stencil for high accuracy spatial derivatives.

## Output

The simulation generates:

1. **Data files**: `lake_*.dat` containing simulation results
2. **Visualizations**: `lake_*.png` heatmap images showing wave propagation
3. **Performance metrics**: Execution times for different implementations

## Performance Comparison

This project enables direct performance comparison between:

- **Languages**: Mojo vs CUDA C++
- **Execution Models**: CPU vs GPU
- **Parallelization**: Single-threaded vs multi-threaded vs distributed (MPI)
- **Acceleration**: Standard vs OpenMP/OpenACC

## License

Open source project for educational and research purposes.
