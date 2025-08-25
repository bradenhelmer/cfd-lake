set terminal png

set xrange[0:1]
set yrange[0:1]

# Cuda
set output 'lake_i_cuda.png'
plot 'lake_i_cuda.dat' using 2:1:3 with image

set output 'lake_f_cuda.png'
plot 'lake_f_cuda.dat' using 2:1:3 with image

set output 'lake_f_gpu_cuda.png'
plot 'lake_f_gpu_cuda.dat' using 2:1:3 with image

# Mojo
set output 'lake_i_mojo.png'
plot 'lake_i_mojo.dat' using 2:1:3 with image

set output 'lake_f_mojo.png'
plot 'lake_f_mojo.dat' using 2:1:3 with image

set output 'lake_f_gpu_mojo.png'
plot 'lake_f_gpu_mojo.dat' using 2:1:3 with image

# OpenMP
set output 'lake_i_omp.png'
plot 'lake_i_omp.dat' using 2:1:3 with image

set output 'lake_f_omp.png'
plot 'lake_f_omp.dat' using 2:1:3 with image

# OpenACC
set output 'lake_i_acc.png'
plot 'lake_i_acc.dat' using 2:1:3 with image

set output 'lake_f_acc.png'
plot 'lake_f_acc.dat' using 2:1:3 with image

# MPI
set output 'lake_i_mpi.png'
plot 'lake_i_mpi.dat' using 2:1:3 with image

set output 'lake_f_mpi_cpu.png'
plot 'lake_f_mpi_cpu.dat' using 2:1:3 with image

set output 'lake_f_mpi_gpu.png'
plot 'lake_f_mpi_gpu.dat' using 2:1:3 with image

