set terminal png

set xrange[0:1]
set yrange[0:1]

set output 'lake_i_cuda.png'
plot 'lake_i_cuda.dat' using 2:1:3 with image

set output 'lake_i_mojo.png'
plot 'lake_i_mojo.dat' using 2:1:3 with image

set output 'lake_f_cuda.png'
plot 'lake_f_cuda.dat' using 2:1:3 with image

set output 'lake_f_gpu_cuda.png'
plot 'lake_f_gpu_cuda.dat' using 2:1:3 with image

set output 'lake_f_mojo.png'
plot 'lake_f_mojo.dat' using 2:1:3 with image

set output 'lake_f_gpu_mojo.png'
plot 'lake_f_gpu_mojo.dat' using 2:1:3 with image
