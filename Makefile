NPOINTS = 256
NPEBS = 4
TIME = 1.0
NTHREADS = 8

all: lake_cuda lake_mojo

lake_cuda: lakegpu.cu lake.cu
	nvcc lakegpu.cu lake.cu -o lake_cuda -O0 -lm

run_cuda: lake_cuda
	./lake_cuda $(NPOINTS) $(NPEBS) $(TIME) $(NTHREADS)

lake_mojo: lake.mojo
	mojo build -O0 lake.mojo -o lake_mojo

run_mojo: lake_mojo
	./lake_mojo $(NPOINTS) $(NPEBS) $(TIME) $(NTHREADS)

run: run_mojo run_cuda plot

plot:
	gnuplot heatmap.gnu

clean:
	rm -f lake_mojo lake_cuda *.dat *.png
