// bthelmer braden t helmer
// hkambha harish kambhampaty
// cwkavana colin w kavanaugh
#include <cuda_runtime.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define __DEBUG
#define ROOT 0
#define TWO_ROW_PAD (n + 4) * 2

#define CUDA_CALL(err) __cudaSafeCall(err, __FILE__, __LINE__)
#define CUDA_CHK_ERR() __cudaCheckError(__FILE__, __LINE__)

/**************************************
 * void __cudaSafeCall(cudaError err, const char *file, const int line)
 * void __cudaCheckError(const char *file, const int line)
 *
 * These routines were taken from the GPU Computing SDK
 * (http://developer.nvidia.com/gpu-computing-sdk) include file "cutil.h"
 **************************************/
inline void __cudaSafeCall(cudaError err, const char *file, const int line) {
#ifdef __DEBUG

#pragma warning(push)
#pragma warning(disable : 4127) // Prevent warning on do-while(0);
  do {
    if (cudaSuccess != err) {
      fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n", file, line,
              cudaGetErrorString(err));
      exit(-1);
    }
  } while (0);
#pragma warning(pop)
#endif // __DEBUG
  return;
}

inline void __cudaCheckError(const char *file, const int line) {
#ifdef __DEBUG
#pragma warning(push)
#pragma warning(disable : 4127) // Prevent warning on do-while(0);
  do {
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
      fprintf(stderr, "cudaCheckError() failed at %s:%i : %s.\n", file, line,
              cudaGetErrorString(err));
      exit(-1);
    }
    // More careful checking. However, this will affect performance.
    // Comment if not needed.
    /*err = cudaThreadSynchronize();
    if( cudaSuccess != err )
    {
      fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s.\n",
               file, line, cudaGetErrorString( err ) );
      exit( -1 );
    }*/
  } while (0);
#pragma warning(pop)
#endif // __DEBUG
  return;
}

#define XMIN 0.0
#define XMAX 1.0
#define YMIN 0.0
#define YMAX 1.0

#define MAX_PSZ 10
#define TSCALE 1.0
#define VSQR 0.1

__device__ double d_f(double p, double t) { return -expf(-TSCALE * t) * p; }

int tpdt(double *t, double dt, double tf) {
  if ((*t) + dt > tf)
    return 0;
  (*t) = (*t) + dt;
  return 1;
}

// Kernel taken from lakegpu.cu
__global__ void d_evolve(double *d_un, double *d_uc, double *d_uo,
                         double *pebbles, int n, double h, double dt,
                         double t) {
  int row = gridDim.x * blockDim.x;
  int p_row = gridDim.x * blockDim.x + 4;
  int p_offset = 2 * p_row + 2;
  int i = (blockIdx.y * blockDim.y + threadIdx.y) * row +
          (blockIdx.x * blockDim.x + threadIdx.x);
  int idx = (blockIdx.y * blockDim.y + threadIdx.y) * p_row +
            (blockIdx.x * blockDim.x + threadIdx.x) + p_offset;

  d_un[idx] =
      2 * d_uc[idx] - d_uo[idx] +
      VSQR * (dt * dt) *
          ((
               // 1st degree cardinals
               1 * (d_uc[idx - 1] + d_uc[idx + 1] + d_uc[idx + p_row] +
                    d_uc[idx - p_row]) +
               // 1st degree ordinals
               0.25 * (d_uc[idx + p_row - 1] + d_uc[idx + p_row + 1] +
                       d_uc[idx - p_row - 1] + d_uc[idx - p_row + 1]) +
               // 2nd degree cardinals
               0.125 * (d_uc[idx - 2] + d_uc[idx + 2] +
                        d_uc[idx + p_row + p_row] + d_uc[idx - p_row - p_row]) -
               5.5 * d_uc[idx]) /
               (h * h) + // normalization
           d_f(pebbles[i], t));
}

void run_gpu_mpi(double *u, double *u0, double *u1, double *pebbles, int n,
                 double h, double end_time, int nthreads, int proc_rank,
                 int numprocs) {
  cudaEvent_t kstart, kstop;
  float ktime;

  double t, dt;
  t = 0.;
  dt = h / 2.;

  /* Set up device timers */
  CUDA_CALL(cudaSetDevice(0));
  CUDA_CALL(cudaEventCreate(&kstart));
  CUDA_CALL(cudaEventCreate(&kstop));

  // MPI Chunking
  const int mpi_chunk_size = n / numprocs;
  int nblocks = n / nthreads;
  double *dchunk_un, *dchunk_uc, *dchunk_uo, *dpebs;

  // Allocs
  cudaMalloc((void **)&dchunk_un,
             sizeof(double) * (n + 4) * (mpi_chunk_size + 4));
  cudaMalloc((void **)&dchunk_uc,
             sizeof(double) * (n + 4) * (mpi_chunk_size + 4));
  cudaMalloc((void **)&dchunk_uo,
             sizeof(double) * (n + 4) * (mpi_chunk_size + 4));
  cudaMalloc((void **)&dpebs, sizeof(double) * n * mpi_chunk_size);

  // Set everything to zero first
  cudaMemset(dchunk_un, 0, (n + 4) * (mpi_chunk_size + 4));
  cudaMemset(dchunk_uc, 0, (n + 4) * (mpi_chunk_size + 4));
  cudaMemset(dchunk_uo, 0, (n + 4) * (mpi_chunk_size + 4));

  // Dimensions
  dim3 grid_size(nblocks, nblocks / numprocs);
  dim3 block_size(nthreads, nthreads);

  // Copy host arrays to device
  cudaMemcpy(dpebs, pebbles, sizeof(double) * n * mpi_chunk_size,
             cudaMemcpyHostToDevice);
  for (int i = 0; i < mpi_chunk_size; i++) {
    cudaMemcpy(dchunk_uo + (n + 4) * (i + 2) + 2, u0 + (n * i),
               sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(dchunk_uc + (n + 4) * (i + 2) + 2, u1 + (n * i),
               sizeof(double) * n, cudaMemcpyHostToDevice);
  }

  /* Start GPU computation timer */
  CUDA_CALL(cudaEventRecord(kstart, 0));
  cudaDeviceSynchronize();
  if (proc_rank < (numprocs - 1)) {

    // Forward send (last 2 rows not in boundary)
    double *send_buf = (double *)malloc(sizeof(double) * TWO_ROW_PAD);
    cudaMemcpy(send_buf, dchunk_uo + (mpi_chunk_size * (n + 4)),
               sizeof(double) * TWO_ROW_PAD, cudaMemcpyDeviceToHost);
    MPI_Send(send_buf, TWO_ROW_PAD, MPI_DOUBLE, proc_rank + 1, 0,
             MPI_COMM_WORLD);

    // Backward receive (into last 2 boundary rows)
    double *recv_buf = (double *)malloc(sizeof(double) * TWO_ROW_PAD);
    MPI_Recv(recv_buf, TWO_ROW_PAD, MPI_DOUBLE, proc_rank + 1, 0,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    cudaMemcpy(dchunk_uo + (mpi_chunk_size + 2) * (n + 4), recv_buf,
               sizeof(double) * TWO_ROW_PAD, cudaMemcpyHostToDevice);
  }
  if (proc_rank > ROOT) {
    // Backward send (top level 2 rows not in boundary)
    double *send_buf = (double *)malloc(sizeof(double) * TWO_ROW_PAD);
    cudaMemcpy(send_buf, dchunk_uo + TWO_ROW_PAD, sizeof(double) * TWO_ROW_PAD,
               cudaMemcpyDeviceToHost);
    MPI_Send(send_buf, TWO_ROW_PAD, MPI_DOUBLE, proc_rank - 1, 0,
             MPI_COMM_WORLD);

    // forward receive (receive into top level boundary rows)
    double *recv_buf = (double *)malloc(sizeof(double) * TWO_ROW_PAD);
    MPI_Recv(recv_buf, TWO_ROW_PAD, MPI_DOUBLE, proc_rank - 1, 0,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    cudaMemcpy(dchunk_uo, recv_buf, sizeof(double) * TWO_ROW_PAD,
               cudaMemcpyHostToDevice);
    free(send_buf);
    free(recv_buf);
  }

  do {
    // Update duc
    if (proc_rank < (numprocs - 1)) {

      // Forward send (last 2 rows not in boundary)
      double *send_buf = (double *)malloc(sizeof(double) * TWO_ROW_PAD);
      cudaMemcpy(send_buf, dchunk_uc + (mpi_chunk_size * (n + 4)),
                 sizeof(double) * TWO_ROW_PAD, cudaMemcpyDeviceToHost);
      MPI_Send(send_buf, TWO_ROW_PAD, MPI_DOUBLE, proc_rank + 1, 0,
               MPI_COMM_WORLD);

      // Backward receive (into last 2 boundary rows)
      double *recv_buf = (double *)malloc(sizeof(double) * TWO_ROW_PAD);
      MPI_Recv(recv_buf, TWO_ROW_PAD, MPI_DOUBLE, proc_rank + 1, 0,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      cudaMemcpy(dchunk_uc + (mpi_chunk_size + 2) * (n + 4), recv_buf,
                 sizeof(double) * TWO_ROW_PAD, cudaMemcpyHostToDevice);
      free(send_buf);
      free(recv_buf);
    }
    if (proc_rank > ROOT) {
      // Backward send (top level 2 rows not in boundary)
      double *send_buf = (double *)malloc(sizeof(double) * TWO_ROW_PAD);
      cudaMemcpy(send_buf, dchunk_uc + TWO_ROW_PAD,
                 sizeof(double) * TWO_ROW_PAD, cudaMemcpyDeviceToHost);
      MPI_Send(send_buf, TWO_ROW_PAD, MPI_DOUBLE, proc_rank - 1, 0,
               MPI_COMM_WORLD);

      // forward receive (receive into top level boundary rows)
      double *recv_buf = (double *)malloc(sizeof(double) * TWO_ROW_PAD);
      MPI_Recv(recv_buf, TWO_ROW_PAD, MPI_DOUBLE, proc_rank - 1, 0,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      cudaMemcpy(dchunk_uc, recv_buf, sizeof(double) * TWO_ROW_PAD,
                 cudaMemcpyHostToDevice);
      free(send_buf);
      free(recv_buf);
    }

    // Call cuda kernel
    d_evolve<<<grid_size, block_size>>>(dchunk_un, dchunk_uc, dchunk_uo, dpebs,
                                        n, h, dt, t);

    double *temp = dchunk_uo;
    dchunk_uo = dchunk_uc;
    dchunk_uc = dchunk_un;
    dchunk_un = temp;
  } while (tpdt(&t, dt, end_time));

  // Copy back from device
  for (int i = 0; i < mpi_chunk_size; i++) {
    cudaMemcpy(u + n * i, dchunk_un + (n + 4) * (i + 2) + 2, sizeof(double) * n,
               cudaMemcpyDeviceToHost);
  }

  /* Stop GPU computation timer */
  CUDA_CALL(cudaEventRecord(kstop, 0));
  CUDA_CALL(cudaEventSynchronize(kstop));
  CUDA_CALL(cudaEventElapsedTime(&ktime, kstart, kstop));
  printf("GPU computation: %f msec\n", ktime);

  /* HW2: Add post CUDA kernel call processing and cleanup here */
  cudaFree(dchunk_un);
  cudaFree(dchunk_uc);
  cudaFree(dchunk_uo);
  cudaFree(dpebs);

  /* timer cleanup */
  CUDA_CALL(cudaEventDestroy(kstart));
  CUDA_CALL(cudaEventDestroy(kstop));
}
