#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define __DEBUG

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

  printf("Running in d_evolve\n");

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

void run_gpu(double *u, double *u0, double *u1, double *pebbles, int n,
             double h, double end_time, int nthreads) {
  cudaEvent_t kstart, kstop;
  float ktime;

  /* HW2: Define your local variables here */

  double t, dt;
  t = 0.;
  dt = h / 2.;

  /* Set up device timers */
  CUDA_CALL(cudaSetDevice(0));
  CUDA_CALL(cudaEventCreate(&kstart));
  CUDA_CALL(cudaEventCreate(&kstop));

  /* HW2: Add CUDA kernel call preparation code here */
  int nblocks = n / nthreads;

  double *d_un, *d_uc, *d_uo, *d_pebbles;
  cudaMalloc((void **)&d_un, sizeof(double) * (n + 4) * (n + 4));
  cudaMalloc((void **)&d_uc, sizeof(double) * (n + 4) * (n + 4));
  cudaMalloc((void **)&d_uo, sizeof(double) * (n + 4) * (n + 4));
  cudaMalloc((void **)&d_pebbles, sizeof(double) * n * n);

  cudaMemset(d_un, 0, sizeof(double) * (n + 4) * (n + 4));
  cudaMemset(d_uc, 0, sizeof(double) * (n + 4) * (n + 4));
  cudaMemset(d_uo, 0, sizeof(double) * (n + 4) * (n + 4));
  cudaMemset(d_pebbles, 0, sizeof(double) * n * n);

  dim3 grid_size(nblocks, nblocks);
  dim3 block_size(nthreads, nthreads);

  //    printf("grid_size:%d, block_size:%d\n", nblocks, nthreads);

  /* Start GPU computation timer */
  CUDA_CALL(cudaEventRecord(kstart, 0));

  /* HW2: Add main lake simulation loop here */
  // memcpy: Host -> Device
  for (int i = 0; i < n; i++) {
    //        printf("%d <-> %d\n", (n + 4) * (i + 2) + 2, n * i);
    cudaMemcpy(d_uo + (n + 4) * (i + 2) + 2, u0 + n * i, sizeof(double) * n,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_uc + (n + 4) * (i + 2) + 2, u1 + n * i, sizeof(double) * n,
               cudaMemcpyHostToDevice);
  }
  cudaMemcpy(d_pebbles, pebbles, sizeof(double) * n * n,
             cudaMemcpyHostToDevice);

  while (1) {
    d_evolve<<<grid_size, block_size>>>(d_un, d_uc, d_uo, d_pebbles, n, h, dt,
                                        t);

    double *temp = d_uo;
    d_uo = d_uc;
    d_uc = d_un;
    d_un = temp;

    if (!tpdt(&t, dt, end_time))
      break;
  }

  // memcpy: Device -> Host
  for (int i = 0; i < n; i++) {
    cudaMemcpy(u + n * i, d_un + (n + 4) * (i + 2) + 2, sizeof(double) * n,
               cudaMemcpyDeviceToHost);
  }

  /* Stop GPU computation timer */
  CUDA_CALL(cudaEventRecord(kstop, 0));
  CUDA_CALL(cudaEventSynchronize(kstop));
  CUDA_CALL(cudaEventElapsedTime(&ktime, kstart, kstop));
  printf("GPU computation: %f msec\n", ktime);

  /* HW2: Add post CUDA kernel call processing and cleanup here */
  cudaFree(d_un);
  cudaFree(d_uc);
  cudaFree(d_uo);
  cudaFree(d_pebbles);

  /* timer cleanup */
  CUDA_CALL(cudaEventDestroy(kstart));
  CUDA_CALL(cudaEventDestroy(kstop));
}
