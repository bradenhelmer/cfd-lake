// bthelmer braden t helmer
// hkambha harish kambhampaty
// cwkavana colin w kavanaugh
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#define _USE_MATH_DEFINES

#define XMIN 0.0
#define XMAX 1.0
#define YMIN 0.0
#define YMAX 1.0

#define MAX_PSZ 10
#define TSCALE 1.0
#define VSQR 0.1

#define ROOT 0

void init(double *u, double *pebbles, int n);
void evolve_mpi(double *un, double *uc, double *uo, double *pebbles, int n,
                double h, double dt, double t, int chunk_size);
extern int tpdt(double *t, double dt, double end_time);
void print_heatmap(const char *filename, double *u, int n, double h);
void print_heatmap(const char *filename, double *u, int n, double h,
                   int chunk_width);
void init_pebbles(double *p, int pn, int n);
void run_cpu_mpi(double *u, double *u0, double *u1, double *pebbles, int n,
                 double h, double end_time, int proc_rank, int numprocs);
extern void run_gpu_mpi(double *u, double *u0, double *u1, double *pebbles,
                        int n, double h, double end_time, int nthreads,
                        int proc_rank, int numprocs);

int main(int argc, char *argv[]) {

  if (argc != 5) {
    printf("Usage: %s npoints npebs time_finish nthreads \n", argv[0]);
    return 0;
  }

  // MPI_Initialization
  int rank, size;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int npoints = atoi(argv[1]);
  int npebs = atoi(argv[2]);
  double end_time = (double)atof(argv[3]);
  int nthreads = atoi(argv[4]);
  int narea = npoints * npoints;

  // MPI Proc chunk arrays;
  double *chunk_uGPU, *chunk_u0, *chunk_u1, *chunk_uCPU;
  // Root arrays
  double *root_uGPU, *root_u0, *root_u1, *root_uCPU;

  // Pebbles
  double *pebs, *chunk_pebs;
  double h;

  double elapsed_cpu, elapsed_gpu;
  struct timeval cpu_start, cpu_end, gpu_start, gpu_end;

  // Proc chunk allocations
  const int chunk_size = narea / size;
  chunk_uGPU = (double *)malloc(sizeof(double) * chunk_size);
  chunk_u0 = (double *)malloc(sizeof(double) * chunk_size);
  chunk_u1 = (double *)malloc(sizeof(double) * chunk_size);
  chunk_pebs = (double *)malloc(sizeof(double) * chunk_size);
  chunk_uCPU = (double *)malloc(sizeof(double) * chunk_size);

  // Root full allocation
  if (rank == ROOT) {
    root_uGPU = (double *)malloc(sizeof(double) * narea);
    pebs = (double *)malloc(sizeof(double) * narea);
    root_u0 = (double *)malloc(sizeof(double) * narea);
    root_u1 = (double *)malloc(sizeof(double) * narea);
    root_uCPU = (double *)malloc(sizeof(double) * narea);

    // Send pebbles and initials to children
    init_pebbles(pebs, npebs, npoints);
    init(root_u0, pebs, npoints);
    init(root_u1, pebs, npoints);
    for (int i = 0; i < size; i++) {
      MPI_Send(pebs + (i * chunk_size), chunk_size, MPI_DOUBLE, i, 0,
               MPI_COMM_WORLD);
      MPI_Send(root_u0 + (i * chunk_size), chunk_size, MPI_DOUBLE, i, 0,
               MPI_COMM_WORLD);
      MPI_Send(root_u1 + (i * chunk_size), chunk_size, MPI_DOUBLE, i, 0,
               MPI_COMM_WORLD);
    }
  }

  // Receive chunks
  MPI_Recv(chunk_pebs, chunk_size, MPI_DOUBLE, ROOT, 0, MPI_COMM_WORLD,
           MPI_STATUS_IGNORE);
  MPI_Recv(chunk_u0, chunk_size, MPI_DOUBLE, ROOT, 0, MPI_COMM_WORLD,
           MPI_STATUS_IGNORE);
  MPI_Recv(chunk_u1, chunk_size, MPI_DOUBLE, ROOT, 0, MPI_COMM_WORLD,
           MPI_STATUS_IGNORE);

  h = (XMAX - XMIN) / npoints;

  // CPU MPI
  if (rank == ROOT) {
    gettimeofday(&cpu_start, NULL);
  }

  // Main call
  run_cpu_mpi(chunk_uCPU, chunk_u0, chunk_u1, chunk_pebs, npoints, h, end_time,
              rank, size);

  // Collection of chunks for CPU
  if (rank == ROOT) {
    memcpy(root_uCPU, chunk_uCPU, sizeof(double) * chunk_size);
    for (int i = 1; i < size; i++) {
      MPI_Recv(root_uCPU + (chunk_size * i), chunk_size, MPI_DOUBLE, i, 0,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Get end time when root has collected all chunks.
    gettimeofday(&cpu_end, NULL);
    elapsed_cpu = ((cpu_end.tv_sec + cpu_end.tv_usec * 1e-6) -
                   (cpu_start.tv_sec + cpu_start.tv_usec * 1e-6));
    // Global data output
    printf("CPU took %f seconds\n", elapsed_cpu);
    print_heatmap("lake_i_mpi.dat", root_u1, npoints, h);
    print_heatmap("lake_f_mpi_cpu.dat", root_uCPU, npoints, h);
    printf("Outputting MPI CPU data...\n");
  } else {
    // Send chunk to root
    MPI_Send(chunk_uCPU, chunk_size, MPI_DOUBLE, ROOT, 0, MPI_COMM_WORLD);
  }

  // Proc specific data files for CPU
  char filename[25];
  snprintf(filename, 25, "lake_f_mpi_cpu_%d.dat", rank);
  print_heatmap(filename, chunk_uCPU, npoints, h, npoints / size);

  // BEGIN GPU MPI
  if (rank == ROOT) {
    printf("Running %s with (%d x %d) grid, until %f, with %d threads\n",
           argv[0], npoints, npoints, end_time, nthreads);

    gettimeofday(&gpu_start, NULL);
  }

  // Main GPU call
  run_gpu_mpi(chunk_uGPU, chunk_u0, chunk_u1, chunk_pebs, npoints, h, end_time,
              nthreads, rank, size);

  // Collection of chunks for GPU
  if (rank == ROOT) {
    memcpy(root_uGPU, chunk_uGPU, sizeof(double) * chunk_size);
    for (int i = 1; i < size; i++) {
      MPI_Recv(root_uGPU + (chunk_size * i), chunk_size, MPI_DOUBLE, i, 0,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Get end time when root has collected all chunks.
    gettimeofday(&gpu_end, NULL);
    elapsed_gpu = ((gpu_end.tv_sec + gpu_end.tv_usec * 1e-6) -
                   (gpu_start.tv_sec + gpu_start.tv_usec * 1e-6));
    // Global data output
    printf("GPU took %f seconds\n", elapsed_gpu);
    printf("Outputting MPI GPU data...\n");
    print_heatmap("lake_f_mpi_gpu.dat", root_uGPU, npoints, h);
  } else {
    // Send chunk to root
    MPI_Send(chunk_uGPU, chunk_size, MPI_DOUBLE, ROOT, 0, MPI_COMM_WORLD);
  }

  // Proc specific data files for GPU
  snprintf(filename, 25, "lake_f_mpi_gpu_%d.dat", rank);
  print_heatmap(filename, chunk_uGPU, npoints, h, npoints / size);

  // Free all chunk memory
  free(chunk_u0);
  free(chunk_u1);
  free(chunk_pebs);
  free(chunk_uGPU);
  free(chunk_uCPU);

  // Free root specific memory;
  if (rank == ROOT) {
    free(root_uCPU);
    free(root_uGPU);
    free(pebs);
    free(root_u0);
    free(root_u1);
  }

  // All important call to MPI finalize.
  MPI_Finalize();

  return 0;
}

void run_cpu_mpi(double *u, double *u0, double *u1, double *pebbles, int n,
                 double h, double end_time, int proc_rank, int numprocs) {

  double *un, *uc, *uo, *upebs;
  double t, dt;

  // Get chunk row size
  const int mpi_chunk_size = n / numprocs;

  // Allocate CPU memory with space for boundary values.
  un = (double *)malloc(sizeof(double) * (n + 4) * (mpi_chunk_size + 4));
  uc = (double *)malloc(sizeof(double) * (n + 4) * (mpi_chunk_size + 4));
  uo = (double *)malloc(sizeof(double) * (n + 4) * (mpi_chunk_size + 4));
  upebs = (double *)malloc(sizeof(double) * (n * mpi_chunk_size));

  // Memset new arrays to zero.
  memset(un, 0, sizeof(double) * (n + 4) * (mpi_chunk_size + 4));
  memset(uo, 0, sizeof(double) * (n + 4) * (mpi_chunk_size + 4));
  memset(uc, 0, sizeof(double) * (n + 4) * (mpi_chunk_size + 4));

  // Copy chunk arrays into function scoped arrays.
  memcpy(upebs, pebbles, sizeof(double) * n * mpi_chunk_size);
  for (int i = 0; i < mpi_chunk_size; i++) {
    memcpy(uo + (n + 4) * (i + 2) + 2, u0 + n * i, sizeof(double) * n);
    memcpy(uc + (n + 4) * (i + 2) + 2, u1 + n * i, sizeof(double) * n);
  }

  t = 0.;
  dt = h / 2.;

  // Initial MPI sends for u old values
  if (proc_rank < (numprocs - 1)) {
    // Forward send (last 2 rows not in boundary)
    MPI_Send(uo + (mpi_chunk_size * (n + 4)), (n + 4) * 2, MPI_DOUBLE,
             proc_rank + 1, 0, MPI_COMM_WORLD);
    // Backward receive (into last 2 boundary rows)
    MPI_Recv(uo + ((mpi_chunk_size + 2) * (n + 4)), (n + 4) * 2, MPI_DOUBLE,
             proc_rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
  if (proc_rank > ROOT) {
    // Backward send (top level 2 rows not in boundary)
    MPI_Send(uo + (2 * (n + 4)), (n + 4) * 2, MPI_DOUBLE, proc_rank - 1, 0,
             MPI_COMM_WORLD);
    // forward receive (receive into top level boundary rows)
    MPI_Recv(uo, (n + 4) * 2, MPI_DOUBLE, proc_rank - 1, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
  }

  do {
    // Update uc
    if (proc_rank < (numprocs - 1)) {
      // Forward send (last 2 rows not in boundary)
      MPI_Send(uc + (mpi_chunk_size * (n + 4)), (n + 4) * 2, MPI_DOUBLE,
               proc_rank + 1, 0, MPI_COMM_WORLD);
      // Backward receive (into last 2 boundary rows)
      MPI_Recv(uc + ((mpi_chunk_size + 2) * (n + 4)), (n + 4) * 2, MPI_DOUBLE,
               proc_rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if (proc_rank > ROOT) {
      // Backward send (top level 2 rows not in boundary)
      MPI_Send(uc + (2 * (n + 4)), (n + 4) * 2, MPI_DOUBLE, proc_rank - 1, 0,
               MPI_COMM_WORLD);
      // forward receive (receive into top level boundary rows)
      MPI_Recv(uc, (n + 4) * 2, MPI_DOUBLE, proc_rank - 1, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
    }

    // Do calculation
    evolve_mpi(un, uc, uo, upebs, n, h, dt, t, mpi_chunk_size);

    // Change pointers
    double *temp = uo;
    uo = uc;
    uc = un;
    un = temp;

  } while (tpdt(&t, dt, end_time));

  // Copy into output chunk array
  for (int i = 0; i < mpi_chunk_size; i++) {
    memcpy(u + n * i, un + (n + 4) * (i + 2) + 2, sizeof(double) * n);
  }

  // Free allocated run_cpu memory
  free(uo);
  free(uc);
  free(un);
  free(upebs);
}

void init_pebbles(double *p, int pn, int n) {
  int i, j, k, idx;
  int sz;

  srand(time(NULL));
  memset(p, 0, sizeof(double) * n * n);

  for (k = 0; k < pn; k++) {
    i = rand() % (n - 4) + 2;
    j = rand() % (n - 4) + 2;
    sz = rand() % MAX_PSZ;
    idx = j + i * n;
    p[idx] = (double)sz;
  }
}

double f(double p, double t) { return -expf(-TSCALE * t) * p; }

void init(double *u, double *pebbles, int n) {
  int i, j, idx;

  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      idx = j + i * n;
      u[idx] = f(pebbles[idx], 0.0);
    }
  }
}

void evolve_mpi(double *un, double *uc, double *uo, double *pebbles, int n,
                double h, double dt, double t, int chunk_size) {
  int i, j, idx;

  // Padded row size
  const int p_row = n + 4;

  for (i = 2; i < chunk_size + 2; i++) {
    for (j = 2; j < n + 2; j++) {
      // un index
      idx = (i * p_row) + j;

      // Remove padding for pebble index
      int idx_p = (i - 2) * n + (j - 2);

      // Calculate
      un[idx] =
          2 * uc[idx] - uo[idx] +
          VSQR * (dt * dt) *
              ((1 * (uc[idx - 1] + uc[idx + 1] + uc[idx + p_row] +
                     uc[idx - p_row]) + // 1st degree cardinals
                0.25 * (uc[idx + p_row - 1] + uc[idx + p_row + 1] +
                        uc[idx - p_row - 1] +
                        uc[idx - p_row + 1]) + // 1st degree ordinals
                0.125 * (uc[idx - 2] + uc[idx + 2] + uc[idx + p_row + p_row] +
                         uc[idx - p_row - p_row]) - // 2nd degree cardinals
                5.5 * uc[idx]) /
                   (h * h) + // normalization
               f(pebbles[idx_p], t));
    }
  }
}

void print_heatmap(const char *filename, double *u, int n, double h) {
  int i, j, idx;

  FILE *fp = fopen(filename, "w");

  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      idx = j + i * n;
      fprintf(fp, "%f %f %f\n", i * h, j * h, u[idx]);
    }
  }

  fclose(fp);
}

void print_heatmap(const char *filename, double *u, int n, double h,
                   int chunk_width) {
  int i, j, idx;

  FILE *fp = fopen(filename, "w");

  for (i = 0; i < n; i++) {
    for (j = 0; j < chunk_width; j++) {
      idx = (i * chunk_width) + j;
      fprintf(fp, "%f %f %f\n", i * h, j * h, u[idx]);
    }
  }

  fclose(fp);
}
