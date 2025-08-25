// bthelmer Braden T Helmer
/*************************************
 * lake.c
 *
 * Models pebbles on a lake
 * Description:
 *
 * This program uses centered finite differencing to
 * solve the wave equation with sources.
 *
 * The interface is given as
 *
 *   lake [grid_size] [# of pebbles] [end time] [# threads]
 *
 * where
 *
 *   grid_size -     integer, size of one edge of the square grid;
 *               so the true size of the computational grid will
 *               be grid_size * grid_size
 *
 *   # of pebbles -  number of simulated "pebbles" to start with
 *
 *   end time -  the simulation starts from t=0.0 and goes to
 *           t=[end time]
 *
 *   # threads -     the number of threads the simulation uses
 *
 **************************************/
#ifdef _OPENMP
#include <omp.h>
#define DATA_INIT_FILE "lake_i_omp.dat"
#define DATA_FINAL_FILE "lake_f_omp.dat"
#endif
#ifdef _OPENACC
#include <openacc.h>
#define DATA_INIT_FILE "lake_i_acc.dat"
#define DATA_FINAL_FILE "lake_f_acc.dat"
#endif
#ifndef DATA_INIT_FILE
#define DATA_INIT_FILE "lake_i.dat"
#endif
#ifndef DATA_FINAL_FILE
#define DATA_FINAL_FILE "lake_f.dat"
#endif
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "./lake.h"
#include "./lake_util.h"

/* Probably not necessary but doesn't hurt */
#define _USE_MATH_DEFINES
#define GRID_SIZE ((n + 4) * (n + 4))

/* Number of OpenMP threads */
int nthreads;

int main(int argc, char *argv[]) {

  if (argc != 5) {
    fprintf(stdout, "Usage: %s npoints npebs time_finish nthreads \n", argv[0]);
    return 0;
  }

  /* grab the arguments and setup some vars */
  int npoints = atoi(argv[1]);
  int npebs = atoi(argv[2]);
  double end_time = (double)atof(argv[3]);
  nthreads = atoi(argv[4]);
  int narea = npoints * npoints;

  /* check input params for resitrictions */
  if (npoints % nthreads != 0) {
    fprintf(stderr,
            "BONK! npoints must be evenly divisible by nthreads\n Try again!");
    return 0;
  }

  /* get the program directory */
  set_wrkdir(argv[0]);
  /* main simulation arrays */
  double *u_i0, *u_i1;
  double *u_cpu, *pebs;

  /* u_err is used when calculating the
   * error between one version of the code
   * and another. */
  /* double *u_err; */

  /* h is the size of each grid cell */
  double h;
  /* used for error analysis */
  /* double avgerr; */

  /* used for time analysis */
  double elapsed_cpu, elapsed_init;
  struct timeval cpu_start, cpu_end;

  /* allocate arrays */
  u_i0 = (double *)malloc(sizeof(double) * narea);
  u_i1 = (double *)malloc(sizeof(double) * narea);
  pebs = (double *)malloc(sizeof(double) * narea);

  u_cpu = (double *)malloc(sizeof(double) * narea);

  start_lake_log("lake.log");

  lake_log("running %s with (%d x %d) grid, until %f, with %d threads\n",
           argv[0], npoints, npoints, end_time, nthreads);

  /* initialize the simulation */
  h = (XMAX - XMIN) / npoints;

#ifdef __DEBUG
  lake_log("grid step size is %f\n", h);
  lake_log("initializing pebbles\n");
#endif

  init_pebbles(pebs, npebs, npoints);

#ifdef __DEBUG
  lake_log("initializing u0, u1\n");
#endif

  gettimeofday(&cpu_start, NULL);
  init(u_i0, pebs, npoints);
  init(u_i1, pebs, npoints);
  gettimeofday(&cpu_end, NULL);
  elapsed_init = ((cpu_end.tv_sec + cpu_end.tv_usec * 1e-6) -
                  (cpu_start.tv_sec + cpu_start.tv_usec * 1e-6));
  lake_log("Initialization took %f seconds\n", elapsed_init);

  /* print the initial configuration */
#ifdef __DEBUG
  lake_log("printing initial configuration file\n");
#endif

  print_heatmap(DATA_INIT_FILE, u_i0, npoints, h);

  /* time, run the simulation */
#ifdef __DEBUG
  lake_log("beginning simulation\n");
#endif

// **P3** Copy in non-simulation variables before.
#ifdef _OPENACC
#pragma acc enter data copyin(u_i0[ : npoints * npoints],                      \
                              u_i1[ : npoints * npoints],                      \
                              pebs[ : npoints * npoints])
#endif

  gettimeofday(&cpu_start, NULL);
  run_sim(u_cpu, u_i0, u_i1, pebs, npoints, h, end_time);
  gettimeofday(&cpu_end, NULL);
  elapsed_cpu = ((cpu_end.tv_sec + cpu_end.tv_usec * 1e-6) -
                 (cpu_start.tv_sec + cpu_start.tv_usec * 1e-6));
  lake_log("Simulation took %f seconds\n", elapsed_cpu);
  lake_log("Init+Simulation took %f seconds\n", elapsed_init + elapsed_cpu);

  /* print the final configuration */
#ifdef __DEBUG
  lake_log("printing final configuration file\n");
#endif

  print_heatmap(DATA_FINAL_FILE, u_cpu, npoints, h);

#ifdef __DEBUG
  lake_log("freeing memory\n");
#endif

  /* free memory */
  free(u_i0);
  free(u_i1);
  free(pebs);
  free(u_cpu);

  stop_lake_log();
  return 0;
}

/*****************************
 * run_sim
 *
 * Input
 * ----------
 *   double *u0 - the inital configuation
 *   double *u1 - the intial + 1 configuration
 *   double *pebbles - the array of pebbles
 *   int n - the grid size
 *   double h - the grid step size
 *   double end_time - the final time
 *
 * Output
 * ----------
 *   double *u - the final configuration
 *
 * Description
 * ----------
 *   run_sim is the main driver of the program.  It takes in the inital
 * configuration and parameters, and runs them until end_time is reached.
 *
 *******************************/
void run_sim(double *u, double *u0, double *u1, double *pebbles, int n,
             double h, double end_time) {

  double *un, *uc, *uo;

  /* time vars */
  double t, dt;
  int i, j, idx, idx_p;

#ifndef _OPENACC
  /* arrays used in the calculation */
  // **V0** Allocate simlation arrays leaving room for boundaries.
  un = (double *)malloc(sizeof(double) * GRID_SIZE);
  uc = (double *)malloc(sizeof(double) * GRID_SIZE);
  uo = (double *)malloc(sizeof(double) * GRID_SIZE);

  // **V0** Memset Simulation Arrays to 0, for east and west.
  memset(uo, 0, sizeof(double) * GRID_SIZE);
  memset(uc, 0, sizeof(double) * GRID_SIZE);
  memset(un, 0, sizeof(double) * GRID_SIZE);

// **V0** Copy initial data into simulation arrays.
// **V2** Parallelize memcpys - The compiler would skip this pragma
// each time. I don't believe it deemed it a useful opt, so commenting out.
#ifdef _OPENMP
#pragma omp parallel for private(i) num_threads(nthreads) shared(u0, uo, u1, uc)
#endif
  for (i = 0; i < n; i++) {
    memcpy(uo + (n + 4) * (i + 2) + 2, u0 + n * i, sizeof(double) * n);
    memcpy(uc + (n + 4) * (i + 2) + 2, u1 + n * i, sizeof(double) * n);
  }

  // **P3** Acc Setup
#else
  struct timeval acc_setup_start, acc_setup_end;
  gettimeofday(&acc_setup_start, NULL);
  uo = acc_malloc(sizeof(double) * GRID_SIZE);
  uc = acc_malloc(sizeof(double) * GRID_SIZE);
  un = acc_malloc(sizeof(double) * GRID_SIZE);

  // **P3** Set up simuation arrays from init arrays already on device
#pragma acc parallel loop present(u0, u1) deviceptr(uo, uc) vector_length(32)
  for (i = 0; i < n; i++) {
    memcpy(uo + (n + 4) * (i + 2) + 2, u0 + n * i, sizeof(double) * n);
    memcpy(uc + (n + 4) * (i + 2) + 2, u1 + n * i, sizeof(double) * n);
  }
  gettimeofday(&acc_setup_end, NULL);
  double elapsed_acc_setup =
      ((acc_setup_end.tv_sec + acc_setup_end.tv_usec * 1e-6) -
       (acc_setup_start.tv_sec + acc_setup_start.tv_usec * 1e-6));
  lake_log("Acc setup took: %f seconds\n", elapsed_acc_setup);
#endif

  /* start at t=0.0 */
  t = 0.;
  /* this is probably not ideal.  In principal, we should
   * keep the time-step at the size determined by the
   * CFL condition
   *
   * dt = h / vel_max
   *
   * where vel_max is the maximum velocity in the current
   * model.  The condition dt = h/2. should suffice, but
   * be aware the possibility exists for madness and mayhem */
  dt = h / 2.;

  const int pad = n + 4;
  struct timeval core_start, core_end;
  gettimeofday(&core_start, NULL);
  while (tpdt(&t, dt, end_time)) {

    // **V0** Copy north and south boundaries for current into position.
#ifndef _OPENACC
    memcpy(uc, uc + (n * (n + 4)), sizeof(double) * (n + 4) * 2);
    memcpy(uc + ((n + 4) * (n + 2)), uc + ((n + 4) * 2),
           sizeof(double) * (n + 4) * 2);
    //**P3** Device instead
#else
#define QUEUE_ID 1
    acc_memcpy_to_device(uc, uc + (n * (n + 4)), sizeof(double) * (n + 4) * 2);
    acc_memcpy_to_device(uc + ((n + 4) * (n + 2)), uc + ((n + 4) * 2),
                         sizeof(double) * (n + 4) * 2);
// **P3** OpenACC parallelization for outer loop
#pragma acc kernels loop deviceptr(uo, uc, un) present(pebbles)                \
    private(idx, idx_p) // async(QUEUE_ID)
#endif

    // **V1** Parallelize both loops.
#ifdef _OPENMP
#pragma omp parallel for private(i) shared(un, uc, uo, n, pebbles)             \
    num_threads(nthreads) /* **V3** schedule(dynamic, 64)*/
#endif
    for (i = 2; i < n + 2; i++) {
#ifdef _OPENMP
#pragma omp parallel for private(j, idx) shared(un, uc, uo, n, pebbles)        \
    num_threads(nthreads) /* **V3** schedule(dynamic, 64)*/
#endif
      for (j = 2; j < n + 2; j++) {
        idx = (i * pad) + j;

        // **V0** Get index for pebble.
        idx_p = (i - 2) * n + (j - 2);

        /* otherwise do the FD scheme */
        un[idx] =
            2 * uc[idx] - uo[idx] +
            VSQR * (dt * dt) *
                ((1 * (uc[idx - 1] + uc[idx + 1] + uc[idx + pad] +
                       uc[idx - pad]) + // 1st degree cardinals
                  0.25 * (uc[idx + pad - 1] + uc[idx + pad + 1] +
                          uc[idx - pad - 1] +
                          uc[idx - pad + 1]) + // 1st degree ordinals
                  0.125 * (uc[idx - 2] + uc[idx + 2] + uc[idx + pad + pad] +
                           uc[idx - pad - pad]) - // 2nd degree cardinals
                  5.5 * uc[idx]) /
                     (h * h) + // normalization
                 f(pebbles[idx_p], t));
      }
    }
    // **V1** Memcpy removal optimization through pointer swapping.
    double *temp = uo;
    uo = uc;
    uc = un;
    un = temp;
  }
  gettimeofday(&core_end, NULL);
  double elapsed_core = ((core_end.tv_sec + core_end.tv_usec * 1e-6) -
                         (core_start.tv_sec + core_start.tv_usec * 1e-6));
  lake_log("Core iteration took: %f seconds\n", elapsed_core);

  // **V0** Iterate and copy back into target array.
  // **V1** Copy out uc as it points to the current last iteration,
  // 		un would point to uo
#ifndef _OPENACC
  for (i = 0; i < n; i++) {
    memcpy(u + (n * i), uc + (n + 4) * (i + 2) + 2, sizeof(double) * n);
    // **P3** copy out from device
  }

  free(un);
  free(uc);
  free(uo);
#else
  for (i = 0; i < n; i++) {
    acc_memcpy_from_device(u + (n * i), uc + (n + 4) * (i + 2) + 2,
                           sizeof(double) * n);
  }
  acc_free(uc);
  acc_free(uo);
  acc_free(un);
#endif
}

/*****************************
 * init_pebbles
 *
 * Input
 * ----------
 *   int pn - the number of pebbles
 *   int n - the grid size
 *
 * Output
 * ----------
 *   double *p - an array (dimensioned same as the grid) that
 *           gives the inital pebble size.
 *
 * Description
 * ----------
 *   init_pebbles creates a random scattering of some pn pebbles,
 * along with a random size.  The range of the can be adjusted by changing
 * the constant MAX_PSZ.
 *
 *******************************/

void init_pebbles(double *p, int pn, int n) {
  int i, j, k, idx;
  int sz;

  srand(10);
  /* set to zero */
  memset(p, 0, sizeof(double) * n * n);

  for (k = 0; k < pn; k++) {
    /* the offset is to ensure that no pebbles
     * are spawned on the very edge of the grid */
    i = rand() % (n - 4) + 2;
    j = rand() % (n - 4) + 2;
    sz = rand() % MAX_PSZ;
    idx = j + i * n;
    p[idx] = (double)sz;
  }
}

/*****************************
 * f
 *
 * Input
 * ----------
 *   double p -  the inital pebble value
 *   double t -  the current time
 * Returns
 * ----------
 *   the value of the "pebble" source term at time t
 *
 * Description
 * ----------
 *   Each pebbles influance on the surface will "fade" as
 *   time marches forward (they may sink away, for instance).
 *   This function models that - at large t ("large" defined
 *   relative to the constant TSCALE) the pebble will have
 *   little to no effect.
 *
 *   NB: this function can be updated to model whatever behavior
 *   you wish the pebbles to have - they could continually jump
 *   up and down on the surface, driving more energic waves, for
 *   example.
 ******************************/
double f(double p, double t) { return -expf(-TSCALE * t) * p; }

int tpdt(double *t, double dt, double tf) {
  if ((*t) + dt > tf)
    return 0;
  (*t) = (*t) + dt;
  return 1;
}

void init(double *u, double *pebbles, int n) {
  int i, j, idx;

  // **V2** Parallelize Init function with both loops
#ifdef _OPENMP
#pragma omp parallel for private(i) shared(u, n, pebbles)                      \
    num_threads(nthreads) /* **V3** schedule(dynamic, 64)*/
#endif
  for (i = 0; i < n; i++) {
#ifdef _OPENMP
#pragma omp parallel for private(j, idx) shared(u, n, pebbles)                 \
    num_threads(nthreads) /* **V3** schedule(dynamic, 64)*/
#endif
    for (j = 0; j < n; j++) {
      idx = j + i * n;
      u[idx] = f(pebbles[idx], 0.0);
    }
  }
}

/*****************************
 * error_u
 *
 * Input
 * ----------
 *   double *ua  -   error 1
 *   double *ub  -   error 2
 *   int n       -   array extent
 *
 * Output
 * ----------
 *   double *uerr - array of errors
 *       double *avgerr - pointer to the average error
 *
 * Description
 * ----------
 *   Calculates the relative error between ua and ub
 *
 ********************************/
void error_u(double *uerr, double *avgerr, double *ua, double *ub, int n) {
  int i, j, idx;

  (*avgerr) = 0.;

  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      idx = j + i * n;
      uerr[idx] = fabs((ua[idx] - ub[idx]) / ua[idx]);
      (*avgerr) = (*avgerr) * ((double)idx / (double)(idx + 1)) +
                  uerr[idx] / (double)(idx + 1);
    }
  }
}

/*****************************
 * print_heatmap
 *
 * Input
 * ----------
 *   char *filename  - the output file name
 *   double *u       - the array to output
 *   int n           - the edge extent of u (ie, u is (n x n)
 *   double h        - the step size in u
 * Output
 * ----------
 *   None
 *
 * Description
 * ----------
 *   Outputs the array u to the file filename
 ********************************/
void print_heatmap(char *filename, double *u, int n, double h) {
  char full_filename[64];
  int i, j, idx;

  dir_string(filename, full_filename);
  FILE *fp = fopen(full_filename, "w");

  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      idx = j + i * n;
      fprintf(fp, "%f %f %f\n", i * h, j * h, u[idx]);
    }
  }

  fclose(fp);
}
