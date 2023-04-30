/***********************************************************************

 gcc -O1 -fopenmp matrix.c test_mmm.c -lrt -o test_mmm && OMP_NUM_THREADS=4
 ./test_mmm

*/

#include <stdio.h>
#include <stdlib.h>

#include <math.h>
#include <omp.h>
#include "matrix.h"
#include "mmm.h"
#include "timer.h"

/* We do *not* use CPNS (cycles per nanosecond) because when multiple
   cores are each executing with their own clock speeds, sometimes overlapping
   in time, measuring "how many cycles" a program takes does not reflect
   how much time it takes. We care about time more than about cycles. */

#define A 16 /* coefficient of x^2 */
#define B 8  /* coefficient of x */
#define C 80 /* constant term */

#define NUM_TESTS 10

#define OPTIONS 6

/* This define is only used if you do not set the environment variable
   OMP_NUM_THREADS as instructed above, and if OpenMP also does not
   automatically detect the hardware capabilities.

   If you have a machine with lots of cores, you may wish to test with
   more threads, but make sure you also include results for THREADS=4
   in your report. */
#define THREADS 4

void detect_threads_setting() {
  long int i, ognt;
  char *env_ONT;

/* Find out how many threads OpenMP thinks it is wants to use */
#pragma omp parallel for
  for (i = 0; i < 1; i++) {
    ognt = omp_get_num_threads();
  }

  printf("omp's default number of threads is %d\n", ognt);

  /* If this is illegal (0 or less), default to the "#define THREADS"
     value that is defined above */
  if (ognt <= 0) {
    if (THREADS != ognt) {
      printf("Overriding with #define THREADS value %d\n", THREADS);
      ognt = THREADS;
    }
  }

  omp_set_num_threads(ognt);

/* Once again ask OpenMP how many threads it is going to use */
#pragma omp parallel for
  for (i = 0; i < 1; i++) {
    ognt = omp_get_num_threads();
  }
  printf("Using %d threads for OpenMP\n", ognt);
}

/************************************************************************/
int main(int argc, char *argv[]) {
  int OPTION;
  struct timespec time_start, time_stop;
  double time_stamp[OPTIONS][NUM_TESTS];
  double final_answer;
  long int x, n, i, j, alloc_size;

  printf("OpenMP Matrix Multiply\n");

  final_answer = wakeup_delay();
  detect_threads_setting();

  /* declare and initialize the matrix structures */
  x = NUM_TESTS - 1;
  alloc_size = A * x * x + B * x + C;
  matrix_ptr a0 = new_matrix(alloc_size);
  init_matrix(a0, alloc_size);
  matrix_ptr b0 = new_matrix(alloc_size);
  init_matrix(b0, alloc_size);
  matrix_ptr c0 = new_matrix(alloc_size);
  zero_matrix(c0, alloc_size);

  for (OPTION = 0; OPTION < OPTIONS; OPTION++) {
    printf("Doing OPTION=%d...\n", OPTION);
    for (x = 0; x < NUM_TESTS && (n = A * x * x + B * x + C, n <= alloc_size);
         x++) {
      set_matrix_rowlen(a0, n);
      set_matrix_rowlen(b0, n);
      set_matrix_rowlen(c0, n);
      clock_gettime(CLOCK_REALTIME, &time_start);
      switch (OPTION) {
      case 0:
        mmm_ijk(a0, b0, c0);
        break;
      case 1:
        mmm_ijk_omp(a0, b0, c0);
        break;
      case 2:
        mmm_ijk_block_omp(a0, b0, c0, 8);
        break;
      case 3:
        mmm_kij(a0, b0, c0);
        break;
      case 4:
        mmm_kij_omp(a0, b0, c0);
        break;
      case 5:
        mmm_kij_block_omp(a0, b0, c0, 8);
        break;
      default:
        break;
      }
      clock_gettime(CLOCK_REALTIME, &time_stop);
      time_stamp[OPTION][x] = interval(time_start, time_stop);
      printf("  iter %d done\r", x);
      fflush(stdout);
    }
    printf("\n");
  }

  printf("\nAll times are in seconds\n");
  printf("rowlen, ijk, ijk_omp, ijk_block_omp, kij, kij_omp, kij_block_omp\n");
  {
    int i, j;
    for (i = 0; i < x; i++) {
      printf("%4ld", A * i * i + B * i + C);
      for (j = 0; j < OPTIONS; j++) {
        printf(",%10.4g", time_stamp[j][i]);
      }
      printf("\n");
    }
  }
  printf("\n");
  printf("Initial delay was calculating: %g \n", final_answer);
} /* end main */
