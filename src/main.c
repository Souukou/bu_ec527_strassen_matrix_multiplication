/***********************************************************************
 * See the "README.md" file for build instructions on how to compile and run
 * this program. Though this file is taken from Lab 6, it has been modified a
 * lot and not recommended to build via gcc directly.
 *
 * If you insist on building directly using gcc, use below command:
 *  gcc -O1 -fopenmp  strassen.c main.c matrix.c mmm.c timer.c -lrt -o main
 * OMP_NUM_THREADS=4 ./main
 */

#include "matrix.h"
#include "mmm.h"
#include "strassen.h"
#include "timer.h"
#include <omp.h>
#include <stdio.h>
#include <time.h>

#define A 16 /* coefficient of x^2 */
#define B 8  /* coefficient of x */
#define C 80 /* constant term */

#define NUM_TESTS 10

#define OPTIONS 1

#define THREADS 4

void detect_threads_setting() {
  long int i;
  int ognt;
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
  long int n, i, j, alloc_size;
  int x;

  printf("OpenMP Matrix Multiply\n");

  final_answer = wakeup_delay();
  detect_threads_setting();

  /* declare and initialize the matrix structures */
  x = NUM_TESTS - 1;
  alloc_size = A * x * x + B * x + C;

  printf("Doing OPTION=%d...\n", OPTION);
  for (x = 0; x < NUM_TESTS && (n = A * x * x + B * x + C, n <= alloc_size);
       x++) {
    matrix_ptr a0 = new_matrix(n);
    init_matrix(a0);
    matrix_ptr b0 = new_matrix(n);
    init_matrix(b0);
    matrix_ptr c0 = new_matrix(n);
    zero_matrix(c0);
    clock_gettime(CLOCK_REALTIME, &time_start);

    strassen(a0, b0, c0);

    clock_gettime(CLOCK_REALTIME, &time_stop);
    time_stamp[OPTION][x] = interval(time_start, time_stop);
    printf("  iter %d done\r", x);
    fflush(stdout);
    free_matrix(&a0);
    free_matrix(&b0);
    free_matrix(&c0);
  }
  printf("\n");

  printf("\nAll times are in seconds\n");
  printf("rowlen, strassen\n");
  {
    int i, j;
    for (i = 0; i < x; i++) {
      printf("%4d", A * i * i + B * i + C);
      for (j = 0; j < OPTIONS; j++) {
        printf(",%10.4g", time_stamp[j][i]);
      }
      printf("\n");
    }
  }
  printf("\n");
  printf("Initial delay was calculating: %g \n", final_answer);
} /* end main */
