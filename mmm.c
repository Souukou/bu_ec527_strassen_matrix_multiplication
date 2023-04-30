#include "mmm.h"
#include <stdio.h>
#include <stdlib.h>



/* MMM ijk */
void mmm_ijk(matrix_ptr a, matrix_ptr b, matrix_ptr c)
{
  long int i, j, k;
  long int row_length = get_matrix_rowlen(a);
  data_t *a0 = get_matrix_start(a);
  data_t *b0 = get_matrix_start(b);
  data_t *c0 = get_matrix_start(c);
  data_t sum;

  for (i = 0; i < row_length; i++) {
    for (j = 0; j < row_length; j++) {
      sum = 0;
      for (k = 0; k < row_length; k++)
        sum += a0[i*row_length+k] * b0[k*row_length+j];
      c0[i*row_length+j] += sum;
    }
  }
}

/* MMM ijk w/ OMP */
void mmm_ijk_omp(matrix_ptr a, matrix_ptr b, matrix_ptr c)
{
  long int i, j, k;
  long int row_length = get_matrix_rowlen(a);
  data_t *a0 = get_matrix_start(a);
  data_t *b0 = get_matrix_start(b);
  data_t *c0 = get_matrix_start(c);
  data_t sum;
  long int part_rows = row_length/4;

#pragma omp parallel shared(a0,b0,c0,row_length) private(i,j,k,sum)
  {
#pragma omp for
    for (i = 0; i < row_length; i++) {
      for (j = 0; j < row_length; j++) {
        sum = 0;
        for (k = 0; k < row_length; k++)
          sum += a0[i*row_length+k] * b0[k*row_length+j];
        c0[i*row_length+j] += sum;
      }
    }
  }
}

void mmm_ijk_block_omp(matrix_ptr a, matrix_ptr b, matrix_ptr c, int bsize)
{
  long int i, j, k, jj, kk;
  long int length = get_matrix_rowlen(a);
  int en = bsize*(length/bsize);

  data_t *a0 = get_matrix_start(a);
  data_t *b0 = get_matrix_start(b);
  data_t *c0 = get_matrix_start(c);
  data_t sum;

#pragma omp parallel for shared(a0,b0,c0,length,bsize,en) private(i,j,k,jj,kk,sum)
  for (jj = 0; jj < en; jj+=bsize) {
    for (kk = 0; kk < en; kk+=bsize) {
      for (i = 0; i < length; ++i){
          for (j = jj; j < jj+bsize; ++j){
              sum = IDENT;
              for (k = kk; k < kk+bsize; ++k){
                  sum += a0[i*length+k] * b0[k*length+j];
              }
              c0[i*length+j] += sum;
          }
      }
    }
  }
}

/* MMM kij */
void mmm_kij(matrix_ptr a, matrix_ptr b, matrix_ptr c)
{
  long int i, j, k;
  long int get_matrix_rowlen(matrix_ptr m);
  data_t *get_matrix_start(matrix_ptr m);
  long int row_length = get_matrix_rowlen(a);
  data_t *a0 = get_matrix_start(a);
  data_t *b0 = get_matrix_start(b);
  data_t *c0 = get_matrix_start(c);
  data_t r;

  for (k = 0; k < row_length; k++) {
    for (i = 0; i < row_length; i++) {
      r = a0[i*row_length+k];
      for (j = 0; j < row_length; j++)
        c0[i*row_length+j] += r*b0[k*row_length+j];
    }
  }
}

/* MMM kij w/ OMP */
void mmm_kij_omp(matrix_ptr a, matrix_ptr b, matrix_ptr c)
{
  long int i, j, k;
  long int get_matrix_rowlen(matrix_ptr m);
  data_t *get_matrix_start(matrix_ptr m);
  long int row_length = get_matrix_rowlen(a);
  data_t *a0 = get_matrix_start(a);
  data_t *b0 = get_matrix_start(b);
  data_t *c0 = get_matrix_start(c);
  data_t r;

#pragma omp parallel shared(a0,b0,c0,row_length) private(i,j,k,r)
  {
    #pragma omp for
    for (k = 0; k < row_length; k++) {
      for (i = 0; i < row_length; i++) {
        r = a0[i*row_length+k];
        for (j = 0; j < row_length; j++)
          c0[i*row_length+j] += r*b0[k*row_length+j];
      }
    }
  }
}


void mmm_kij_block_omp(matrix_ptr a, matrix_ptr b, matrix_ptr c, int bsize)
{
  long int i, j, k, jj, ii;
  long int length = get_matrix_rowlen(a);
  int en = bsize*(length/bsize);

  data_t *a0 = get_matrix_start(a);
  data_t *b0 = get_matrix_start(b);
  data_t *c0 = get_matrix_start(c);
  data_t sum;

  
#pragma omp parallel for shared(a0,b0,c0,length,bsize,en) private(i,j,k,jj,ii,sum)
  for (ii = 0; ii < en; ii+=bsize) {
    for (jj = 0; jj < en; jj+=bsize) {
      for (k = 0; k < length; k++) {
        for (i = ii; i < ii+bsize; ++i){
          sum = IDENT;
          for (j = jj; j < jj+bsize; ++j){
                  sum += a0[i*length+k] * b0[k*length+j];
              }
              c0[i*length+j] += sum;
          }
        }
    }
  }
}

