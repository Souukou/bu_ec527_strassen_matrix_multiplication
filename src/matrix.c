#include "matrix.h"
#include <immintrin.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

/* Create matrix of specified length */
matrix_ptr new_matrix(long int rowlen) {
  long int i;

  /* Allocate and declare header structure */
  matrix_ptr result = (matrix_ptr)malloc(sizeof(matrix_rec));
  if (!result)
    return NULL; /* Couldn't allocate storage */
  result->rowlen = rowlen;

  /* Allocate and declare array */
  if (rowlen > 0) {
    data_t *data = (data_t *)calloc(rowlen * rowlen, sizeof(data_t));
    if (!data) {
      free((void *)result);
      printf("COULD NOT ALLOCATE %ld BYTES STORAGE \n",
             rowlen * rowlen * sizeof(data_t));
      exit(-1);
    }
    result->data = data;
  } else
    result->data = NULL;

  return result;
}

void free_matrix(matrix_ptr *m) {
  free((*m)->data);
  free(*m);
  (*m) = NULL;
}

/* Set row length of matrix */
int set_matrix_rowlen(matrix_ptr m, int rowlen) {
  if (rowlen < 0)
    return 0;
  if (m->data)
    free(m->data);
  m->data = (data_t *)calloc(rowlen * rowlen, sizeof(data_t));
  m->rowlen = rowlen;
  return 1;
}

/* Return row length of matrix */
long int get_matrix_rowlen(matrix_ptr m) { return m->rowlen; }

/* initialize matrix */
int init_matrix(matrix_ptr m) {
  long int i;
  int rowlen = m->rowlen;

  if (rowlen > 0) {
    m->rowlen = rowlen;
    for (i = 0; i < rowlen * rowlen; i++)
      m->data[i] = (data_t)(i);
    return 1;
  } else
    return 0;
}

/* initialize matrix */
int zero_matrix(matrix_ptr m) {
  long int i, j;
  int rowlen = m->rowlen;

  if (rowlen > 0) {
    m->rowlen = rowlen;
    for (i = 0; i < rowlen * rowlen; i++) {
      m->data[i] = 0;
    }
    return 1;
  } else
    return 0;
}

double fRand(double fMin, double fMax) {
  double f = (double)random() / RAND_MAX;
  return fMin + f * (fMax - fMin);
}

int rand_matrix(matrix_ptr m, int max, int min) {
  long int i, j;
  int rowlen = m->rowlen;

  if (rowlen > 0) {
    m->rowlen = rowlen;
    for (i = 0; i < rowlen * rowlen; i++) {
      m->data[i] = (data_t)(fRand((double)(max), (double)(min)));
    }

    return 1;
  } else
    return 0;
}

bool equal_matrix(matrix_ptr a, matrix_ptr b) {
  return equal_matrix_tol(a, b, 0.0);
}

bool equal_matrix_tol(matrix_ptr a, matrix_ptr b, double tol) {
  if (a->rowlen != b->rowlen)
    return false;
  for (int i = 0; i < a->rowlen * a->rowlen; i++) {
    if (fabs(a->data[i] - b->data[i]) > tol)
      return false;
  }
  return true;
}

bool equal_matrix_percent(matrix_ptr a, matrix_ptr b, double percent) {
  if (a->rowlen != b->rowlen)
    return false;
  for (int i = 0; i < a->rowlen * a->rowlen; i++) {
    if (fabs(a->data[i] - b->data[i]) > fabs(a->data[i] * percent))
      return false;
  }
  return true;
}

data_t *get_matrix_start(matrix_ptr m) { return m->data; }

matrix_ptr add_matrix(matrix_ptr a, matrix_ptr b, matrix_ptr c) {
  int i, j;
  int row_length = get_matrix_rowlen(a);
  data_t *a0 = get_matrix_start(a);
  data_t *b0 = get_matrix_start(b);
  data_t *c0 = get_matrix_start(c);

  for (i = 0; i < row_length; i++) {
    for (j = 0; j < row_length; j++) {
      c0[i * row_length + j] = a0[i * row_length + j] + b0[i * row_length + j];
    }
  }
  return c;
}

matrix_ptr sub_matrix(matrix_ptr a, matrix_ptr b, matrix_ptr c) {
  int i, j;
  int row_length = get_matrix_rowlen(a);
  data_t *a0 = get_matrix_start(a);
  data_t *b0 = get_matrix_start(b);
  data_t *c0 = get_matrix_start(c);

  for (i = 0; i < row_length; i++) {
    for (j = 0; j < row_length; j++) {
      c0[i * row_length + j] = a0[i * row_length + j] - b0[i * row_length + j];
    }
  }
  return c;
}

matrix_ptr add_matrix_avx256(matrix_ptr a, matrix_ptr b, matrix_ptr c) {
  int i, j;
  int row_length = get_matrix_rowlen(a);
  row_length = row_length - (row_length % 8);
  data_t *a0 = get_matrix_start(a);
  data_t *b0 = get_matrix_start(b);
  data_t *c0 = get_matrix_start(c);

  for (i = 0; i < row_length; i++) {
    for (j = 0; j < row_length; j += 8) { // Process 8 elements at a time
      __m256 a_vec = _mm256_loadu_ps(&a0[i * row_length + j]);
      __m256 b_vec = _mm256_loadu_ps(&b0[i * row_length + j]);
      __m256 c_vec = _mm256_add_ps(a_vec, b_vec);
      _mm256_storeu_ps(&c0[i * row_length + j], c_vec);
    }
  }
  return c;
}

matrix_ptr sub_matrix_avx256(matrix_ptr a, matrix_ptr b, matrix_ptr c) {
  int i, j;
  int row_length = get_matrix_rowlen(a);
  row_length = row_length - (row_length % 8);
  data_t *a0 = get_matrix_start(a);
  data_t *b0 = get_matrix_start(b);
  data_t *c0 = get_matrix_start(c);

  for (i = 0; i < row_length; i++) {
    for (j = 0; j < row_length; j += 8) { // Process 8 elements at a time
      __m256 a_vec = _mm256_loadu_ps(&a0[i * row_length + j]);
      __m256 b_vec = _mm256_loadu_ps(&b0[i * row_length + j]);
      __m256 c_vec = _mm256_sub_ps(a_vec, b_vec);
      _mm256_storeu_ps(&c0[i * row_length + j], c_vec);
    }
  }
  return c;
}
