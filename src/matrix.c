#include "matrix.h"
#include <stdlib.h>
#include <stdio.h>

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

void free_matrix(matrix_ptr m) {
  free(m->data);
  free(m);
}

/* Set row length of matrix */
int set_matrix_rowlen(matrix_ptr m, long int rowlen) {
  m->rowlen = rowlen;
  return 1;
}

/* Return row length of matrix */
long int get_matrix_rowlen(matrix_ptr m) { return m->rowlen; }

/* initialize matrix */
int init_matrix(matrix_ptr m, long int rowlen) {
  long int i;

  if (rowlen > 0) {
    m->rowlen = rowlen;
    for (i = 0; i < rowlen * rowlen; i++)
      m->data[i] = (data_t)(i);
    return 1;
  } else
    return 0;
}

/* initialize matrix */
int zero_matrix(matrix_ptr m, long int rowlen) {
  long int i, j;

  if (rowlen > 0) {
    m->rowlen = rowlen;
    for (i = 0; i < rowlen * rowlen; i++) {
      m->data[i] = 0;
    }
    return 1;
  } else
    return 0;
}

data_t *get_matrix_start(matrix_ptr m) { return m->data; }