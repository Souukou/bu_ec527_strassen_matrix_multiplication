#ifndef MATRIX_H
#define MATRIX_H

#include <stdbool.h>
#define IDENT 0
typedef float data_t;

/* Create abstract data type for matrix */
typedef struct {
  long int rowlen;
  data_t *data;
} matrix_rec, *matrix_ptr;

matrix_ptr new_matrix(long int rowlen);
void free_matrix(matrix_ptr *m);
int set_matrix_rowlen(matrix_ptr m, int rowlen);
long int get_matrix_rowlen(matrix_ptr m);
int init_matrix(matrix_ptr m);
int zero_matrix(matrix_ptr m);
int rand_matrix(matrix_ptr m, int max, int min);
data_t *get_matrix_start(matrix_ptr m);
bool equal_matrix(matrix_ptr a, matrix_ptr b);
bool equal_matrix_tol(matrix_ptr a, matrix_ptr b, double tol);
matrix_ptr add_matrix(matrix_ptr a, matrix_ptr b, matrix_ptr c);
matrix_ptr sub_matrix(matrix_ptr a, matrix_ptr b, matrix_ptr c);

#endif