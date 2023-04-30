#ifndef MATRIX_H
#define MATRIX_H

#define IDENT 0
typedef float data_t;

/* Create abstract data type for matrix */
typedef struct {
  long int rowlen;
  data_t *data;
} matrix_rec, *matrix_ptr;

matrix_ptr new_matrix(long int rowlen);
void free_matrix(matrix_ptr m);
int set_matrix_rowlen(matrix_ptr m, long int rowlen);
long int get_matrix_rowlen(matrix_ptr m);
int init_matrix(matrix_ptr m, long int rowlen);
int zero_matrix(matrix_ptr m, long int rowlen);
data_t *get_matrix_start(matrix_ptr m);

#endif