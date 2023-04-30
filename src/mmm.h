#include "matrix.h"

void mmm_ijk(matrix_ptr a, matrix_ptr b, matrix_ptr c);
void mmm_ijk_omp(matrix_ptr a, matrix_ptr b, matrix_ptr c);
void mmm_kij(matrix_ptr a, matrix_ptr b, matrix_ptr c);
void mmm_kij_omp(matrix_ptr a, matrix_ptr b, matrix_ptr c);
void mmm_ijk_block_omp(matrix_ptr a, matrix_ptr b, matrix_ptr c, int bsize);
void mmm_kij_block_omp(matrix_ptr a, matrix_ptr b, matrix_ptr c, int bsize);