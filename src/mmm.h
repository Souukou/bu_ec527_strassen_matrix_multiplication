#ifndef MMM_H
#define MMM_H

#include "matrix.h"

void mmm_ijk(matrix_ptr a, matrix_ptr b, matrix_ptr c);
void mmm_ijk_omp(matrix_ptr a, matrix_ptr b, matrix_ptr c);
void mmm_kij(matrix_ptr a, matrix_ptr b, matrix_ptr c);
void mmm_kij_omp(matrix_ptr a, matrix_ptr b, matrix_ptr c);
void mmm_ijk_block_omp(matrix_ptr a, matrix_ptr b, matrix_ptr c, int bsize);
void mmm_kij_block_omp(matrix_ptr a, matrix_ptr b, matrix_ptr c, int bsize);
void mmm_kij_block_omp_avx256(matrix_ptr a, matrix_ptr b, matrix_ptr c,
                              int bsize);
void mmm_kij_block_omp_offset_avx256(matrix_ptr a, matrix_ptr b, matrix_ptr c,
                                      int a_row, int a_col, int b_row, int b_col,
                                      int c_row, int c_col, int bsize,
                                      int length);

#endif