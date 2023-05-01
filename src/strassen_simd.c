#include "strassen_simd.h"
#include "matrix.h"
#include "mmm.h"

matrix_ptr strassen_simd(matrix_ptr a, matrix_ptr b, matrix_ptr c) {

  int length = get_matrix_rowlen(a);

  data_t *a0 = get_matrix_start(a);
  data_t *b0 = get_matrix_start(b);
  data_t *c0 = get_matrix_start(c);
  data_t sum;

  if (length < 256) {
    mmm_kij_block_omp_avx256(a, b, c, 8);
    return c;
  }

  int len = length / 2;

  matrix_ptr a11 = new_matrix(len), a12 = new_matrix(len),
             a21 = new_matrix(len), a22 = new_matrix(len),
             b11 = new_matrix(len), b12 = new_matrix(len),
             b21 = new_matrix(len), b22 = new_matrix(len),
             c11 = new_matrix(len), c12 = new_matrix(len),
             c21 = new_matrix(len), c22 = new_matrix(len), m1 = new_matrix(len),
             m2 = new_matrix(len), m3 = new_matrix(len), m4 = new_matrix(len),
             m5 = new_matrix(len), m6 = new_matrix(len), m7 = new_matrix(len),
             temp1 = new_matrix(len), temp2 = new_matrix(len);

  zero_matrix(a11);
  zero_matrix(a12);
  zero_matrix(a21);
  zero_matrix(a22);

  zero_matrix(b11);
  zero_matrix(b12);
  zero_matrix(b21);
  zero_matrix(b22);

  zero_matrix(c11);
  zero_matrix(c12);
  zero_matrix(c21);
  zero_matrix(c22);

  zero_matrix(m1);
  zero_matrix(m2);
  zero_matrix(m3);
  zero_matrix(m4);
  zero_matrix(m5);
  zero_matrix(m6);
  zero_matrix(m7);

  zero_matrix(temp1);
  zero_matrix(temp2);

  for (int i = 0; i < len; i++) {
    for (int j = 0; j < len; j++) {
      a11->data[i * len + j] = a->data[i * len + j];
      a12->data[i * len + j] = a->data[i * len + j + len];
      a21->data[i * len + j] = a->data[(i + len) * len + j];
      a22->data[i * len + j] = a->data[(i + len) * len + j + len];

      b11->data[i * len + j] = b->data[i * len + j];
      b12->data[i * len + j] = b->data[i * len + j + len];
      b21->data[i * len + j] = b->data[(i + len) * len + j];
      b22->data[i * len + j] = b->data[(i + len) * len + j + len];
    }
  }

  strassen_simd(add_matrix_avx256(a11, a22, temp1),
                add_matrix_avx256(b11, b22, temp2), m1);
  strassen_simd(add_matrix_avx256(a21, a22, temp1), b11, m2);
  strassen_simd(a11, sub_matrix_avx256(b12, b22, temp1), m3);
  strassen_simd(a22, sub_matrix_avx256(b21, b11, temp1), m4);
  strassen_simd(add_matrix_avx256(a11, a12, temp1), b22, m5);
  strassen_simd(sub_matrix_avx256(a21, a11, temp1),
                add_matrix_avx256(b11, b12, temp2), m6);
  strassen_simd(sub_matrix_avx256(a12, a22, temp1),
                add_matrix_avx256(b21, b22, temp2), m7);

  sub_matrix_avx256(add_matrix_avx256(m1, m4, temp1),
                    sub_matrix_avx256(m5, m7, temp2), c11);
  add_matrix_avx256(m3, m5, c12);
  add_matrix_avx256(m2, m4, c21);
  add_matrix_avx256(sub_matrix_avx256(m1, m2, temp1),
                    add_matrix_avx256(m3, m6, temp2), c22);

  for (int i = 0; i < len; i++) {
    for (int j = 0; j < len; j++) {
      c->data[i * length + j] = c11->data[i * len + j];
      c->data[i * length + j + len] = c12->data[i * len + j];
      c->data[(i + len) * length + j] = c21->data[i * len + j];
      c->data[(i + len) * length + j + len] = c22->data[i * len + j];
    }
  }

  free_matrix(&a11);
  free_matrix(&a12);
  free_matrix(&a21);
  free_matrix(&a22);

  free_matrix(&b11);
  free_matrix(&b12);
  free_matrix(&b21);
  free_matrix(&b22);

  free_matrix(&c11);
  free_matrix(&c12);
  free_matrix(&c21);
  free_matrix(&c22);

  free_matrix(&m1);
  free_matrix(&m2);
  free_matrix(&m3);
  free_matrix(&m4);
  free_matrix(&m5);
  free_matrix(&m6);
  free_matrix(&m7);

  free_matrix(&temp1);
  free_matrix(&temp2);

  return c;
}
