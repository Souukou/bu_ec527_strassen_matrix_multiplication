#include "matrix.h"
#include "strassen_simd.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>

void mmm_ground_truth(const matrix_ptr A, const matrix_ptr B, matrix_ptr C) {
  assert(A->rowlen == B->rowlen);
  assert(A->rowlen == C->rowlen);
  int rowlen = A->rowlen;
  for (int i = 0; i < rowlen; i++) {
    for (int j = 0; j < rowlen; j++) {
      double sum = 0;
      for (int k = 0; k < rowlen; k++) {
        sum += A->data[i * rowlen + k] * B->data[k * rowlen + j];
      }
      C->data[i * rowlen + j] = sum;
    }
  }
}

void test_strassen_simd() {
  matrix_ptr A = new_matrix(512);
  matrix_ptr B = new_matrix(512);
  matrix_ptr C = new_matrix(512);
  matrix_ptr C_gt = new_matrix(512);
  rand_matrix(A, 0, 10);
  rand_matrix(B, 0, 10);
  zero_matrix(C);
  zero_matrix(C_gt);

  mmm_ground_truth(A, B, C_gt);
  strassen_simd(A, B, C);

  double diff = 0;
  for (int i = 0; i < 256 * 256; i++) {
    if (fabs(C->data[i] - C_gt->data[i]) > fabs(C_gt->data[i] * 0.5)) {
      assert(false);
    }
    diff += fabs(C->data[i] - C_gt->data[i]) / fabs(C_gt->data[i]);
  }
  diff /= 512 * 512;
  printf("diff = %f\n", diff);
  assert(diff < 0.1);
  free_matrix(&A);
  free_matrix(&B);
  free_matrix(&C);
  free_matrix(&C_gt);
}

int main() {
  test_strassen_simd();
  printf("All test_strassen passed!\n");
  return 0;
}