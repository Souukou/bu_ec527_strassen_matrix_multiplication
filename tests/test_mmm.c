#include "mmm.h"
#include "matrix.h"
#include <assert.h>
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

void test_mmm() {
  matrix_ptr A = new_matrix(256);
  matrix_ptr B = new_matrix(256);
  matrix_ptr C = new_matrix(256);
  matrix_ptr C_gt = new_matrix(256);
  rand_matrix(A, 0, 1);
  rand_matrix(B, 0, 1);
  zero_matrix(C);
  zero_matrix(C_gt);
  mmm_ground_truth(A, B, C_gt);
  mmm_kij_omp(A, B, C);
  assert(equal_matrix_percent(C, C_gt, 0.01));
  free_matrix(&A);
  free_matrix(&B);
  free_matrix(&C);
  free_matrix(&C_gt);
}

int main() {
  test_mmm();
  printf("All test_mmm passed!\n");
  return 0;
}