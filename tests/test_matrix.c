#include "matrix.h"
#include <assert.h>
#include <stdio.h>

void test_new_matrix() {
  matrix_ptr a0 = new_matrix(1000);

  assert(a0 != NULL);
  assert(a0->rowlen == 1000);
  assert(a0->data != NULL);
  
  free_matrix(&a0);

  assert(a0 == NULL);
}

void test_init_matrix(){
  matrix_ptr a0 = new_matrix(100);

  assert(a0 != NULL);
  assert(a0->rowlen == 100);  
  assert(a0->data != NULL);

  init_matrix(a0);

  for(int i = 0; i < 100*100; i++){
    assert(a0->data[i] == i);
  }
}

void test_zero_matrix(){
  matrix_ptr a0 = new_matrix(100);

  assert(a0 != NULL);
  assert(a0->rowlen == 100);  
  assert(a0->data != NULL);

  zero_matrix(a0);

  for(int i = 0; i < 100*100; i++){
    assert(a0->data[i] == 0);
  }
}

void test_get_matrix_start(){
  matrix_ptr a0 = new_matrix(100);

  assert(a0 != NULL);
  assert(a0->rowlen == 100);  
  assert(a0->data != NULL);

  init_matrix(a0);

  data_t *start = get_matrix_start(a0);

  for(int i = 0; i < 100*100; i++){
    assert(start[i] == i);
  }
}

void test_equal_matrix(){
  matrix_ptr a0 = new_matrix(100);
  matrix_ptr a1 = new_matrix(100);
  matrix_ptr a2 = new_matrix(100);
  init_matrix(a0);
  init_matrix(a1);
  zero_matrix(a2);
  assert(equal_matrix(a0, a1));
  assert(!equal_matrix(a0, a2));
}

void test_equal_matrix_tol(){
  matrix_ptr a0 = new_matrix(100);
  matrix_ptr a1 = new_matrix(100);
  init_matrix(a0);
  init_matrix(a1);
  a0->data[0] += 0.0005;
  assert(equal_matrix_tol(a0, a1, 0.001));
  assert(!equal_matrix_tol(a0, a1, 0.0001));
}

void test_equal_matrix_percent(){
  matrix_ptr a0 = new_matrix(100);
  matrix_ptr a1 = new_matrix(100);
  init_matrix(a0);
  init_matrix(a1);
  a0->data[0] += 0.05;
  assert(equal_matrix_tol(a0, a1, 0.055));
  assert(!equal_matrix_tol(a0, a1, 0.045));
}

int main() {
  test_new_matrix();
  test_init_matrix();
  test_zero_matrix();
  test_get_matrix_start();
  test_equal_matrix();
  test_equal_matrix_tol();
  test_equal_matrix_percent();
  printf("All test_matrix passed!\n");
  return 0;
}