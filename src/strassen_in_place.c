#include "strassen_in_place.h"
#include "matrix.h"
#include "mmm.h"

matrix_ptr strassen_in_place_helper(matrix_ptr a, matrix_ptr b, matrix_ptr c, int a_row, int a_col, int b_row, int b_col, int c_row, int c_col, int length, matrix_ptr temp1, matrix_ptr temp2) {

  int length = get_matrix_rowlen(a);

  if (length < 256) {
    mmm_kij_block_omp_offset_avx256(a, b, c, a_row, a_col, b_row, b_col, c_row, c_col, 8, length);
    return;
  }

  int len = length / 2;

   // Use the strassen_in_place function instead of the original strassen function
  strassen_in_place_helper(a, b, temp1, a_row, a_col, b_row, b_col, 0, 0, len, temp2, c);
  strassen_in_place_helper(a, b, temp2, a_row, a_col + len, b_row + len, b_col, 0, 0, len, temp1, c);
  add_matrix_in_place(temp1, temp2, c, c_row, c_col, len);

  // Repeat for the other quadrants
  strassen_in_place_helper(a, b, temp1, a_row + len, a_col, b_row, b_col, 0, 0, len, temp2, c);
  strassen_in_place_helper(a, b, temp2, a_row + len, a_col + len, b_row + len, b_col, 0, 0, len, temp1, c);
  add_matrix_in_place(temp1, temp2, c, c_row + len, c_col, len);

  strassen_in_place_helper(a, b, temp1, a_row, a_col, b_row, b_col + len, 0, 0, len, temp2, c);
  strassen_in_place_helper(a, b, temp2, a_row, a_col + len, b_row + len, b_col + len, 0, 0, len, temp1, c);
  add_matrix_in_place(temp1, temp2, c, c_row, c_col + len, len);

  strassen_in_place_helper(a, b, temp1, a_row + len, a_col, b_row, b_col + len, 0, 0, len, temp2, c);
  strassen_in_place_helper(a, b, temp2, a_row + len, a_col + len, b_row + len, b_col + len, 0, 0, len, temp1, c);
  add_matrix_in_place(temp1, temp2, c, c_row + len, c_col + len, len);

}

matrix_ptr strassen_in_place(matrix_ptr a, matrix_ptr b, matrix_ptr c) {
  int length = get_matrix_rowlen(a);

  if (length < 256) {
    mmm_kij_block_omp(a, b, c, 8);
    return c;
  }

  int len = length / 2;

  matrix_ptr temp1 = new_matrix(len);
  matrix_ptr temp2 = new_matrix(len);

zero_matrix(temp1);
zero_matrix(temp2);

strassen_in_place_helper(a, b, c, 0, 0, 0, 0, 0, 0, length, temp1, temp2);

free_matrix(&temp1);
free_matrix(&temp2);

return c;
}
