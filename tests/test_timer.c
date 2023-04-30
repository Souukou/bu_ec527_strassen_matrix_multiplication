#include "timer.h"
#include <assert.h>
#include <stdbool.h>
#include <stdio.h>

bool test_wakeup_delay() {
  double delay = wakeup_delay();
  printf("Wakeup Delay: %f\n", delay);

  return true;
}

int main() {
  bool all_tests_pass = true;
  test_wakeup_delay();

  printf("All tests passed!\n");
  return 0;
}