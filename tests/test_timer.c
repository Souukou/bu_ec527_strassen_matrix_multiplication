#include "timer.h"
#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <unistd.h>

void test_wakeup_delay() {
  double delay = wakeup_delay();
  printf("Wakeup Delay: %f\n", delay);
  assert(delay > 0);
  return;
}

void test_interval_1() {
  struct timespec start, stop;
  start.tv_sec = 0;
  start.tv_nsec = 0;
  stop.tv_sec = 0;
  stop.tv_nsec = 0;

  double time = interval(start, stop);
  printf("Time: %f\n", time);

  assert(time == 0);

  return;
}

void test_interval_2() {
  struct timespec start, stop;
  clock_gettime(CLOCK_REALTIME, &start);
  sleep(1);
  clock_gettime(CLOCK_REALTIME, &stop);
  double time = interval(start, stop);
  printf("Time: %f\n", time);

  assert(time > 1);

  return;
}

int main() {
  test_wakeup_delay();
  test_interval_1();
  test_interval_2();
  printf("All test_timer passed!\n");
  return 0;
}