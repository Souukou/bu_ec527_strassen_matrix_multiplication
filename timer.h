#include <time.h>

// use clock_gettime() from time.h to get the time
// int clock_gettime(clockid_t clk_id, struct timespec *tp);

double interval(struct timespec start, struct timespec end);

double wakeup_delay();