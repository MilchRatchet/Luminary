#include "bench.h"

#include <stdint.h>
#include <stdio.h>

#include "log.h"

// windows.h must be included before any other windows header
#ifdef _WIN32
#include <windows.h>
#endif

#ifdef _WIN32
#include <profileapi.h>
#else
#include <time.h>
#endif

static int bench_active = 0;
static uint64_t t       = 0;

void bench_activate() {
  bench_active = 1;
}

static uint64_t get_time() {
#ifdef _WIN32
  LARGE_INTEGER ticks;
  if (!QueryPerformanceCounter(&ticks))
    return 0;
  return ticks.QuadPart;
#else
  return (uint64_t) clock();
#endif
}

static double get_diff(const uint64_t t0, const uint64_t t1) {
#ifdef _WIN32
  LARGE_INTEGER freq;
  if (!QueryPerformanceFrequency(&freq))
    return 0.0;

  const double t_freq = ((double) freq.QuadPart);

  return (t1 - t0) / t_freq;
#else
  return ((double) (t1 - t0)) / (CLOCKS_PER_SEC);
#endif
}

void bench_tic() {
  if (!bench_active)
    return;

  t = get_time();
}

void bench_toc(char* text) {
  if (!bench_active)
    return;

  print_info("%-32s [%.3fs]", text, get_diff(t, get_time()));
}
