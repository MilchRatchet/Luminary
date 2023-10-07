#include "bench.h"

#include <math.h>
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
static const char* str  = (const char*) 0;

void bench_activate(void) {
  bench_active = 1;
}

static uint64_t _bench_get_time(void) {
#ifdef _WIN32
  LARGE_INTEGER ticks;
  if (!QueryPerformanceCounter(&ticks))
    return 0;
  return ticks.QuadPart;
#else
  return (uint64_t) clock();
#endif
}

static double _bench_get_time_diff(const uint64_t t0, const uint64_t t1) {
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

void bench_tic(const char* text) {
  if (!bench_active)
    return;

  if (!text) {
    error_message("Text was NULL.");
    return;
  }

  t   = _bench_get_time();
  str = text;

  print_info_inline("%-32s [......]", str);
}

void bench_toc(void) {
  if (!bench_active)
    return;

  if (!str) {
    error_message("Str was NULL.");
    return;
  }

  print_info("%-32s [%.3fs]", str, _bench_get_time_diff(t, _bench_get_time()));
  str = (const char*) 0;
}
