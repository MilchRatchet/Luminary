#include "frametime.h"
#include <math.h>

#ifdef _WIN32
#include <windows.h>
#include <profileapi.h>
#else
#include <time.h>
#endif

#define FRAMETIME_WEIGHT 0.2

static uint64_t sample_time() {
#ifdef _WIN32
  LARGE_INTEGER ticks;
  if (!QueryPerformanceCounter(&ticks))
    return 0;
  return ticks.QuadPart;
#else
  return (uint64_t) clock();
#endif
}

static double time_diff(const uint64_t t0, const uint64_t t1) {
#ifdef _WIN32
  LARGE_INTEGER freq;
  if (!QueryPerformanceFrequency(&freq))
    return 0.0;

  const double t_freq = ((double) freq.QuadPart) / 1000.0;

  return (t1 - t0) / t_freq;
#else
  return 1000.0 * ((double) (t1 - t0)) / (CLOCKS_PER_SEC);
#endif
}

Frametime init_frametime() {
  Frametime frametime;
  frametime.last     = sample_time();
  frametime.average  = 0.0;
  frametime.variance = 0.0;

  return frametime;
}

void start_frametime(Frametime* frametime) {
  frametime->last = sample_time();
}

void sample_frametime(Frametime* frametime) {
  const uint64_t curr_time = sample_time();
  const double diff_time   = time_diff(frametime->last, curr_time);
  const double variance    = (frametime->average - diff_time) * (frametime->average - diff_time);
  frametime->last          = curr_time;
  frametime->average = diff_time * FRAMETIME_WEIGHT + frametime->average * (1.0 - FRAMETIME_WEIGHT);
  frametime->variance =
    variance * FRAMETIME_WEIGHT + frametime->variance * (1.0 - FRAMETIME_WEIGHT);
}

double get_frametime(Frametime* frametime) {
  return frametime->average;
}

double get_variance(Frametime* frametime) {
  return frametime->variance;
}

double get_deviation(Frametime* frametime) {
  return sqrt(frametime->variance);
}
