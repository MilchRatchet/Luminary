#ifndef FRAMETIME_H
#define FRAMETIME_H

#include <stdint.h>

/*
 * Cross-platform timer which measures frametimes and
 * computes its moving average
 */
struct Frametime {
  uint64_t last;
  double average;
  double variance;
} typedef Frametime;

Frametime init_frametime();
void sample_frametime(Frametime* frametime);
double get_frametime(Frametime* frametime);
double get_variance(Frametime* frametime);
double get_deviation(Frametime* frametime);

#endif /* FRAMETIME_H */
