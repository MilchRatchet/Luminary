#ifndef CU_DIRECTIVES_H
#define CU_DIRECTIVES_H

#include "utils.cuh"

/*
 * Define WEIGHT_BASED_EXIT for the following to apply
 *
 * Rays with an accumulated weight below CUTOFF are cancelled
 * Higher values provide better performance at the cost of extreme bright lights not being shaded properly
 *
 * Rays with an accumulated weight below PROBABILISTIC_CUTOFF have a chance equal to
 * the inverse linear interpolation of the weight and CUTOFF and PROBABILISTIC_CUTOFF
 * Higher values provide much better performance at the cost of noisy/too dark areas
 * Setting PROBABILISTIC_CUTOFF >= CUTOFF will deactivate this option
 *
 * Tests show a significant performance increase with very minimal visual impact. The visuals
 * are perceptively unaffected.
 */
#define WEIGHT_BASED_EXIT
#define BRIGHTEST_EMISSION (device.scene.camera.exposure * device.scene.camera.russian_roulette_bias)
#define CUTOFF ((4.0f) / (BRIGHTEST_EMISSION))
#define PROBABILISTIC_CUTOFF (16.0f * CUTOFF)

__device__ int validate_trace_task(TraceTask task, RGBF& record) {
  int valid = 1;

#ifdef WEIGHT_BASED_EXIT
  const float max = luminance(record);
  if (max < CUTOFF) {
    valid = 0;
  }
  else if (max < PROBABILISTIC_CUTOFF) {
    const float p = (max - CUTOFF) / (PROBABILISTIC_CUTOFF - CUTOFF);
    if (white_noise() > p) {
      valid = 0;
    }
    else {
      record = scale_color(record, 1.0f / p);
    }
  }
#endif

  return valid;
}

#endif /* CU_DIRECTIVES_H */
