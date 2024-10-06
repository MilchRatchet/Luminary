#ifndef CU_DIRECTIVES_H
#define CU_DIRECTIVES_H

#include "math.cuh"
#include "random.cuh"
#include "utils.cuh"

/*
 * This is just russian roulette. Before I knew of this, I made this file and experimented with different
 * ways of culling less important rays. I can clean this up eventually.
 */
#define WEIGHT_BASED_EXIT
#define RUSSIAN_ROULETTE_CLAMP (1.0f / 8.0f)

__device__ int validate_trace_task(const TraceTask task, RGBF& record) {
  int valid = 1;

#ifdef WEIGHT_BASED_EXIT
  const float value = color_importance(record);

  // Inf and NaN are handled in the temporal accumulation.
  if (value < device.camera.russian_roulette_threshold) {
    // Clamp probability to avoid fireflies. Always remove paths that carry no light at all.
    const float p = (value > 0.0f) ? fmaxf(value / device.camera.russian_roulette_threshold, RUSSIAN_ROULETTE_CLAMP) : 0.0f;
    if (quasirandom_sequence_1D(QUASI_RANDOM_TARGET_RUSSIAN_ROULETTE, task.index) > p) {
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
