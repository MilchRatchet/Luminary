#ifndef CU_DIRECTIVES_H
#define CU_DIRECTIVES_H

#include "math.cuh"
#include "random.cuh"
#include "utils.cuh"

// TODO: It is time to move this to another file, directives is a name coming from some really ancient Luminary days.
#define RUSSIAN_ROULETTE_CLAMP (1.0f / 8.0f)

LUMINARY_FUNCTION bool task_russian_roulette(const DeviceTask task, const uint8_t state, RGBF& record) {
  if (state & STATE_FLAG_DELTA_PATH)
    return true;

  bool accepted = true;

  const float value = color_importance(record);

  // Inf and NaN are handled in the temporal accumulation.
  if (value < device.camera.russian_roulette_threshold) {
    // Clamp probability to avoid fireflies. Always remove paths that carry no light at all.
    const float p = (value > 0.0f) ? fmaxf(value / device.camera.russian_roulette_threshold, RUSSIAN_ROULETTE_CLAMP) : 0.0f;
    if (random_1D(RANDOM_TARGET_RUSSIAN_ROULETTE, task.index) > p) {
      accepted = false;
    }
    else {
      record = scale_color(record, 1.0f / p);
    }
  }

  return accepted;
}

#endif /* CU_DIRECTIVES_H */
