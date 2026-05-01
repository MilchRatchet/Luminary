#ifndef CU_LUMINARY_RUSSIAN_ROULETTE_H
#define CU_LUMINARY_RUSSIAN_ROULETTE_H

#include "math.cuh"
#include "random.cuh"
#include "utils.cuh"

#define RUSSIAN_ROULETTE_CLAMP (1.0f / 8.0f)

LUMINARY_FUNCTION bool russian_roulette_apply(const PathID path_id, const uint8_t state, RGBF& record) {
  if (state & STATE_FLAG_DELTA_PATH)
    return true;

  bool accepted = true;

  const float value = color_importance(record);

  const float threshold = device.settings.russian_roulette_threshold;

  // Inf and NaN are handled in the temporal accumulation.
  if (value < threshold) {
    // Clamp probability to avoid fireflies. Always remove paths that carry no light at all.
    const float p = (value > 0.0f) ? fmaxf(value / threshold, RUSSIAN_ROULETTE_CLAMP) : 0.0f;
    if (random_1D(RANDOM_TARGET_RUSSIAN_ROULETTE, path_id) > p) {
      accepted = false;
    }
    else {
      record = scale_color(record, 1.0f / p);
    }
  }

  return accepted;
}

#endif /* CU_LUMINARY_RUSSIAN_ROULETTE_H */
