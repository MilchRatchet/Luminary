#ifndef CU_LUMINARY_FILM_H
#define CU_LUMINARY_FILM_H

#include "math.cuh"
#include "random.cuh"
#include "utils.cuh"

LUMINARY_FUNCTION float film_grain_layer_apply(const float value, uint32_t x, uint32_t y, uint32_t layer_id) {
  const float activation_probability = 1.0f - expf(-device.camera.sensor.film_grain_sensitity * value);

  const uint32_t film_grains_per_pixel = device.camera.sensor.film_grains_per_pixel;

  const float random              = random_grain(x, y, layer_id);
  const uint32_t activated_grains = random_binomial_approx(film_grains_per_pixel, activation_probability, random);

  const float activation_fraction = __saturatef(((float) activated_grains) / film_grains_per_pixel);

  // --- 1. Convert grain ratio to exposure proxy ---
  // Invert the activation model: f = 1 - exp(-alpha * E)
  // → E ≈ -ln(1 - f)
  const float exposure = -logf(fmaxf(1.0f - activation_fraction, 1e-6f));

  return lerp(value, exposure, device.camera.sensor.film_grain_strength);
}

LUMINARY_FUNCTION RGBF film_grain_apply(RGBF color, uint32_t x, uint32_t y) {
  if (device.camera.sensor.film_grain_strength == 0.0f)
    return color;

  color.r = film_grain_layer_apply(color.r, x, y, 2);
  color.g = film_grain_layer_apply(color.g, x, y, 1);
  color.b = film_grain_layer_apply(color.b, x, y, 0);

  return color;
}

#endif /* CU_LUMINARY_FILM_H */
