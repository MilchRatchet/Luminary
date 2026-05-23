#ifndef CU_LUMINARY_FILM_H
#define CU_LUMINARY_FILM_H

#include "math.cuh"
#include "random.cuh"
#include "utils.cuh"

LUMINARY_FUNCTION float film_grain_layer_apply(
  const float value, uint32_t x, uint32_t y, uint32_t layer_id, const uint32_t film_grains_per_pixel) {
  if (device.camera.sensor.film_grain_strength == 0.0f)
    return value;

  const float activation_probability = 1.0f - expf(-value);

  const float random           = random_grain(x, y, layer_id);
  const float activated_grains = random_binomial_approx(film_grains_per_pixel, activation_probability, random);

  const float activation_fraction = __saturatef(((float) activated_grains) / film_grains_per_pixel);
  const float exposure            = copysignf(logf(fmaxf(1.0f - activation_fraction, 1e-12f)), 1.0f);

  return lerp(value, exposure, device.camera.sensor.film_grain_strength);
}

LUMINARY_FUNCTION RGBF film_grain_apply(RGBF color, uint32_t x, uint32_t y) {
  const float iso_factor = fmaxf(device.camera.sensor.iso, 1e-7f);

  const uint32_t film_grains_per_pixel = fmaxf(1.0f, 8192.0f * 100.0f / iso_factor);

  color = scale_color(color, device.camera.exposure_time * iso_factor);

  color.r = film_grain_layer_apply(color.r, x, y, 2, film_grains_per_pixel);
  color.g = film_grain_layer_apply(color.g, x, y, 1, film_grains_per_pixel);
  color.b = film_grain_layer_apply(color.b, x, y, 0, film_grains_per_pixel);

  return color;
}

#endif /* CU_LUMINARY_FILM_H */
