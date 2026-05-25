#ifndef CU_LUMINARY_FILM_H
#define CU_LUMINARY_FILM_H

#include "math.cuh"
#include "random.cuh"
#include "utils.cuh"

LUMINARY_FUNCTION float film_grain_layer_apply(const float value, const float random, const uint32_t film_grains_per_pixel) {
  if (device.camera.sensor.film_grain_strength == 0.0f)
    return value;

  const float activation_probability = 1.0f - expf(-value);

  const float activated_grains = random_binomial_approx(film_grains_per_pixel, activation_probability, random);

  const float activation_fraction = ((float) activated_grains) / film_grains_per_pixel;
  const float exposure            = -logf(fmaxf(1.0f - activation_fraction, 1e-12f));

  return lerp(value, exposure, device.camera.sensor.film_grain_strength);
}

LUMINARY_FUNCTION RGBF film_grain_apply(RGBF color, uint32_t x, uint32_t y) {
  const float iso_factor = fmaxf(device.camera.sensor.iso, 1e-7f);

  // High quality film typically has tens of thousands of grains per digital pixel area at base ISO
  const uint32_t film_grains_per_pixel = fmaxf(1.0f, 65536.0f * 100.0f / iso_factor);

  color = scale_color(color, device.camera.exposure_time * iso_factor);

  if (device.camera.sensor.film_grain_strength == 0.0f)
    return color;

  // Real film grain clump sizes scale with film speed.
  // At low and medium ISOs (<= 400), grains are roughly digital-pixel-scale (size 1.0)
  const float grain_size = fmaxf(1.0f, sqrtf(iso_factor / 400.0f));
  const float fx         = x / grain_size;
  const float fy         = y / grain_size;

  const float random_lum = random_grain_smooth(fx, fy, 3);
  const float random_r   = random_grain_smooth(fx, fy, 2);
  const float random_g   = random_grain_smooth(fx, fy, 1);
  const float random_b   = random_grain_smooth(fx, fy, 0);

  const float lum              = color_luminance(color);
  const float lum_grain        = film_grain_layer_apply(lum, random_lum, film_grains_per_pixel);
  const float grain_multiplier = (lum > 0.0f) ? (lum_grain / lum) : 1.0f;

  const float r_grain = film_grain_layer_apply(color.r, random_r, film_grains_per_pixel);
  const float g_grain = film_grain_layer_apply(color.g, random_g, film_grains_per_pixel);
  const float b_grain = film_grain_layer_apply(color.b, random_b, film_grains_per_pixel);

  const float chroma_weight = 0.3f;
  color.r                   = lerp(color.r * grain_multiplier, r_grain, chroma_weight);
  color.g                   = lerp(color.g * grain_multiplier, g_grain, chroma_weight);
  color.b                   = lerp(color.b * grain_multiplier, b_grain, chroma_weight);

  return color;
}

#endif /* CU_LUMINARY_FILM_H */
