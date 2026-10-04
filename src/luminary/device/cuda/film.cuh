#ifndef CU_LUMINARY_FILM_H
#define CU_LUMINARY_FILM_H

#include "math.cuh"
#include "random.cuh"
#include "utils.cuh"

#define FILM_GRAIN_BASE_LOG_SIGMA 0.035f
#define FILM_GRAIN_MAX_LOG_SIGMA 0.5f
#define FILM_GRAIN_MIN_EXPOSURE 0.03f

LUMINARY_FUNCTION float film_grain_noise_multiplier(const float noise, const float log_sigma) {
  return expf(log_sigma * noise - 0.5f * log_sigma * log_sigma);
}

LUMINARY_FUNCTION RGBF
  film_grain_apply(RGBF color, const float film_x_mm, const float film_y_mm, const float pixel_width_mm, const float pixel_height_mm) {
  const float strength = fminf(fmaxf(device.camera.sensor.film_grain_strength, 0.0f), 1.0f);
  if (strength == 0.0f)
    return color;

  const float fx          = film_x_mm / device.camera.sensor.film_grain_size;
  const float fy          = film_y_mm / device.camera.sensor.film_grain_size;
  const float half_width  = 0.5f * pixel_width_mm / device.camera.sensor.film_grain_size;
  const float half_height = 0.5f * pixel_height_mm / device.camera.sensor.film_grain_size;

  const float4 grain_noise = random_grain_filtered(fx, fy, half_width, half_height);
  const float shared_noise = grain_noise.x;
  const float red_noise    = grain_noise.y;
  const float green_noise  = grain_noise.z;
  const float blue_noise   = grain_noise.w;

  const float luminance = fmaxf(color_luminance(color), 0.0f);
  const float amplitude = FILM_GRAIN_BASE_LOG_SIGMA * device.camera.sensor.film_grain_amplitude;

  float log_sigma = fminf(amplitude * rsqrtf(fmaxf(luminance, FILM_GRAIN_MIN_EXPOSURE)), FILM_GRAIN_MAX_LOG_SIGMA);
  log_sigma       = log_sigma * strength;

  const float shared_weight      = 0.6f;
  const float independent_weight = 0.4f;

  color.r *= film_grain_noise_multiplier(shared_weight * shared_noise + independent_weight * red_noise, log_sigma);
  color.g *= film_grain_noise_multiplier(shared_weight * shared_noise + independent_weight * green_noise, log_sigma);
  color.b *= film_grain_noise_multiplier(shared_weight * shared_noise + independent_weight * blue_noise, log_sigma);

  return color;
}

#endif /* CU_LUMINARY_FILM_H */
