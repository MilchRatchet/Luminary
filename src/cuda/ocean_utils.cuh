#ifndef CU_OCEAN_UTILS_H
#define CU_OCEAN_UTILS_H

#include "utils.cuh"

#define OCEAN_MAX_HEIGHT (device.scene.ocean.height + 2.66f * device.scene.ocean.amplitude)
#define OCEAN_MIN_HEIGHT (device.scene.ocean.height)

// Coefficients taken from
// M. Droske, J. Hanika, J. Vorba, A. Weidlich, M. Sabbadin, _Path Tracing in Production: The Path of Water_, ACM SIGGRAPH 2023 Courses,
// 2023.

__device__ RGBF ocean_jerlov_scattering_coefficient(const JerlovWaterType type) {
  switch (type) {
    case JERLOV_WATER_TYPE_I:
      return get_color(0.001f, 0.002f, 0.004f);
    case JERLOV_WATER_TYPE_IA:
      return get_color(0.002f, 0.004f, 0.007f);
    case JERLOV_WATER_TYPE_IB:
      return get_color(0.045f, 0.054f, 0.07f);
    case JERLOV_WATER_TYPE_II:
      return get_color(0.27f, 0.365f, 0.516f);
    case JERLOV_WATER_TYPE_III:
      return get_color(0.737f, 0.998f, 1.413f);
    case JERLOV_WATER_TYPE_1C:
      return get_color(0.274f, 0.372f, 0.526f);
    case JERLOV_WATER_TYPE_3C:
      return get_color(0.904f, 1.071f, 1.532f);
    case JERLOV_WATER_TYPE_5C:
      return get_color(3.589f, 1.382f, 1.857f);
    case JERLOV_WATER_TYPE_7C:
      return get_color(1.772f, 2.394f, 3.376f);
    case JERLOV_WATER_TYPE_9C:
      return get_color(2.347f, 3.18f, 4.496f);
  }

  return get_color(0.0f, 0.0f, 0.0f);
}

__device__ RGBF ocean_jerlov_absorption_coefficient(const JerlovWaterType type) {
  switch (type) {
    case JERLOV_WATER_TYPE_I:
      return get_color(0.309f, 0.053f, 0.009f);
    case JERLOV_WATER_TYPE_IA:
      return get_color(0.309f, 0.054f, 0.014f);
    case JERLOV_WATER_TYPE_IB:
      return get_color(0.309f, 0.054f, 0.015f);
    case JERLOV_WATER_TYPE_II:
      return get_color(0.31f, 0.054f, 0.016f);
    case JERLOV_WATER_TYPE_III:
      return get_color(0.31f, 0.056f, 0.031f);
    case JERLOV_WATER_TYPE_1C:
      return get_color(0.316f, 0.067f, 0.105f);
    case JERLOV_WATER_TYPE_3C:
      return get_color(0.508f, 0.052f, 0.161f);
    case JERLOV_WATER_TYPE_5C:
      return get_color(4.638f, 0.222f, 0.216f);
    case JERLOV_WATER_TYPE_7C:
      return get_color(0.351f, 0.188f, 0.574f);
    case JERLOV_WATER_TYPE_9C:
      return get_color(0.398f, 0.349f, 0.995f);
  }

  return get_color(0.0f, 0.0f, 0.0f);
}

__device__ float ocean_molecular_weight(const JerlovWaterType type) {
  switch (type) {
    case JERLOV_WATER_TYPE_I:
      return 0.93f;
    case JERLOV_WATER_TYPE_IA:
      return 0.44f;
    case JERLOV_WATER_TYPE_IB:
      return 0.06f;
    case JERLOV_WATER_TYPE_II:
      return 0.007f;
    case JERLOV_WATER_TYPE_III:
      return 0.003f;
    case JERLOV_WATER_TYPE_1C:
      return 0.005f;
    case JERLOV_WATER_TYPE_3C:
      return 0.003f;
    case JERLOV_WATER_TYPE_5C:
      return 0.001f;
    case JERLOV_WATER_TYPE_7C:
      return 0.0f;
    case JERLOV_WATER_TYPE_9C:
      return 0.0f;
  }

  return 0.0f;
}

// Henyey Greenstein importance sampling for g = 0
// pbrt v3 - Light Transport II: Volume Rendering - Sampling Volume Scattering
__device__ float ocean_molecular_phase_sampling_cosine(const vec3 ray, const float r) {
  return 2.0f * r - 1.0f;
}

// Henyey Greenstein importance sampling for g != 0
// pbrt v3 - Light Transport II: Volume Rendering - Sampling Volume Scattering
__device__ float ocean_particle_phase_sampling_cosine(const vec3 ray, const float r) {
  const float g = 0.924f;

  float denom = (1.0f - g + 2.0f * g * r);
  if (fabsf(denom) < eps) {
    denom = copysignf(eps, denom);
  }

  const float s = (1.0f - g * g) / denom;

  return (1.0f + g * g - s * s) / (2.0f * g);
}

__device__ vec3 ocean_phase_sampling(const vec3 ray) {
  const float molecular_weight = ocean_molecular_weight(device.scene.ocean.water_type);

  const float r = white_noise();

  float cos_angle;
  if (white_noise() < molecular_weight) {
    cos_angle = ocean_molecular_phase_sampling_cosine(ray, r);
  }
  else {
    cos_angle = ocean_particle_phase_sampling_cosine(ray, r);
  }

  return phase_sample_basis(cos_angle, white_noise(), ray);
}

__device__ float ocean_molecular_phase(const float cos_angle) {
  return henyey_greenstein_phase_function(cos_angle, 0.0f);
}

__device__ float ocean_particle_phase(const float cos_angle) {
  return henyey_greenstein_phase_function(cos_angle, 0.924f);
}

__device__ float ocean_phase(const float cos_angle) {
  const float molecular_weight = ocean_molecular_weight(device.scene.ocean.water_type);

  const float molecular_phase = ocean_molecular_phase(cos_angle);
  const float particle_phase  = ocean_particle_phase(cos_angle);

  return molecular_phase * molecular_weight + particle_phase * (1.0f - molecular_weight);
}

#endif /* CU_OCEAN_UTILS_H */
