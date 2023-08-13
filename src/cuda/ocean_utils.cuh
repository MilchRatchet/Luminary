#ifndef CU_OCEAN_UTILS_H
#define CU_OCEAN_UTILS_H

#include "utils.cuh"

#define OCEAN_MAX_HEIGHT (device.scene.ocean.height + 2.66f * device.scene.ocean.amplitude)
#define OCEAN_MIN_HEIGHT (device.scene.ocean.height)

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

#endif /* CU_OCEAN_UTILS_H */
