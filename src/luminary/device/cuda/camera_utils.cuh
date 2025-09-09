#ifndef CU_LUMINARY_CAMERA_UTILS_H
#define CU_LUMINARY_CAMERA_UTILS_H

#include "random.cuh"
#include "utils.cuh"

// #define CAMERA_DEBUG_RENDER

// mm to m
#define CAMERA_COMMON_SCALE (0.001f)
#define CAMERA_COMMON_INV_SCALE (1.0f / CAMERA_COMMON_SCALE)

#define CAMERA_FRAUNHOFER_D_LINE (587.6f)
#define CAMERA_FRAUNHOFER_F_LINE (486.1f)
#define CAMERA_FRAUNHOFER_C_LINE (656.3f)

#define CAMERA_DESIGN_WAVELENGTH CAMERA_FRAUNHOFER_D_LINE

struct CameraSampleResult {
  vec3 origin;
  vec3 ray;
  RGBF weight;
} typedef CameraSampleResult;

__device__ float2 camera_get_jitter() {
#ifndef CAMERA_DEBUG_RENDER
  if (device.state.sample_id == 0)
    return make_float2(0.5f, 0.5f);

  return random_2D_base_float(RANDOM_TARGET_CAMERA_JITTER, make_ushort2(0, 0), device.state.sample_id, 0);
#else
  return make_float2(0.5f, 0.5f);
#endif
}

__device__ float camera_get_image_plane() {
  const float f = device.camera.physical.focal_length;
  const float o = device.camera.physical.front_principal_point - device.camera.object_distance;

  float i = (f * o) / (o - f);

  return i + device.camera.physical.back_principal_point;
}

////////////////////////////////////////////////////////////////////
// Dispersion
////////////////////////////////////////////////////////////////////

__device__ float camera_ior_cauchy_approximation(const float nd, const float abbe, const float wavelength) {
  const float range_factor =
    (1.0f / (CAMERA_FRAUNHOFER_F_LINE * CAMERA_FRAUNHOFER_F_LINE)) - (1.0f / (CAMERA_FRAUNHOFER_C_LINE * CAMERA_FRAUNHOFER_C_LINE));

  const float b = (nd - 1.0f) / (abbe * range_factor);

  const float a = nd - b * (1.0f / (CAMERA_FRAUNHOFER_D_LINE * CAMERA_FRAUNHOFER_D_LINE));

  return a + b / (wavelength * wavelength);
}

template <bool SPECTRAL_RENDERING>
__device__ float camera_medium_get_ior(const DeviceCameraMedium medium, const float wavelength) {
  if constexpr (SPECTRAL_RENDERING)
    return camera_ior_cauchy_approximation(medium.design_ior, medium.abbe, wavelength);

  return medium.design_ior;
}

#endif /* CU_LUMINARY_CAMERA_UTILS_H */
