#ifndef CU_LUMINARY_CAMERA_UTILS_H
#define CU_LUMINARY_CAMERA_UTILS_H

#include "random.cuh"
#include "utils.cuh"

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

LUMINARY_FUNCTION float2 camera_get_jitter() {
  if (device.state.sample_id == 0)
    return make_float2(0.5f, 0.5f);

  return random_2D_base_float(RANDOM_TARGET_CAMERA_JITTER, make_ushort2(0, 0), device.state.sample_id, 0);
}

LUMINARY_FUNCTION float camera_get_image_plane() {
#if 0
  const float f = device.camera.physical.focal_length;
  const float o = device.camera.object_distance * CAMERA_COMMON_INV_SCALE - device.camera.physical.front_principal_point;

  const float i = (f * o) / (o - f);

  return i - device.camera.physical.back_principal_point;
#else
  return device.camera.physical.image_plane_distance;
#endif
}

////////////////////////////////////////////////////////////////////
// Dispersion
////////////////////////////////////////////////////////////////////

LUMINARY_FUNCTION float camera_ior_cauchy_approximation(const float nd, const float abbe, const float wavelength) {
  const float range_factor =
    (1.0f / (CAMERA_FRAUNHOFER_F_LINE * CAMERA_FRAUNHOFER_F_LINE)) - (1.0f / (CAMERA_FRAUNHOFER_C_LINE * CAMERA_FRAUNHOFER_C_LINE));

  const float b = (nd - 1.0f) / (abbe * range_factor);

  const float a = nd - b * (1.0f / (CAMERA_FRAUNHOFER_D_LINE * CAMERA_FRAUNHOFER_D_LINE));

  return a + b / (wavelength * wavelength);
}

template <bool SPECTRAL_RENDERING>
LUMINARY_FUNCTION float camera_medium_get_ior(const DeviceCameraMedium medium, const float wavelength) {
  if constexpr (SPECTRAL_RENDERING)
    return (medium.abbe != 0.0f) ? camera_ior_cauchy_approximation(medium.design_ior, medium.abbe, wavelength) : medium.design_ior;

  return medium.design_ior;
}

#endif /* CU_LUMINARY_CAMERA_UTILS_H */
