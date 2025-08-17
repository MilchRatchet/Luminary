#ifndef CU_LUMINARY_CAMERA_UTILS_H
#define CU_LUMINARY_CAMERA_UTILS_H

#include "random.cuh"
#include "utils.cuh"

// #define CAMERA_DEBUG_RENDER

struct CameraSimulationResult {
  vec3 origin;
  vec3 ray;
  RGBF weight;
} typedef CameraSimulationResult;

__device__ float2 camera_get_jitter() {
#ifndef CAMERA_DEBUG_RENDER
  if (device.state.sample_id == 0)
    return make_float2(0.5f, 0.5f);

  return random_2D_base_float(RANDOM_TARGET_CAMERA_JITTER, make_ushort2(0, 0), device.state.sample_id, 0);
#else
  return make_float2(0.5f, 0.5f);
#endif
}

__device__ float camera_thin_lens_focal_length() {
  float focal_length = 0.0f;

  focal_length += 1.0f / device.camera.thin_lens_radius1;
  focal_length += 1.0f / device.camera.thin_lens_radius2;
  focal_length += ((device.camera.thin_lens_ior - 1.0f) * device.camera.thin_lens_thickness)
                  / (device.camera.thin_lens_radius1 * device.camera.thin_lens_radius2 * device.camera.thin_lens_ior);

  focal_length *= (device.camera.thin_lens_ior - 1.0f);

  return focal_length;
}

#endif /* CU_LUMINARY_CAMERA_UTILS_H */
