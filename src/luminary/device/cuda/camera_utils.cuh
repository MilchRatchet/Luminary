#ifndef CU_LUMINARY_CAMERA_UTILS_H
#define CU_LUMINARY_CAMERA_UTILS_H

#include "random.cuh"
#include "utils.cuh"

// #define CAMERA_DEBUG_RENDER

// mm to m
#define CAMERA_COMMON_SCALE (0.001f)
#define CAMERA_COMMON_INV_SCALE (1.0f / CAMERA_COMMON_SCALE)

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

__device__ float camera_thin_lens_inv_focal_length() {
  const float radius1 = device.camera.thin_lens_radius;
  const float radius2 = -device.camera.thin_lens_radius;

  float focal_length = 0.0f;

  focal_length += 1.0f / radius1;
  focal_length -= 1.0f / radius2;
  focal_length +=
    ((device.camera.thin_lens_ior - 1.0f) * device.camera.thin_lens_thickness) / (radius1 * radius2 * device.camera.thin_lens_ior);

  focal_length *= (device.camera.thin_lens_ior - 1.0f);

  return focal_length;
}

__device__ float camera_thin_lens_focal_length() {
  return 1.0f / camera_thin_lens_inv_focal_length();
}

__device__ float camera_thin_lens_front_focal_length() {
  const float f     = camera_thin_lens_focal_length();
  const float num   = (device.camera.thin_lens_ior - 1.0f) * device.camera.thin_lens_thickness;
  const float denom = device.camera.thin_lens_ior * device.camera.thin_lens_radius;

  return f * (1.0f - num / denom);
}

__device__ float camera_thin_lens_back_focal_length() {
  const float f     = camera_thin_lens_focal_length();
  const float num   = (device.camera.thin_lens_ior - 1.0f) * device.camera.thin_lens_thickness;
  const float denom = -device.camera.thin_lens_ior * device.camera.thin_lens_radius;

  return f * (1.0f - num / denom);
}

__device__ float camera_thin_lens_front_principal_plane() {
  return device.camera.thin_lens_thickness - camera_thin_lens_focal_length() + camera_thin_lens_front_focal_length();
}

__device__ float camera_thin_lens_back_principal_plane() {
  return camera_thin_lens_focal_length() - camera_thin_lens_front_focal_length();
}

__device__ float camera_thin_lens_object_distance() {
  // Hack, focal_length is actually the object distance with respect to the origin of the camera / back vertex
  float o = device.camera.focal_length * CAMERA_COMMON_INV_SCALE * (1.0f / device.camera.camera_scale);

  // Distance between object and front principal plane
  return o - camera_thin_lens_front_principal_plane();
}

__device__ float camera_thin_lens_image_plane() {
  const float f = camera_thin_lens_focal_length();
  const float o = camera_thin_lens_object_distance();

  float i = (f * o) / (o - f);

  return i + camera_thin_lens_back_principal_plane();
}

#endif /* CU_LUMINARY_CAMERA_UTILS_H */
