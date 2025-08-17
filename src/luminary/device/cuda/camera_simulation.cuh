#ifndef CU_LUMINARY_CAMERA_SIMULATION_H
#define CU_LUMINARY_CAMERA_SIMULATION_H

#include "camera_utils.cuh"
#include "math.cuh"
#include "utils.cuh"

struct CameraLensSemiCircle {
  float distance;
  float radius;
} typedef CameraLensSemiCircle;

struct CameraSimulationState {
  vec3 origin;
  vec3 ray;
  float ior;
  RGBF weight;
} typedef CameraSimulationState;

#define THIN_LENS_NUM_SEMI_CIRCLES 2

__device__ void camera_simulation_step(CameraSimulationState& state, const uint32_t semi_circle_id, const ushort2 pixel) {
  // TODO: These will be written by the host into a dedicated buffer
  const float base_offset                                       = device.camera.focal_length + camera_thin_lens_focal_length();
  CameraLensSemiCircle semi_circles[THIN_LENS_NUM_SEMI_CIRCLES] = {
    {base_offset - 0.5f * device.camera.thin_lens_thickness + device.camera.thin_lens_radius2, device.camera.thin_lens_radius2},
    {base_offset + 0.5f * device.camera.thin_lens_thickness - device.camera.thin_lens_radius1, device.camera.thin_lens_radius1}};

  const CameraLensSemiCircle semi_circle = semi_circles[semi_circle_id];

  const vec3 semi_circle_center = get_vector(0.0f, 0.0f, -semi_circle.distance);

  const float dist = sphere_ray_intersection(state.ray, state.origin, semi_circle_center, semi_circle.radius);

  // No hit
  if (dist == FLT_MAX) {
    state.weight = splat_color(0.0f);

    return;
  }

  state.origin = add_vector(state.origin, scale_vector(state.ray, dist));

  const vec3 normal = normalize_vector(sub_vector(state.origin, semi_circle_center));

  // TODO
  const float lens_ior = semi_circle_id == 0 ? device.camera.thin_lens_ior : 1.0f;

  bool total_reflection;
  state.ray = refract_vector(scale_vector(state.ray, -1.0f), normal, state.ior / lens_ior, total_reflection);

  // TODO: Handle reflections
  if (total_reflection) {
    state.weight = splat_color(0.0f);

    return;
  }

  state.ior = lens_ior;
}

__device__ CameraSimulationResult camera_simulation_trace(const vec3 sensor_point, const vec3 initial_direction, const ushort2 pixel) {
  CameraSimulationState state;
  state.origin = sensor_point;
  state.ray    = initial_direction;
  state.ior    = 1.0f;
  state.weight = splat_color(1.0f);

  camera_simulation_step(state, 0, pixel);
  camera_simulation_step(state, 1, pixel);

  CameraSimulationResult result;
  result.origin = state.origin;
  result.ray    = state.ray;
  result.weight = state.weight;

  return result;
}

#endif /* CU_LUMINARY_CAMERA_SIMULATION_H */
