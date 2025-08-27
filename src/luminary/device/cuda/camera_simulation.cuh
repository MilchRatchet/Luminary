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
  float weight;
  bool is_forward;
} typedef CameraSimulationState;

#define THIN_LENS_NUM_SEMI_CIRCLES 2

__device__ int32_t
  camera_simulation_step(CameraSimulationState& state, const uint32_t iteration, const int32_t semi_circle_id, const ushort2 pixel) {
  // TODO: These will be written by the host into a dedicated buffer
  CameraLensSemiCircle semi_circles[THIN_LENS_NUM_SEMI_CIRCLES] = {
    {-device.camera.thin_lens_radius, device.camera.thin_lens_radius},
    {-device.camera.thin_lens_thickness + device.camera.thin_lens_radius, device.camera.thin_lens_radius}};

  const CameraLensSemiCircle semi_circle = semi_circles[semi_circle_id];

  const vec3 semi_circle_center = get_vector(0.0f, 0.0f, semi_circle.distance);

  const float dist = sphere_ray_intersection(state.ray, state.origin, semi_circle_center, semi_circle.radius);

  // No hit
  if (dist == FLT_MAX) {
    state.weight = 0.0f;

    return 0;
  }

  // TODO: Optimize
  const bool is_inside = get_length(sub_vector(state.origin, semi_circle_center)) < semi_circle.radius;

  state.origin = add_vector(state.origin, scale_vector(state.ray, dist));

  vec3 normal = normalize_vector(sub_vector(state.origin, semi_circle_center));

  // Flip normal if we are inside
  if (is_inside) {
    normal = scale_vector(normal, -1.0f);
  }

  // TODO
  const float lens_ior = (is_inside == false) ? device.camera.thin_lens_ior : 1.0f;

  const vec3 V = scale_vector(state.ray, -1.0f);

  const float ior = state.ior / lens_ior;

  bool total_reflection;
  const vec3 refraction = refract_vector(V, normal, ior, total_reflection);
  const vec3 reflection = reflect_vector(V, normal);

  const bool allow_reflection = semi_circle_id != 0 || iteration != 0;
  const bool allow_refraction = semi_circle_id != 0 || iteration == 0;

  float weight;
  bool sampled_refraction;
  if (total_reflection) {
    weight             = allow_reflection ? 1.0f : 0.0f;
    sampled_refraction = false;
  }
  else {
    const float fresnel = bsdf_fresnel(normal, V, refraction, ior);

    if (allow_refraction && allow_reflection) {
      const float random = random_1D(RANDOM_TARGET_LENS_METHOD + iteration, pixel);

      weight             = 1.0f;
      sampled_refraction = random >= fresnel;
    }
    else if (allow_refraction) {
      weight             = 1.0f - fresnel;
      sampled_refraction = true;
    }
    else {
      weight             = fresnel;
      sampled_refraction = false;
    }
  }

  state.weight *= weight;

  state.ray        = sampled_refraction ? refraction : reflection;
  state.ior        = sampled_refraction ? lens_ior : state.ior;
  state.is_forward = sampled_refraction ? state.is_forward : !state.is_forward;

  return state.is_forward ? 1 : -1;
}

__device__ CameraSimulationResult camera_simulation_trace(const vec3 sensor_point, const vec3 initial_direction, const ushort2 pixel) {
  CameraSimulationState state;
  state.origin     = sensor_point;
  state.ray        = initial_direction;
  state.ior        = 1.0f;
  state.weight     = 1.0f;
  state.is_forward = true;

  int32_t current_semicircle = 0;
  uint32_t iteration         = 0;

  for (; iteration < RANDOM_LENS_MAX_INTERSECTIONS; iteration++) {
    current_semicircle += camera_simulation_step(state, iteration, current_semicircle, pixel);

    if (current_semicircle >= 2 || current_semicircle < 0 || state.weight == 0.0f)
      break;
  }

  if (current_semicircle < 0 || (iteration == RANDOM_LENS_MAX_INTERSECTIONS && current_semicircle < 2))
    state.weight = 0.0f;

  CameraSimulationResult result;
  result.origin = state.origin;
  result.ray    = state.ray;
  result.weight = splat_color(state.weight);

  return result;
}

#endif /* CU_LUMINARY_CAMERA_SIMULATION_H */
