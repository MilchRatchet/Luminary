#ifndef CU_LUMINARY_CAMERA_PHYSICAL_H
#define CU_LUMINARY_CAMERA_PHYSICAL_H

#include "camera_utils.cuh"
#include "math.cuh"
#include "ris.cuh"
#include "utils.cuh"

LUMINARY_FUNCTION vec3
  camera_physical_sample_exit_pupil(const vec3 sensor_point, const PathID& path_id, const uint32_t sample_id, float& sampling_weight) {
  const float2 random = random_2D(RANDOM_TARGET_LENS + sample_id, path_id);

  const float alpha = random.x * 2.0f * PI;
  const float beta  = sqrtf(random.y) * device.camera_aux.exit_pupil_radius;

  const vec3 target_point = get_vector(cosf(alpha) * beta, sinf(alpha) * beta, device.camera_aux.exit_pupil_point);

  const vec3 diff = sub_vector(target_point, sensor_point);

  const float dist = get_length(diff);
  const float area = device.camera_aux.exit_pupil_radius * device.camera_aux.exit_pupil_radius * PI;

  const vec3 ray = normalize_vector(diff);

  sampling_weight = area * fabsf(ray.z) / (dist * dist);

  return ray;
}

struct CameraSimulationState {
  vec3 origin;
  vec3 ray;
  float ior;
  float cylindrical_radius;
  float throughput;
  float probability_density;
  float wavelength;
  bool is_forward;
  bool has_retro_reflected;  // Biased optimization: Only allow one pair of reflections
  bool has_forward_reflected;
} typedef CameraSimulationState;

struct CameraSimulationResult {
  vec3 origin;
  vec3 ray;
  float throughput;
  float probability_density;
  bool has_reflected;
} typedef CameraSimulationResult;

LUMINARY_FUNCTION bool camera_simulation_intersect_aperture(const vec3 origin, const vec3 ray, const float dist) {
  const float aperture_dist = (ray.z != 0.0f) ? (device.camera_aux.aperture_point - origin.z) / ray.z : -FLT_MAX;
  if (aperture_dist < 0.0f || aperture_dist > dist)
    return false;

  const vec3 aperture_hit = add_vector(origin, scale_vector(ray, aperture_dist));

  const float vertical_aperture_hit_dist_sq = aperture_hit.x * aperture_hit.x + aperture_hit.y * aperture_hit.y;
  const float aperture_radius               = device.camera_aux.aperture_radius;

  if (vertical_aperture_hit_dist_sq > aperture_radius * aperture_radius)
    return true;

  if (device.camera.aperture_shape == LUMINARY_APERTURE_ROUND)
    return false;

  float angle = atan2f(aperture_hit.y, aperture_hit.x);
  angle       = (angle < 0.0f) ? (angle + (2.0f * PI)) : angle;

  uint32_t section_id = angle * (device.camera.aperture_blade_count * (1.0f / (2.0f * PI)));

  if (section_id >= device.camera.aperture_blade_count)
    section_id = device.camera.aperture_blade_count - 1;

  const float angle_a = section_id * (2.0f * PI) / device.camera.aperture_blade_count;
  const float angle_b = (section_id + 1) * (2.0f * PI) / device.camera.aperture_blade_count;

  const float2 point_a = make_float2(aperture_radius * cosf(angle_a), aperture_radius * sinf(angle_a));
  const float2 point_b = make_float2(aperture_radius * cosf(angle_b), aperture_radius * sinf(angle_b));

  const float2 blade_edge   = make_float2(point_b.x - point_a.x, point_b.y - point_a.y);
  const float2 blade_normal = make_float2(blade_edge.y, -blade_edge.x);
  const float2 hit_rel_a    = make_float2(aperture_hit.x - point_a.x, aperture_hit.y - point_a.y);

  const float dot = blade_normal.x * hit_rel_a.x + blade_normal.y * hit_rel_a.y;

  if (dot >= 0.0f)
    return true;

  return false;
}

LUMINARY_FUNCTION bool camera_simulation_intersect_medium_cylinder(
  vec3& origin, vec3& ray, float& throughput, const float dist, const float cylindrical_radius, const float medium_ior) {
  if (cylindrical_radius == FLT_MAX)
    return false;

  vec3 cylindrical_ray           = get_vector(ray.x, ray.y, 0.0f);
  const float cylindrical_length = get_length(cylindrical_ray);

  if (cylindrical_length == 0.0f)
    return false;

  cylindrical_ray = scale_vector(cylindrical_ray, 1.0f / cylindrical_length);

  const vec3 cylindrical_origin = get_vector(origin.x, origin.y, 0.0f);

  // TODO: Optimize, this only needs 2D math but I am lazy so I use existing 3D implementations
  float cylindrical_dist = sphere_ray_intersection(cylindrical_ray, cylindrical_origin, get_vector(0.0f, 0.0f, 0.0f), cylindrical_radius);

  cylindrical_dist *= 1.0f / cylindrical_length;

  if (cylindrical_dist > 0.0f && cylindrical_dist < dist) {
    origin = add_vector(origin, scale_vector(ray, cylindrical_dist));

    const vec3 cylindrical_normal = normalize_vector(get_vector(-origin.x, -origin.y, 0.0f));
    const float ior               = medium_ior * (1.0f / IOR_AIR);
    const vec3 V                  = scale_vector(ray, -1.0f);

    bool total_reflection;
    const vec3 refraction = refract_vector(V, cylindrical_normal, ior, total_reflection);

    const float fresnel = (total_reflection == false) ? bsdf_fresnel(cylindrical_normal, V, refraction, ior) : 1.0f;

    throughput *= fresnel;

    ray = reflect_vector(V, cylindrical_normal);

    return true;
  }

  return false;
}

LUMINARY_FUNCTION float camera_simulation_interface_intersection(CameraSimulationState& state, const vec3 center, const float radius) {
  if (radius == FLT_MAX) {
    return (state.ray.z != 0.0f) ? (center.z - state.origin.z) / state.ray.z : FLT_MAX;
  }

  return sphere_ray_intersection(state.ray, state.origin, center, fabsf(radius));
}

template <bool ALLOW_REFLECTIONS, bool SPECTRAL_RENDERING>
LUMINARY_FUNCTION int32_t camera_simulation_step(
  CameraSimulationState& state, const uint32_t iteration, const int32_t interface_id, const PathID& path_id, const uint32_t sample_id) {
  const DeviceCameraInterface interface = device.ptrs.camera_interfaces[interface_id];

  const float center            = (interface.radius != FLT_MAX) ? interface.vertex - interface.radius : interface.vertex;
  const vec3 semi_circle_center = get_vector(0.0f, 0.0f, center);
  float dist                    = camera_simulation_interface_intersection(state, semi_circle_center, interface.radius);

  // No hit
  if (dist == FLT_MAX) {
    state.throughput = 0.0f;
    return 0;
  }

  if (camera_simulation_intersect_aperture(state.origin, state.ray, dist)) {
    state.throughput = 0.0f;
    return 0;
  }

  // This must happen before the origin gets modified
  // TODO: Optimize
  const bool is_inside = get_length(sub_vector(state.origin, semi_circle_center)) < fabsf(interface.radius);

  if (camera_simulation_intersect_medium_cylinder(state.origin, state.ray, state.throughput, dist, state.cylindrical_radius, state.ior)) {
    dist = camera_simulation_interface_intersection(state, semi_circle_center, interface.radius);

    state.has_forward_reflected = true;

    if (dist == FLT_MAX) {
      state.throughput = 0.0f;
      return 0;
    }

    if (camera_simulation_intersect_aperture(state.origin, state.ray, dist)) {
      state.throughput = 0.0f;
      return 0;
    }
  }

  const int32_t medium_id         = state.is_forward ? interface_id + 1 : interface_id;
  const DeviceCameraMedium medium = device.ptrs.camera_media[medium_id];
  const float medium_ior          = camera_medium_get_ior<SPECTRAL_RENDERING>(medium, state.wavelength);

  state.origin = add_vector(state.origin, scale_vector(state.ray, dist));

  const float vertical_hit_dist_sq = state.origin.x * state.origin.x + state.origin.y * state.origin.y;
  if (vertical_hit_dist_sq > interface.cylindrical_radius * interface.cylindrical_radius) {
    // Hit is past the vertical limits of the interface
    state.throughput = 0.0f;
    return 0;
  }

  vec3 normal = (interface.radius != FLT_MAX) ? normalize_vector(sub_vector(state.origin, semi_circle_center))
                                              : get_vector(0.0f, 0.0f, (state.origin.z > center) ? 1.0f : -1.0f);

  // Flip normal if we are inside
  if (is_inside && interface.radius != FLT_MAX) {
    normal = scale_vector(normal, -1.0f);
  }

  const vec3 V = scale_vector(state.ray, -1.0f);

  const float ior = state.ior / medium_ior;

  bool total_reflection;
  const vec3 refraction = refract_vector(V, normal, ior, total_reflection);
  const vec3 reflection = reflect_vector(V, normal);

  bool allow_reflection = false;
  if constexpr (ALLOW_REFLECTIONS) {
    allow_reflection = (interface_id != 0 || iteration != 0) && ((state.has_retro_reflected == false) || (state.is_forward == false));
  }

  const bool allow_refraction = interface_id != 0 || iteration == 0;

  float throughput;
  float probality;
  bool sampled_refraction;
  if (total_reflection) {
    throughput         = allow_reflection ? 1.0f : 0.0f;
    probality          = 1.0f;
    sampled_refraction = false;
  }
  else {
    const float fresnel = bsdf_fresnel(normal, V, refraction, ior);

    if (allow_refraction && allow_reflection) {
      const float random = random_1D(RANDOM_TARGET_LENS_METHOD + iteration + sample_id * RANDOM_LENS_MAX_INTERSECTIONS, path_id);

      probality = 1.0f / device.camera_aux.num_interfaces;

      sampled_refraction = random >= probality;

      throughput = (sampled_refraction) ? 1.0f - fresnel : fresnel;
      probality  = (sampled_refraction) ? 1.0f - probality : probality;
    }
    else if (allow_reflection) {
      throughput         = fresnel;
      probality          = 1.0f;
      sampled_refraction = false;
    }
    else {
      throughput         = 1.0f - fresnel;
      probality          = 1.0f;
      sampled_refraction = true;
    }
  }

  state.throughput *= throughput;
  state.probability_density *= probality;

  state.ray                 = sampled_refraction ? refraction : reflection;
  state.ior                 = sampled_refraction ? medium_ior : state.ior;
  state.cylindrical_radius  = sampled_refraction ? medium.cylindrical_radius : state.cylindrical_radius;
  state.is_forward          = sampled_refraction ? state.is_forward : !state.is_forward;
  state.has_retro_reflected = sampled_refraction ? state.has_retro_reflected : true;

  return state.is_forward ? 1 : -1;
}

template <bool ALLOW_REFLECTIONS, bool SPECTRAL_RENDERING>
LUMINARY_FUNCTION CameraSimulationResult camera_simulation_trace(
  const vec3 sensor_point, const vec3 initial_direction, const float wavelength, const PathID& path_id, uint32_t sample_id) {
  CameraSimulationState state;
  state.origin                = sensor_point;
  state.ray                   = initial_direction;
  state.ior                   = IOR_AIR;
  state.cylindrical_radius    = FLT_MAX;
  state.throughput            = 1.0f;
  state.probability_density   = 1.0f;
  state.wavelength            = wavelength;
  state.is_forward            = true;
  state.has_retro_reflected   = false;
  state.has_forward_reflected = false;

  // There are num_interfaces + 1 media.
  const uint32_t num_interfaces = device.camera_aux.num_interfaces;

  uint32_t iteration        = 0;
  int32_t current_interface = 0;

  for (; iteration < RANDOM_LENS_MAX_INTERSECTIONS; iteration++) {
    current_interface +=
      camera_simulation_step<ALLOW_REFLECTIONS, SPECTRAL_RENDERING>(state, iteration, current_interface, path_id, sample_id);

    if (current_interface >= num_interfaces || current_interface < 0 || state.throughput == 0.0f)
      break;
  }

  if (current_interface < 0 || (iteration == RANDOM_LENS_MAX_INTERSECTIONS && current_interface <= num_interfaces))
    state.throughput = 0.0f;

  CameraSimulationResult result;
  result.origin              = state.origin;
  result.ray                 = state.ray;
  result.throughput          = state.throughput;
  result.probability_density = state.probability_density;
  result.has_reflected       = state.has_retro_reflected || state.has_forward_reflected;

  return result;
}

template <bool ALLOW_REFLECTIONS, bool SPECTRAL_RENDERING>
LUMINARY_FUNCTION CameraSampleResult camera_physical_sample(const PathID& path_id) {
  float wavelength_pdf;
  const float wavelength = spectral_sample_wavelength(random_1D(RANDOM_TARGET_LENS_WAVELENGTH, path_id), wavelength_pdf);

  const vec3 sensor_point = camera_sample_sensor(path_id);

  CameraSimulationResult selected_simulation_result;
  selected_simulation_result.origin              = get_vector(0.0f, 0.0f, 0.0f);
  selected_simulation_result.ray                 = get_vector(0.0f, 0.0f, 0.0f);
  selected_simulation_result.throughput          = 0.0f;
  selected_simulation_result.probability_density = 1.0f;

  RISReservoir ris_reservoir = ris_reservoir_init(random_1D(RANDOM_TARGET_LENS_RESAMPLING, path_id));

  for (uint32_t sample_id = 0; sample_id < RANDOM_LENS_MAX_SAMPLES; sample_id++) {
    float pupil_sampling_weight;
    const vec3 initial_direction = camera_physical_sample_exit_pupil(sensor_point, path_id, sample_id, pupil_sampling_weight);

    const CameraSimulationResult simulation_result =
      camera_simulation_trace<ALLOW_REFLECTIONS, SPECTRAL_RENDERING>(sensor_point, initial_direction, wavelength, path_id, sample_id);

    float target = (simulation_result.throughput > 0.0f) ? 1.0f : 0.0f;

    if (simulation_result.has_reflected)
      target *= 128.0f;

    const float sampling_weight = pupil_sampling_weight / (RANDOM_LENS_MAX_SAMPLES * simulation_result.probability_density);

    if (ris_reservoir_add_sample(ris_reservoir, target, sampling_weight))
      selected_simulation_result = simulation_result;
  }

  CameraSampleResult result;
  result.origin = selected_simulation_result.origin;
  result.ray    = selected_simulation_result.ray;
  result.weight = splat_color(selected_simulation_result.throughput * ris_reservoir_get_sampling_weight(ris_reservoir));

  // Convert from spectral to RGB
  if constexpr (SPECTRAL_RENDERING) {
    result.weight = mul_color(result.weight, spectral_wavelength_to_rgb(wavelength));
    result.weight = scale_color(result.weight, 1.0f / wavelength_pdf);
  }

  // We center the result around the last vertex. This allows for a more stable perspective when scaling the camera.
  result.origin.z -= device.camera_aux.last_vertex;

  // Camera simulation is in +Z direction but Luminary uses -Z convention
  result.origin.z = -result.origin.z;
  result.ray.z    = -result.ray.z;

  return result;
}

#endif /* CU_LUMINARY_CAMERA_PHYSICAL_H */
