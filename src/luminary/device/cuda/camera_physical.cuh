#ifndef CU_LUMINARY_CAMERA_PHYSICAL_H
#define CU_LUMINARY_CAMERA_PHYSICAL_H

#include "camera_utils.cuh"
#include "math.cuh"
#include "utils.cuh"

LUMINARY_FUNCTION vec3 camera_physical_sample_sensor(const PathID& path_id) {
  const float2 jitter = camera_get_jitter(path_id);

  const float step = 2.0f * (device.camera.physical.sensor_width / device.settings.width);
  const float vfov = step * device.settings.height * 0.5f;

  const ushort2 sensor_pixel = path_id_get_pixel(path_id);

  vec3 sensor_point;
  sensor_point.x = device.camera.physical.sensor_width - step * (sensor_pixel.x + jitter.x);
  sensor_point.y = -vfov + step * (sensor_pixel.y + jitter.y);
  sensor_point.z = -camera_get_image_plane();

  return sensor_point;
}

LUMINARY_FUNCTION vec3 camera_physical_sample_exit_pupil(const vec3 sensor_point, const PathID& path_id, float& weight) {
  const float2 random = random_2D(RANDOM_TARGET_LENS, path_id);

  const float alpha = random.x * 2.0f * PI;
  const float beta  = sqrtf(random.y) * device.camera.physical.exit_pupil_radius;

  const vec3 target_point = get_vector(cosf(alpha) * beta, sinf(alpha) * beta, device.camera.physical.exit_pupil_point);

  const vec3 diff = sub_vector(target_point, sensor_point);

  const float dist = get_length(diff);
  const float area = device.camera.physical.exit_pupil_radius * device.camera.physical.exit_pupil_radius * PI;

  const vec3 ray = normalize_vector(diff);

  weight = area * fabsf(ray.z) / (dist * dist);

  return ray;
}

struct CameraSimulationState {
  vec3 origin;
  vec3 ray;
  float ior;
  float cylindrical_radius;
  float weight;
  float wavelength;
  bool is_forward;
  bool has_reflected;  // Biased optimization: Only allow one pair of reflections
} typedef CameraSimulationState;

struct CameraSimulationResult {
  vec3 origin;
  vec3 ray;
  float weight;
} typedef CameraSimulationResult;

LUMINARY_FUNCTION bool camera_simulation_intersect_aperture(const vec3 origin, const vec3 ray, const float dist) {
  const float aperture_dist = (device.camera.physical.aperture_point - origin.z) / ray.z;
  if (aperture_dist > 0.0f && aperture_dist < dist) {
    const vec3 aperture_hit = add_vector(origin, scale_vector(ray, aperture_dist));

    const float vertical_aperture_hit_dist_sq = aperture_hit.x * aperture_hit.x + aperture_hit.y * aperture_hit.y;
    const float aperture_radius               = device.camera.physical.aperture_radius;

    if (vertical_aperture_hit_dist_sq > aperture_radius * aperture_radius) {
      return true;
    }
  }

  return false;
}

LUMINARY_FUNCTION bool camera_simulation_intersect_medium_cylinder(
  vec3& origin, vec3& ray, float& weight, const float dist, const float cylindrical_radius, const float medium_ior) {
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

    weight *= fresnel;

    ray = reflect_vector(V, cylindrical_normal);

    return true;
  }

  return false;
}

template <bool ALLOW_REFLECTIONS, bool SPECTRAL_RENDERING>
LUMINARY_FUNCTION int32_t
  camera_simulation_step(CameraSimulationState& state, const uint32_t iteration, const int32_t interface_id, const PathID& path_id) {
  const DeviceCameraInterface interface = device.ptrs.camera_interfaces[interface_id];

  const vec3 semi_circle_center = get_vector(0.0f, 0.0f, interface.vertex - interface.radius);
  float dist                    = sphere_ray_intersection(state.ray, state.origin, semi_circle_center, fabsf(interface.radius));

  // No hit
  if (dist == FLT_MAX) {
    state.weight = 0.0f;
    return 0;
  }

  if (camera_simulation_intersect_aperture(state.origin, state.ray, dist)) {
    state.weight = 0.0f;
    return 0;
  }

  // This must happen before the origin gets modified
  // TODO: Optimize
  const bool is_inside = get_length(sub_vector(state.origin, semi_circle_center)) < fabsf(interface.radius);

  if (camera_simulation_intersect_medium_cylinder(state.origin, state.ray, state.weight, dist, state.cylindrical_radius, state.ior)) {
    dist = sphere_ray_intersection(state.ray, state.origin, semi_circle_center, fabsf(interface.radius));

    if (dist == FLT_MAX) {
      state.weight = 0.0f;
      return 0;
    }

    if (camera_simulation_intersect_aperture(state.origin, state.ray, dist)) {
      state.weight = 0.0f;
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
    state.weight = 0.0f;
    return 0;
  }

  vec3 normal = normalize_vector(sub_vector(state.origin, semi_circle_center));

  // Flip normal if we are inside
  if (is_inside) {
    normal = scale_vector(normal, -1.0f);
  }

  const vec3 V = scale_vector(state.ray, -1.0f);

  const float ior = state.ior / medium_ior;

  bool total_reflection;
  const vec3 refraction = refract_vector(V, normal, ior, total_reflection);
  const vec3 reflection = reflect_vector(V, normal);

  bool allow_reflection = false;
  if constexpr (ALLOW_REFLECTIONS) {
    allow_reflection = (interface_id != 0 || iteration != 0) && ((state.has_reflected == false) || (state.is_forward == false));
  }

  const bool allow_refraction = interface_id != 0 || iteration == 0;

  float weight;
  bool sampled_refraction;
  if (total_reflection) {
    weight             = allow_reflection ? 1.0f : 0.0f;
    sampled_refraction = false;
  }
  else {
    const float fresnel = bsdf_fresnel(normal, V, refraction, ior);

    if (allow_refraction && allow_reflection) {
      const float random = random_1D(RANDOM_TARGET_LENS_METHOD + iteration, path_id);

      weight             = 1.0f;
      sampled_refraction = random >= fresnel;
    }
    else if (allow_reflection) {
      weight             = fresnel;
      sampled_refraction = false;
    }
    else {
      weight             = 1.0f - fresnel;
      sampled_refraction = true;
    }
  }

  state.weight *= weight;

  state.ray                = sampled_refraction ? refraction : reflection;
  state.ior                = sampled_refraction ? medium_ior : state.ior;
  state.cylindrical_radius = sampled_refraction ? medium.cylindrical_radius : state.cylindrical_radius;
  state.is_forward         = sampled_refraction ? state.is_forward : !state.is_forward;
  state.has_reflected      = sampled_refraction ? state.has_reflected : true;

  return state.is_forward ? 1 : -1;
}

template <bool ALLOW_REFLECTIONS, bool SPECTRAL_RENDERING>
LUMINARY_FUNCTION CameraSimulationResult
  camera_simulation_trace(const vec3 sensor_point, const vec3 initial_direction, const float wavelength, const PathID& path_id) {
  CameraSimulationState state;
  state.origin             = sensor_point;
  state.ray                = initial_direction;
  state.ior                = IOR_AIR;
  state.cylindrical_radius = FLT_MAX;
  state.weight             = 1.0f;
  state.wavelength         = wavelength;
  state.is_forward         = true;
  state.has_reflected      = false;

  // There are num_interfaces + 1 media.
  const uint32_t num_interfaces = device.camera.physical.num_interfaces;

  uint32_t iteration        = 0;
  int32_t current_interface = 0;

  for (; iteration < RANDOM_LENS_MAX_INTERSECTIONS; iteration++) {
    current_interface += camera_simulation_step<ALLOW_REFLECTIONS, SPECTRAL_RENDERING>(state, iteration, current_interface, path_id);

    if (current_interface >= num_interfaces || current_interface < 0 || state.weight == 0.0f)
      break;
  }

  if (current_interface < 0 || (iteration == RANDOM_LENS_MAX_INTERSECTIONS && current_interface <= num_interfaces))
    state.weight = 0.0f;

  CameraSimulationResult result;
  result.origin = state.origin;
  result.ray    = state.ray;
  result.weight = state.weight;

  return result;
}

template <bool ALLOW_REFLECTIONS, bool SPECTRAL_RENDERING>
LUMINARY_FUNCTION CameraSampleResult camera_physical_sample(const PathID& path_id) {
  float wavelength_pdf;
  const float wavelength = spectral_sample_wavelength(random_1D(RANDOM_TARGET_LENS_WAVELENGTH, path_id), wavelength_pdf);

  const vec3 sensor_point = camera_physical_sample_sensor(path_id);

  float initial_weight;
  const vec3 initial_direction = camera_physical_sample_exit_pupil(sensor_point, path_id, initial_weight);

  const CameraSimulationResult simulation_result =
    camera_simulation_trace<ALLOW_REFLECTIONS, SPECTRAL_RENDERING>(sensor_point, initial_direction, wavelength, path_id);

  CameraSampleResult result;
  result.origin = simulation_result.origin;
  result.ray    = simulation_result.ray;
  result.weight = splat_color(simulation_result.weight * initial_weight);

  // Convert from spectral to RGB
  if constexpr (SPECTRAL_RENDERING) {
    result.weight = mul_color(result.weight, spectral_wavelength_to_rgb(wavelength));
    result.weight = scale_color(result.weight, 1.0f / wavelength_pdf);
  }

  // Physical camera simulation is in +Z direction but Luminary uses -Z convention
  result.origin.z = -result.origin.z;
  result.ray.z    = -result.ray.z;

  return result;
}

#endif /* CU_LUMINARY_CAMERA_PHYSICAL_H */
