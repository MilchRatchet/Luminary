#ifndef CU_LUMINARY_CAMERA_PHYSICAL_H
#define CU_LUMINARY_CAMERA_PHYSICAL_H

#include "camera_utils.cuh"
#include "math.cuh"
#include "utils.cuh"

__device__ vec3 camera_physical_sample_sensor(const ushort2 pixel) {
  const float2 jitter = camera_get_jitter();

  const float step = 2.0f * (device.camera.physical.sensor_width / device.settings.width);
  const float vfov = step * device.settings.height * 0.5f;

#ifndef CAMERA_DEBUG_RENDER
  const ushort2 sensor_pixel = pixel;
#else
  const ushort2 sensor_pixel = make_ushort2(device.settings.width >> 1, device.settings.height >> 1);
#endif

  vec3 sensor_point;
  sensor_point.x = device.camera.physical.sensor_width - step * (sensor_pixel.x + jitter.x);
  sensor_point.y = -vfov + step * (sensor_pixel.y + jitter.y);
  sensor_point.z = camera_get_image_plane();

  return sensor_point;
}

__device__ vec3 camera_physical_sample_exit_pupil(const vec3 sensor_point, const ushort2 pixel, float& weight) {
#ifndef CAMERA_DEBUG_RENDER
  const float2 random = random_2D(RANDOM_TARGET_LENS, pixel);
#else
  const float2 random = make_float2(((float) pixel_coords.x) / device.settings.width, ((float) pixel_coords.y) / device.settings.height);
#endif

  const float alpha = random.x * 2.0f * PI;
  const float beta  = sqrtf(random.y) * device.camera.physical.exit_pupil_radius;

  const vec3 target_point = get_vector(cosf(alpha) * beta, sinf(alpha) * beta, device.camera.physical.exit_pupil_point);

  const vec3 diff = sub_vector(target_point, sensor_point);

  const float dist = get_length(diff);

  const vec3 ray = normalize_vector(diff);

  weight = fabsf(ray.z) / (dist * dist);

  return ray;
}

struct CameraSimulationState {
  vec3 origin;
  vec3 ray;
  float ior;
  float weight;
  float wavelength;
  bool is_forward;
} typedef CameraSimulationState;

struct CameraSimulationResult {
  vec3 origin;
  vec3 ray;
  float weight;
} typedef CameraSimulationResult;

template <bool ALLOW_REFLECTIONS, bool SPECTRAL_RENDERING>
__device__ int32_t
  camera_simulation_step(CameraSimulationState& state, const uint32_t iteration, const int32_t interface_id, const ushort2 pixel) {
  const DeviceCameraInterface interface = device.ptrs.camera_interfaces[interface_id];

  const vec3 semi_circle_center = get_vector(0.0f, 0.0f, interface.center);
  const float dist              = sphere_ray_intersection(state.ray, state.origin, semi_circle_center, interface.radius);

  // No hit
  if (dist == FLT_MAX) {
    state.weight = 0.0f;

    return 0;
  }

  const int32_t medium_id         = state.is_forward ? interface_id + 1 : interface_id;
  const DeviceCameraMedium medium = device.ptrs.camera_media[medium_id];
  const float medium_ior          = camera_medium_get_ior<SPECTRAL_RENDERING>(medium, state.wavelength);

  // TODO: Optimize
  const bool is_inside = get_length(sub_vector(state.origin, semi_circle_center)) < interface.radius;

  state.origin = add_vector(state.origin, scale_vector(state.ray, dist));

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

  const bool allow_reflection = ALLOW_REFLECTIONS && (interface_id != 0 || iteration != 0);
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
      const float random = random_1D(RANDOM_TARGET_LENS_METHOD + iteration, pixel);

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

  state.ray        = sampled_refraction ? refraction : reflection;
  state.ior        = sampled_refraction ? medium_ior : state.ior;
  state.is_forward = sampled_refraction ? state.is_forward : !state.is_forward;

  return state.is_forward ? 1 : -1;
}

template <bool ALLOW_REFLECTIONS, bool SPECTRAL_RENDERING>
__device__ CameraSimulationResult
  camera_simulation_trace(const vec3 sensor_point, const vec3 initial_direction, const float wavelength, const ushort2 pixel) {
  CameraSimulationState state;
  state.origin     = sensor_point;
  state.ray        = initial_direction;
  state.ior        = 1.0f;
  state.weight     = 1.0f;
  state.wavelength = wavelength;
  state.is_forward = true;

  // There are num_interfaces + 1 media.
  const uint32_t num_interfaces = device.camera.physical.num_interfaces;

  uint32_t iteration        = 0;
  int32_t current_interface = 0;

  for (; iteration < RANDOM_LENS_MAX_INTERSECTIONS; iteration++) {
    current_interface += camera_simulation_step<ALLOW_REFLECTIONS, SPECTRAL_RENDERING>(state, iteration, current_interface, pixel);

    if (current_interface > num_interfaces || current_interface < 0 || state.weight == 0.0f)
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
__device__ CameraSampleResult camera_physical_sample(const ushort2 pixel) {
  float wavelength_pdf;
  const float wavelength = spectral_sample_wavelength(random_1D(RANDOM_TARGET_LENS_WAVELENGTH, pixel), wavelength_pdf);

  const vec3 sensor_point = camera_physical_sample_sensor(pixel);

  float initial_weight;
  const vec3 initial_direction = camera_physical_sample_exit_pupil(sensor_point, pixel, initial_weight);

  const CameraSimulationResult simulation_result =
    camera_simulation_trace<ALLOW_REFLECTIONS, SPECTRAL_RENDERING>(sensor_point, initial_direction, wavelength, pixel);

  CameraSampleResult result;
  result.origin = simulation_result.origin;
  result.ray    = simulation_result.ray;
  result.weight = splat_color(simulation_result.weight * initial_weight);

  // Convert from spectral to RGB
  if constexpr (SPECTRAL_RENDERING) {
    result.weight = mul_color(result.weight, spectral_wavelength_to_rgb(wavelength));
    result.weight = scale_color(result.weight, 1.0f / wavelength_pdf);
  }

  return result;
}

#endif /* CU_LUMINARY_CAMERA_PHYSICAL_H */
