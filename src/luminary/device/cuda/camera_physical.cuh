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

LUMINARY_FUNCTION bool camera_simulation_intersect_aperture(
  const vec3 origin, const vec3 ray, const float dist, float& edge_dist, vec3& hit_point, float2& edge_normal) {
  const float aperture_dist = (ray.z != 0.0f) ? (device.camera_aux.aperture_point - origin.z) / ray.z : -FLT_MAX;
  if (aperture_dist < 0.0f || aperture_dist > dist)
    return false;

  const vec3 aperture_hit = add_vector(origin, scale_vector(ray, aperture_dist));
  hit_point               = aperture_hit;

  const float vertical_aperture_hit_dist_sq = aperture_hit.x * aperture_hit.x + aperture_hit.y * aperture_hit.y;
  const float aperture_radius               = device.camera_aux.aperture_radius;

  if (vertical_aperture_hit_dist_sq > aperture_radius * aperture_radius)
    return true;

  if (device.camera.aperture_shape == LUMINARY_APERTURE_ROUND) {
    edge_dist   = aperture_radius - sqrtf(vertical_aperture_hit_dist_sq);
    edge_normal = make_float2(-aperture_hit.x, -aperture_hit.y);
    return false;
  }

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

  const float blade_length       = sqrtf(blade_edge.x * blade_edge.x + blade_edge.y * blade_edge.y);
  const float2 blade_normal_norm = make_float2(blade_normal.x / blade_length, blade_normal.y / blade_length);

  const float2 hit_rel_a = make_float2(aperture_hit.x - point_a.x, aperture_hit.y - point_a.y);

  const float dot = blade_normal.x * hit_rel_a.x + blade_normal.y * hit_rel_a.y;

  if (dot >= 0.0f)
    return true;

  edge_dist   = -(blade_normal_norm.x * hit_rel_a.x + blade_normal_norm.y * hit_rel_a.y);
  edge_normal = blade_normal_norm;
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

LUMINARY_FUNCTION float camera_simulation_interface_intersection(
  CameraSimulationState& state, vec3 center, float radius, float cylindrical_radius) {
  float dist;
  if (radius == FLT_MAX) {
    dist = (state.ray.z != 0.0f) ? (center.z - state.origin.z) / state.ray.z : FLT_MAX;

    dist = (state.is_forward) ? dist : -dist;
  }
  else {
    const vec3 diff = sub_vector(state.origin, center);
    const float dot = dot_product(diff, state.ray);
    const float r2  = radius * radius;
    const float c   = dot_product(diff, diff) - r2;
    const vec3 k    = sub_vector(diff, scale_vector(state.ray, dot));
    const float d   = r2 - dot_product(k, k);

    if (d < 0.0f)
      return FLT_MAX;

    const float sd = sqrtf(d);
    const float q  = -dot - copysignf(sd, dot);

    float t0 = (q != 0.0f) ? c / q : FLT_MAX;
    float t1 = q;

    t0 = (t0 >= 0.0f) ? t0 : FLT_MAX;
    t1 = (t1 >= 0.0f) ? t1 : FLT_MAX;

    bool interface_curving_away = radius < 0.0f;
    interface_curving_away ^= state.is_forward == false;

    const float t_min = fminf(t0, t1);

    dist = t_min;

    // If the interface is curving away, we want the first hit
    // If the interface is curving away, we always want the back hit, even if we in theory would have an earlier hit. This is simply a
    // symptom of the interface representation being a sphere.
    if (interface_curving_away == false) {
      const float t_max = fmaxf(t0, t1);

      dist = (t0 != FLT_MAX && t1 != FLT_MAX) ? t_max : t_min;
    }
  }

  if (dist == FLT_MAX)
    return FLT_MAX;

  const vec3 interface_hit_point = add_vector(state.origin, scale_vector(state.ray, dist));

  const float vertical_hit_dist_sq = interface_hit_point.x * interface_hit_point.x + interface_hit_point.y * interface_hit_point.y;

  // Hit is past the vertical limits of the interface
  if (vertical_hit_dist_sq > cylindrical_radius * cylindrical_radius)
    dist = FLT_MAX;

  return dist;
}

LUMINARY_FUNCTION vec3 camera_aperture_diffraction_sample(
  const vec3 ray, const float edge_dist, const float2 edge_normal, const float wavelength, const float2 random, float& pdf) {
  if (edge_dist <= 0.0f) {
    pdf = 1.0f;
    return ray;
  }

  // The base diffraction model based on the uncertainty principle:
  // delta_theta ~ lambda / (2 * pi * x)
  const float lambda_mm  = wavelength * 1e-6f;
  const float base_angle = lambda_mm / (2.0f * PI * edge_dist);

  // Sample a positive half-Cauchy distribution for the polar angular deviation
  // This maps [0, 1) uniformly to [0, inf)
  const float sample_u        = fmaxf(random.x, 1e-6f);
  const float angle_deviation = base_angle * tanf((PI * 0.5f) * sample_u);

  // PDF of the sampled theta (Cauchy distribution PDF)
  const float theta_normalized = angle_deviation / base_angle;
  const float pdf_theta        = 2.0f / (PI * base_angle * (1.0f + theta_normalized * theta_normalized));

  // Construct a tangent basis aligned with the aperture edge normal
  vec3 edge_normal_3d  = get_vector(edge_normal.x, edge_normal.y, 0.0f);
  vec3 U               = sub_vector(edge_normal_3d, scale_vector(ray, dot_product(edge_normal_3d, ray)));
  const float U_length = get_length(U);

  if (U_length > 1e-6f) {
    U = scale_vector(U, 1.0f / U_length);
  }
  else {
    if (fabsf(ray.x) > fabsf(ray.y))
      U = normalize_vector(get_vector(ray.z, 0.0f, -ray.x));
    else
      U = normalize_vector(get_vector(0.0f, -ray.z, ray.y));
  }
  vec3 V = cross_product(ray, U);

  const float v_spread_angle = 1e-4f;
  const float phi = (random.y > 0.5f ? 0.0f : PI) + (random.y > 0.5f ? (random.y - 0.75f) : (random.y - 0.25f)) * 4.0f * v_spread_angle;

  const float cos_phi = cosf(phi);
  const float sin_phi = sinf(phi);

  // Apply perturbation
  const float cos_theta = cosf(angle_deviation);
  const float sin_theta = sinf(angle_deviation);

  // Convert theta and phi to a solid angle PDF
  // p(omega) = p(theta) * p(phi) / sin(theta)
  const float pdf_phi = 1.0f / (4.0f * v_spread_angle);
  pdf                 = pdf_theta * pdf_phi / fmaxf(sin_theta, eps);

  const vec3 diffracted =
    add_vector(scale_vector(ray, cos_theta), add_vector(scale_vector(U, sin_theta * cos_phi), scale_vector(V, sin_theta * sin_phi)));

  return normalize_vector(diffracted);
}

LUMINARY_FUNCTION bool camera_aperture_interaction(
  CameraSimulationState& state, const PathID& path_id, const uint32_t sample_id, const DeviceCameraInterface interface,
  const vec3 semi_circle_center, float& dist) {
  float edge_dist = FLT_MAX;
  vec3 aperture_hit_point;
  float2 edge_normal;
  if (camera_simulation_intersect_aperture(state.origin, state.ray, dist, edge_dist, aperture_hit_point, edge_normal))
    return true;

  if (device.camera.enable_diffraction && edge_dist != FLT_MAX && state.has_retro_reflected == false) {
    state.origin                    = aperture_hit_point;
    const float2 random_diffraction = random_2D(RANDOM_TARGET_LENS_DIFFRACTION + sample_id, path_id);

    float pdf = 1.0f;
    state.ray = camera_aperture_diffraction_sample(state.ray, edge_dist, edge_normal, state.wavelength, random_diffraction, pdf);
    state.throughput *= pdf;
    state.probability_density *= pdf;

    dist = camera_simulation_interface_intersection(state, semi_circle_center, interface.radius, interface.cylindrical_radius);
  }

  return (dist == FLT_MAX);
}

LUMINARY_FUNCTION bool camera_simulation_interaction(
  CameraSimulationState& state, const PathID& path_id, const uint32_t sample_id, const DeviceCameraInterface interface,
  const vec3 semi_circle_center, float& dist) {
  dist = camera_simulation_interface_intersection(state, semi_circle_center, interface.radius, interface.cylindrical_radius);

  if (dist == FLT_MAX)
    return true;

  if (camera_aperture_interaction(state, path_id, sample_id, interface, semi_circle_center, dist))
    return true;

  return false;
}

template <bool ALLOW_REFLECTIONS, bool SPECTRAL_RENDERING>
LUMINARY_FUNCTION int32_t camera_simulation_step(
  CameraSimulationState& state, const uint32_t iteration, const int32_t interface_id, const PathID& path_id, const uint32_t sample_id) {
  const DeviceCameraInterface interface = device.ptrs.camera_interfaces[interface_id];

  const float center            = (interface.radius != FLT_MAX) ? interface.vertex - interface.radius : interface.vertex;
  const vec3 semi_circle_center = get_vector(0.0f, 0.0f, center);

  float dist = FLT_MAX;
  if (camera_simulation_interaction(state, path_id, sample_id, interface, semi_circle_center, dist)) {
    state.throughput = 0.0f;
    return 0;
  }

  if (camera_simulation_intersect_medium_cylinder(state.origin, state.ray, state.throughput, dist, state.cylindrical_radius, state.ior)) {
    state.has_forward_reflected = true;
    if (camera_simulation_interaction(state, path_id, sample_id, interface, semi_circle_center, dist)) {
      state.throughput = 0.0f;
      return 0;
    }
  }

  const int32_t medium_id         = state.is_forward ? interface_id + 1 : interface_id;
  const DeviceCameraMedium medium = device.ptrs.camera_media[medium_id];
  const float medium_ior          = camera_medium_get_ior<SPECTRAL_RENDERING>(medium, state.wavelength);

  state.origin = add_vector(state.origin, scale_vector(state.ray, dist));

  vec3 normal = (interface.radius != FLT_MAX) ? normalize_vector(sub_vector(state.origin, semi_circle_center))
                                              : get_vector(0.0f, 0.0f, (state.origin.z > center) ? 1.0f : -1.0f);

  bool interface_curving_away = interface.radius < 0.0f;
  interface_curving_away ^= state.is_forward == false;

  if (interface_curving_away == false && interface.radius != FLT_MAX) {
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
  state.is_forward          = state.ray.z >= 0.0f;
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

  const vec3 sensor_point = camera_sample_sensor<false>(path_id);

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
