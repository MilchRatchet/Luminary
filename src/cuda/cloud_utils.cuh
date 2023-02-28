#ifndef CLOUD_UTILS_CUH
#define CLOUD_UTILS_CUH

////////////////////////////////////////////////////////////////////
// Defines
////////////////////////////////////////////////////////////////////

// It is important that extinction >= scattering to not amplify the energy in the system
#define CLOUD_SCATTERING_DENSITY (1000.0f * 0.1f * 0.9f)
#define CLOUD_EXTINCTION_DENSITY (1000.0f * 0.1f)
#define CLOUD_EXTINCTION_STEP_SIZE 0.01f
#define CLOUD_EXTINCTION_STEP_MULTIPLY 1.5f
#define CLOUD_EXTINCTION_STEP_MAX 0.75f

#define CLOUD_GRADIENT_STRATUS make_float4(0.01f, 0.1f, 0.11f, 0.2f)
#define CLOUD_GRADIENT_STRATOCUMULUS make_float4(0.01f, 0.08f, 0.3f, 0.4f)
#define CLOUD_GRADIENT_CUMULUS make_float4(0.01f, 0.06f, 0.75f, 0.95f)

#define CLOUD_WIND_DIR get_vector(1.0f, 0.0f, 0.0f)
#define CLOUD_WIND_SKEW 0.7f
#define CLOUD_WIND_SKEW_WEATHER 2.5f

// CLOUD_SCATTERING_OCTAVES must be larger than 0, and EXTINCTION_FACTOR >= SCATTERING_FACTOR
#define CLOUD_SCATTERING_OCTAVES 9
#define CLOUD_OCTAVE_SCATTERING_FACTOR 0.5f
#define CLOUD_OCTAVE_EXTINCTION_FACTOR 0.5f
#define CLOUD_OCTAVE_PHASE_FACTOR 0.5f

////////////////////////////////////////////////////////////////////
// Structs
////////////////////////////////////////////////////////////////////

struct CloudExtinctionOctaves {
  float E[CLOUD_SCATTERING_OCTAVES];
} typedef CloudExtinctionOctaves;

struct CloudPhaseOctaves {
  float P[CLOUD_SCATTERING_OCTAVES];
} typedef CloudPhaseOctaves;

////////////////////////////////////////////////////////////////////
// Utility functions
////////////////////////////////////////////////////////////////////

__device__ float cloud_gradient(float4 gradient, float height) {
  return smoothstep(height, gradient.x, gradient.y) - smoothstep(height, gradient.z, gradient.w);
}

__device__ float cloud_height(const vec3 pos) {
  return (sky_height(pos) - world_to_sky_scale(device.scene.sky.cloud.height_min))
         / world_to_sky_scale(device.scene.sky.cloud.height_max - device.scene.sky.cloud.height_min);
}

__device__ vec3 cloud_weather(vec3 pos, const float height) {
  pos.x += device.scene.sky.cloud.offset_x;
  pos.z += device.scene.sky.cloud.offset_z;

  vec3 weather_pos = pos;
  weather_pos      = add_vector(weather_pos, scale_vector(CLOUD_WIND_DIR, CLOUD_WIND_SKEW_WEATHER * height));
  weather_pos      = scale_vector(weather_pos, 0.012f * device.scene.sky.cloud.noise_weather_scale);

  float4 weather = tex2D<float4>(device.ptrs.cloud_noise[2], weather_pos.x, weather_pos.z);

  weather.x = powf(fabsf(weather.x), __saturatef(remap(height * 3.0f, 0.7f, 0.8f, 1.0f, lerp(1.0f, 0.5f, device.scene.sky.cloud.anvil))));

  weather.x = __saturatef(
    remap(weather.x * device.scene.sky.cloud.coverage, 0.0f, 1.0f, __saturatef(device.scene.sky.cloud.coverage_min - 1.0f), 1.0f));

  return get_vector(weather.x, weather.y, weather.z);
}

__device__ vec3 cloud_weather_type(const vec3 weather) {
  const float small  = 1.0f - __saturatef(weather.y * 2.0f);
  const float medium = 1.0f - fabsf(weather.y - 0.5f) * 2.0f;
  const float large  = __saturatef(weather.y - 0.5f) * 2.0f;

  return get_vector(small, medium, large);
}

__device__ float4 cloud_density_height_gradient_type(const vec3 weather) {
  const vec3 weather_type = cloud_weather_type(weather);

  const float4 stratus       = CLOUD_GRADIENT_STRATUS;
  const float4 stratocumulus = CLOUD_GRADIENT_STRATOCUMULUS;
  const float4 cumulus       = CLOUD_GRADIENT_CUMULUS;

  return make_float4(
    weather_type.x * stratus.x + weather_type.y * stratocumulus.x + weather_type.z * cumulus.x,
    weather_type.x * stratus.y + weather_type.y * stratocumulus.y + weather_type.z * cumulus.y,
    weather_type.x * stratus.z + weather_type.y * stratocumulus.z + weather_type.z * cumulus.z,
    weather_type.x * stratus.w + weather_type.y * stratocumulus.w + weather_type.z * cumulus.w);
}

__device__ float cloud_density_height_gradient(const float height, const vec3 weather) {
  const float4 gradient = cloud_density_height_gradient_type(weather);

  return cloud_gradient(gradient, height);
}

__device__ float cloud_weather_wetness(vec3 weather) {
  return lerp(1.0f, 1.0f - device.scene.sky.cloud.wetness, __saturatef(weather.z));
}

__device__ float cloud_dual_lobe_henvey_greenstein(const float cos_angle, const float factor) {
  const float mie0 = henvey_greenstein(cos_angle, device.scene.sky.cloud.forward_scattering * factor);
  const float mie1 = henvey_greenstein(cos_angle, device.scene.sky.cloud.backward_scattering * factor);
  return lerp(mie0, mie1, device.scene.sky.cloud.lobe_lerp);
}

__device__ float cloud_powder(const float density, const float step_size) {
  const float powder = 1.0f - expf(-density * step_size);
  return lerp(1.0f, __saturatef(powder), device.scene.sky.cloud.powder);
}

__device__ float2 cloud_get_intersection(const vec3 origin, const vec3 ray, const float limit) {
  const float h_min = world_to_sky_scale(device.scene.sky.cloud.height_min) + SKY_EARTH_RADIUS;
  const float h_max = world_to_sky_scale(device.scene.sky.cloud.height_max) + SKY_EARTH_RADIUS;

  const float height = get_length(origin);

  float distance;
  float start = 0.0f;
  if (height > h_max) {
    const float max_dist = sph_ray_int_p0(ray, origin, h_max);

    start = max_dist;

    const float max_dist_back = sph_ray_int_back_p0(ray, origin, h_max);
    const float min_dist      = sph_ray_int_p0(ray, origin, h_min);

    distance = fminf(min_dist, max_dist_back) - start;
  }
  else if (height < h_min) {
    const float min_dist = sph_ray_int_p0(ray, origin, h_min);
    const float max_dist = sph_ray_int_p0(ray, origin, h_max);

    start    = min_dist;
    distance = max_dist - start;
  }
  else {
    const float min_dist = sph_ray_int_p0(ray, origin, h_min);
    const float max_dist = sph_ray_int_p0(ray, origin, h_max);
    distance             = fminf(min_dist, max_dist);
  }

  const float earth_hit = sph_ray_int_p0(ray, origin, SKY_EARTH_RADIUS);
  distance              = fminf(distance, fminf(earth_hit, limit) - start);
  distance              = fmaxf(0.0f, distance);

  return make_float2(start, distance);
}

////////////////////////////////////////////////////////////////////
// Density function
////////////////////////////////////////////////////////////////////

__device__ float cloud_base_density(const vec3 pos, const float height, const vec3 weather) {
  vec3 shape_pos = pos;
  shape_pos      = add_vector(shape_pos, scale_vector(CLOUD_WIND_DIR, CLOUD_WIND_SKEW * height));
  shape_pos      = scale_vector(shape_pos, 0.4f * device.scene.sky.cloud.noise_shape_scale);

  float4 shape = tex3D<float4>(device.ptrs.cloud_noise[0], shape_pos.x, shape_pos.y, shape_pos.z);

  const vec3 gradient = get_vector(
    cloud_gradient(CLOUD_GRADIENT_STRATUS, height), cloud_gradient(CLOUD_GRADIENT_STRATOCUMULUS, height),
    cloud_gradient(CLOUD_GRADIENT_CUMULUS, height));

  shape.y *= gradient.x * 0.2f;
  shape.z *= gradient.y * 0.2f;
  shape.w *= gradient.z * 0.2f;

  float density_gradient = cloud_density_height_gradient(height, weather);

  float density = (shape.x + shape.y + shape.z + shape.w) * 0.8f * density_gradient;
  density       = powf(fabsf(density), __saturatef(height * 6.0f));

  density = smoothstep(density, 0.25f, 1.1f);

  density = __saturatef(density - (1.0f - weather.x)) * weather.x;

  return density;
}

__device__ float cloud_erode_density(const vec3 pos, float density, const float height, const vec3 weather) {
  vec3 curl_pos = pos;
  curl_pos      = scale_vector(curl_pos, 3.0f * device.scene.sky.cloud.noise_curl_scale);

  const float4 curl = tex2D<float4>(device.ptrs.cloud_noise[3], curl_pos.x, curl_pos.z);

  vec3 curl_shift = get_vector(curl.x, curl.y, curl.z);
  curl_shift      = sub_vector(curl_shift, get_vector(0.5f, 0.5f, 0.5f));
  curl_shift      = scale_vector(curl_shift, 2.0f * 0.55f * height);

  vec3 detail_pos = pos;
  detail_pos      = add_vector(detail_pos, curl_shift);
  detail_pos      = add_vector(detail_pos, scale_vector(CLOUD_WIND_DIR, CLOUD_WIND_SKEW * height));
  detail_pos      = scale_vector(detail_pos, 2.0f * device.scene.sky.cloud.noise_detail_scale);

  const float4 detail = tex3D<float4>(device.ptrs.cloud_noise[1], detail_pos.x, detail_pos.y, detail_pos.z);

  float detail_fbm = __saturatef(detail.x * 0.625f + detail.y * 0.25f + detail.z * 0.125f);

  float noise_modifier = lerp(1.0f - detail_fbm, detail_fbm, __saturatef(height * 10.0f));

  density = remap(density, noise_modifier * 0.2f, 1.0f, 0.0f, 1.0f);

  return density;
}

__device__ float cloud_density(vec3 pos, const float height, const vec3 weather) {
  pos.x += device.scene.sky.cloud.offset_x;
  pos.z += device.scene.sky.cloud.offset_z;

  float density = cloud_base_density(pos, height, weather);

  if (density > 0.0f) {
    density = cloud_erode_density(pos, density, height, weather);
  }

  return fmaxf(density * device.scene.sky.cloud.density, 0.0f);
}

#endif /* CLOUD_UTILS_CUH */
