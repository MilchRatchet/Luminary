#ifndef CLOUD_UTILS_CUH
#define CLOUD_UTILS_CUH

////////////////////////////////////////////////////////////////////
// Defines
////////////////////////////////////////////////////////////////////

// It is important that extinction >= scattering to not amplify the energy in the system
#define CLOUD_SCATTERING_DENSITY (1000.0f * 0.1f * 0.9f)
#define CLOUD_EXTINCTION_DENSITY (1000.0f * 0.1f)

// Low-level clouds
#define CLOUD_GRADIENT_STRATUS make_float4(0.01f, 0.15f, 0.17f, 0.3f)
#define CLOUD_GRADIENT_STRATOCUMULUS make_float4(0.01f, 0.12f, 0.45f, 0.6f)
#define CLOUD_GRADIENT_CUMULUS make_float4(0.01f, 0.06f, 0.8f, 0.99f)

// Mid-level clouds
#define CLOUD_GRADIENT_ALTOSTRATUS make_float4(0.01f, 0.20f, 0.80f, 0.95f)
#define CLOUD_GRADIENT_ALTOCUMULUS make_float4(0.25f, 0.30f, 0.60f, 0.75f)

#define CLOUD_WIND_DIR get_vector(1.0f, 0.0f, 0.0f)
#define CLOUD_WIND_SKEW 0.7f
#define CLOUD_WIND_SKEW_WEATHER 2.5f

// EXTINCTION_FACTOR >= SCATTERING_FACTOR
// Extinction factor must be 0.5f as the code is optimized with that in mind
#define CLOUD_OCTAVE_SCATTERING_FACTOR 0.5f
#define CLOUD_OCTAVE_EXTINCTION_FACTOR 0.5f
#define CLOUD_OCTAVE_PHASE_FACTOR 0.5f

#define CLOUD_WEATHER_CUTOFF 0.05f

////////////////////////////////////////////////////////////////////
// Structs
////////////////////////////////////////////////////////////////////

enum CloudLayerType { CLOUD_LAYER_LOW = 0, CLOUD_LAYER_MID = 1, CLOUD_LAYER_TOP = 2 } typedef CloudLayerType;

struct CloudWeather {
  float coverage_low;
  float type_low;
  float coverage_mid;
  float type_mid;
} typedef CloudWeather;

struct CloudRenderResult {
  RGBF scattered_light;
  float transmittance;
  float hit_dist;
} typedef CloudRenderResult;

////////////////////////////////////////////////////////////////////
// Utility functions
////////////////////////////////////////////////////////////////////

__device__ float cloud_gradient(float4 gradient, float height) {
  return smoothstep(height, gradient.x, gradient.y) - smoothstep(height, gradient.z, gradient.w);
}

__device__ float cloud_height(const vec3 pos, const CloudLayerType layer) {
  switch (layer) {
    case CLOUD_LAYER_LOW:
      return (sky_height(pos) - world_to_sky_scale(device.scene.sky.cloud.height_low_min))
             / world_to_sky_scale(device.scene.sky.cloud.height_low_max - device.scene.sky.cloud.height_low_min);
    case CLOUD_LAYER_MID:
      return (sky_height(pos) - world_to_sky_scale(device.scene.sky.cloud.height_mid_min))
             / world_to_sky_scale(device.scene.sky.cloud.height_mid_max - device.scene.sky.cloud.height_mid_min);
    case CLOUD_LAYER_TOP:
      return (sky_height(pos) - world_to_sky_scale(device.scene.sky.cloud.height_top_min))
             / world_to_sky_scale(device.scene.sky.cloud.height_top_max - device.scene.sky.cloud.height_top_min);
    default:
      return FLT_MAX;
  }
}

__device__ CloudWeather cloud_weather(vec3 pos, const float height) {
  pos.x += device.scene.sky.cloud.offset_x;
  pos.z += device.scene.sky.cloud.offset_z;

  vec3 low_pos = pos;
  low_pos      = add_vector(low_pos, scale_vector(CLOUD_WIND_DIR, CLOUD_WIND_SKEW_WEATHER * height));
  low_pos      = scale_vector(low_pos, 0.012f * device.scene.sky.cloud.noise_weather_scale);

  float4 tex = tex2D<float4>(device.ptrs.cloud_noise[2], low_pos.x, low_pos.z);

  CloudWeather weather;
  weather.coverage_low = __saturatef(remap(tex.x * device.scene.sky.cloud.coverage, 0.0f, 1.0f, device.scene.sky.cloud.coverage_min, 1.0f));
  weather.type_low     = __saturatef(remap(tex.y * device.scene.sky.cloud.type, 0.0f, 1.0f, device.scene.sky.cloud.type_min, 1.0f));
  weather.coverage_mid = __saturatef(remap(tex.z * device.scene.sky.cloud.coverage, 0.0f, 1.0f, device.scene.sky.cloud.coverage_min, 1.0f));
  weather.type_mid     = __saturatef(remap(tex.w * device.scene.sky.cloud.type, 0.0f, 1.0f, device.scene.sky.cloud.type_min, 1.0f));

  return weather;
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

////////////////////////////////////////////////////////////////////
// Low level cloud functions
////////////////////////////////////////////////////////////////////

__device__ vec3 cloud_weather_type_low_level(const CloudWeather weather) {
  const float stratus       = 1.0f - __saturatef(weather.type_low * 2.0f);
  const float stratocumulus = 1.0f - fabsf(2.0f * weather.type_low - 1.0f);
  const float cumulus       = __saturatef(2.0f * weather.type_low - 1.0f);

  return get_vector(stratus, stratocumulus, cumulus);
}

__device__ float4 cloud_density_height_gradient_type_low_level(const CloudWeather weather) {
  const vec3 weather_type = cloud_weather_type_low_level(weather);

  const float4 stratus       = CLOUD_GRADIENT_STRATUS;
  const float4 stratocumulus = CLOUD_GRADIENT_STRATOCUMULUS;
  const float4 cumulus       = CLOUD_GRADIENT_CUMULUS;

  return make_float4(
    weather_type.x * stratus.x + weather_type.y * stratocumulus.x + weather_type.z * cumulus.x,
    weather_type.x * stratus.y + weather_type.y * stratocumulus.y + weather_type.z * cumulus.y,
    weather_type.x * stratus.z + weather_type.y * stratocumulus.z + weather_type.z * cumulus.z,
    weather_type.x * stratus.w + weather_type.y * stratocumulus.w + weather_type.z * cumulus.w);
}

__device__ float cloud_density_height_gradient_low_level(const float height, const CloudWeather weather) {
  const float4 gradient = cloud_density_height_gradient_type_low_level(weather);

  return cloud_gradient(gradient, height);
}

////////////////////////////////////////////////////////////////////
// Mid level cloud functions
////////////////////////////////////////////////////////////////////

__device__ vec3 cloud_weather_type_mid_level(const CloudWeather weather) {
  const float altostratus = 1.0f - __saturatef(weather.type_mid);
  const float altocumulus = __saturatef(weather.type_mid);

  return get_vector(altostratus, altocumulus, 0.0f);
}

__device__ float4 cloud_density_height_gradient_type_mid_level(const CloudWeather weather) {
  const vec3 weather_type = cloud_weather_type_mid_level(weather);

  const float4 altostratus = CLOUD_GRADIENT_ALTOSTRATUS;
  const float4 altocumulus = CLOUD_GRADIENT_ALTOCUMULUS;

  return make_float4(
    weather_type.x * altostratus.x + weather_type.y * altocumulus.x, weather_type.x * altostratus.y + weather_type.y * altocumulus.y,
    weather_type.x * altostratus.z + weather_type.y * altocumulus.z, weather_type.x * altostratus.w + weather_type.y * altocumulus.w);
}

__device__ float cloud_density_height_gradient_mid_level(const float height, const CloudWeather weather) {
  const float4 gradient = cloud_density_height_gradient_type_mid_level(weather);

  return cloud_gradient(gradient, height);
}

////////////////////////////////////////////////////////////////////
// Cloud layer intersection functions
////////////////////////////////////////////////////////////////////

__device__ float2 cloud_get_layer_intersection(const vec3 origin, const vec3 ray, const float limit, const float hmin, const float hmax) {
  const float height = get_length(origin);

  const float dist_hmax = sph_ray_int_p0(ray, origin, hmax);
  const float dist_hmin = sph_ray_int_p0(ray, origin, hmin);

  float start;

  if (height > hmax) {
    start = dist_hmax;
  }
  else if (height < hmin) {
    start = dist_hmin;
  }
  else {
    start = 0.0f;
  }

  const float end_1 = (height < hmin) ? dist_hmax : dist_hmin;
  const float end_2 = (height > hmax) ? sph_ray_int_back_p0(ray, origin, hmax) : dist_hmax;

  const float end_dist  = fminf(end_1, end_2);
  const float earth_hit = sph_ray_int_p0(ray, origin, SKY_EARTH_RADIUS);
  const float distance  = fminf(earth_hit, fminf(limit, end_dist)) - start;

  if (distance < 0.0f) {
    start = FLT_MAX;
  }

  return make_float2(start, distance);
}

__device__ float2 cloud_get_lowlayer_intersection(const vec3 origin, const vec3 ray, const float limit) {
  if (!device.scene.sky.cloud.layer_low) {
    return make_float2(FLT_MAX, 0.0f);
  }

  const float hmin = world_to_sky_scale(device.scene.sky.cloud.height_low_min) + SKY_EARTH_RADIUS;
  const float hmax = world_to_sky_scale(device.scene.sky.cloud.height_low_max) + SKY_EARTH_RADIUS;

  return cloud_get_layer_intersection(origin, ray, limit, hmin, hmax);
}

__device__ float2 cloud_get_midlayer_intersection(const vec3 origin, const vec3 ray, const float limit) {
  if (!device.scene.sky.cloud.layer_mid) {
    return make_float2(FLT_MAX, 0.0f);
  }

  const float hmin = world_to_sky_scale(device.scene.sky.cloud.height_mid_min) + SKY_EARTH_RADIUS;
  const float hmax = world_to_sky_scale(device.scene.sky.cloud.height_mid_max) + SKY_EARTH_RADIUS;

  return cloud_get_layer_intersection(origin, ray, limit, hmin, hmax);
}

__device__ float2 cloud_get_toplayer_intersection(const vec3 origin, const vec3 ray, const float limit) {
  if (!device.scene.sky.cloud.layer_top) {
    return make_float2(FLT_MAX, 0.0f);
  }

  const float hmin = device.scene.sky.cloud.height_top_min + SKY_EARTH_RADIUS;
  const float hmax = device.scene.sky.cloud.height_top_max + SKY_EARTH_RADIUS;

  return cloud_get_layer_intersection(origin, ray, limit, hmin, hmax);
}

////////////////////////////////////////////////////////////////////
// Density function
////////////////////////////////////////////////////////////////////

__device__ float cloud_base_density_low(const vec3 pos, const float height, const CloudWeather weather, float mip_bias) {
  mip_bias += device.scene.sky.cloud.mipmap_bias;
  mip_bias += (is_first_ray()) ? 0.0f : 1.0f;

  float density = 0.0f;

  if (device.scene.sky.cloud.layer_low) {
    vec3 shape_pos = pos;
    shape_pos      = add_vector(shape_pos, scale_vector(CLOUD_WIND_DIR, CLOUD_WIND_SKEW * height));
    shape_pos      = scale_vector(shape_pos, 0.4f * device.scene.sky.cloud.noise_shape_scale);

    float4 shape = tex3DLod<float4>(device.ptrs.cloud_noise[0], shape_pos.x, shape_pos.y, shape_pos.z, mip_bias);

    float shape_sum = shape.x * 5.0f;
    shape_sum += shape.y * cloud_gradient(CLOUD_GRADIENT_STRATUS, height);
    shape_sum += shape.z * cloud_gradient(CLOUD_GRADIENT_STRATOCUMULUS, height);
    shape_sum += shape.w * cloud_gradient(CLOUD_GRADIENT_CUMULUS, height);
    shape_sum *= 0.16f;

    const float density_gradient = cloud_density_height_gradient_low_level(height, weather);

    density = shape_sum * density_gradient;
    density = powf(fabsf(density), __saturatef(height * 6.0f));
    density = smoothstep(density, 0.25f, 1.1f);
    density = __saturatef(density - (1.0f - weather.coverage_low)) * weather.coverage_low;
  }

  return density;
}

__device__ float cloud_base_density_mid(const vec3 pos, const float height, const CloudWeather weather, float mip_bias) {
  mip_bias += device.scene.sky.cloud.mipmap_bias;
  mip_bias += (is_first_ray()) ? 0.0f : 1.0f;

  float density = 0.0f;

  if (device.scene.sky.cloud.layer_mid) {
    vec3 shape_pos = pos;
    shape_pos      = add_vector(shape_pos, scale_vector(CLOUD_WIND_DIR, CLOUD_WIND_SKEW * height));
    shape_pos      = scale_vector(shape_pos, 0.2f * device.scene.sky.cloud.noise_shape_scale);

    float4 shape = tex3DLod<float4>(device.ptrs.cloud_noise[0], shape_pos.x, shape_pos.y, shape_pos.z, mip_bias);

    float shape_sum = (shape.x + shape.y * 6.0f + shape.w + shape.z) * 0.1f;

    const float density_gradient = cloud_density_height_gradient_mid_level(height, weather);

    density = shape_sum * density_gradient;
    density = remap(weather.type_mid, 0.0f, 1.0f, smoothstep(density, -0.1f, 1.0f), smoothstep(density, 0.50f, 1.0f));
    density = __saturatef(density) * weather.coverage_mid;
  }

  return density;
}

__device__ float cloud_erode_density(const vec3 pos, float density, const float height, const CloudWeather weather, float mip_bias) {
  vec3 curl_pos = pos;
  curl_pos      = scale_vector(curl_pos, 3.0f * device.scene.sky.cloud.noise_curl_scale);

  mip_bias += device.scene.sky.cloud.mipmap_bias;
  mip_bias += (is_first_ray()) ? 0.0f : 1.0f;

  const float4 curl = tex2DLod<float4>(device.ptrs.cloud_noise[3], curl_pos.x, curl_pos.z, mip_bias);

  vec3 curl_shift = get_vector(curl.x, curl.y, curl.z);
  curl_shift      = sub_vector(curl_shift, get_vector(0.5f, 0.5f, 0.5f));
  curl_shift      = scale_vector(curl_shift, 2.0f * 0.55f * height);

  vec3 detail_pos = pos;
  detail_pos      = add_vector(detail_pos, curl_shift);
  detail_pos      = add_vector(detail_pos, scale_vector(CLOUD_WIND_DIR, CLOUD_WIND_SKEW * height));
  detail_pos      = scale_vector(detail_pos, 2.0f * device.scene.sky.cloud.noise_detail_scale);

  const float4 detail = tex3DLod<float4>(device.ptrs.cloud_noise[1], detail_pos.x, detail_pos.y, detail_pos.z, mip_bias);

  float detail_fbm = __saturatef(detail.x * 0.625f + detail.y * 0.25f + detail.z * 0.125f);

  float noise_modifier = lerp(1.0f - detail_fbm, detail_fbm, __saturatef(height * 10.0f));

  density = remap(density, noise_modifier * device.scene.sky.cloud.erosion, 1.0f, 0.0f, 1.0f);

  return density;
}

__device__ float cloud_density(vec3 pos, const float height, const CloudWeather weather, const float mip_bias, const CloudLayerType layer) {
  pos.x += device.scene.sky.cloud.offset_x;
  pos.z += device.scene.sky.cloud.offset_z;

  float density;

  switch (layer) {
    case CLOUD_LAYER_LOW:
      density = cloud_base_density_low(pos, height, weather, mip_bias);
      break;
    case CLOUD_LAYER_MID:
      density = cloud_base_density_mid(pos, height, weather, mip_bias);
      break;
    case CLOUD_LAYER_TOP:
    default:
      density = 0.0f;
      break;
  }

  if (density > 0.0f) {
    density = cloud_erode_density(pos, density, height, weather, mip_bias);
  }

  return fmaxf(density * device.scene.sky.cloud.density, 0.0f);
}

////////////////////////////////////////////////////////////////////
// Cutoff function
////////////////////////////////////////////////////////////////////

__device__ bool cloud_significant_point(const float height, const CloudWeather weather, const CloudLayerType layer) {
  switch (layer) {
    case CLOUD_LAYER_LOW: {
      const float4 type = cloud_density_height_gradient_type_low_level(weather);
      return (weather.coverage_low > CLOUD_WEATHER_CUTOFF) && (type.x < height) && (type.w > height);
    }
    case CLOUD_LAYER_MID: {
      const float4 type = cloud_density_height_gradient_type_mid_level(weather);
      return (weather.coverage_mid > CLOUD_WEATHER_CUTOFF) && (type.x < height) && (type.w > height);
    }
    case CLOUD_LAYER_TOP:
    default:
      return false;
  }
}

#endif /* CLOUD_UTILS_CUH */
