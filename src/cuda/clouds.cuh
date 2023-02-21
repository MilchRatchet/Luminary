#ifndef CU_CLOUDS_H
#define CU_CLOUDS_H

#include "math.cuh"
#include "sky.cuh"
#include "utils.cuh"

//
// The code of this file is based on the cloud rendering in https://github.com/turanszkij/WickedEngine.
// It follows the basic ideas of using raymarching with noise based density.
//

////////////////////////////////////////////////////////////////////
// Defines
////////////////////////////////////////////////////////////////////

#define CLOUD_SCATTERING_DENSITY (1000.0f * 0.9f)
#define CLOUD_EXTINCTION_DENSITY (1000.0f * 0.01f)
#define CLOUD_EXTINCTION_STEP_SIZE 0.01f
#define CLOUD_EXTINCTION_STEP_MULTIPLY 1.5f
#define CLOUD_EXTINCTION_STEP_MAX 0.75f

#define CLOUD_GRADIENT_STRATUS make_float4(0.01f, 0.1f, 0.11f, 0.2f)
#define CLOUD_GRADIENT_STRATOCUMULUS make_float4(0.01f, 0.08f, 0.3f, 0.4f)
#define CLOUD_GRADIENT_CUMULUS make_float4(0.01f, 0.06f, 0.75f, 0.95f)

////////////////////////////////////////////////////////////////////
// Utility functions
////////////////////////////////////////////////////////////////////

__device__ float cloud_gradient(float4 gradient, float height) {
  return smoothstep(height, gradient.x, gradient.y) - smoothstep(height, gradient.z, gradient.w);
}

__device__ float cloud_height(const vec3 pos) {
  return (sky_height(pos) - world_to_sky_scale(device_scene.sky.cloud.height_min))
         / world_to_sky_scale(device_scene.sky.cloud.height_max - device_scene.sky.cloud.height_min);
}

__device__ vec3 cloud_weather(vec3 pos, float height) {
  pos.x += device_scene.sky.cloud.offset_x;
  pos.z += device_scene.sky.cloud.offset_z;

  vec3 weather_pos = pos;
  weather_pos      = scale_vector(weather_pos, 0.006f * device_scene.sky.cloud.noise_weather_scale);

  float4 weather = tex2D<float4>(device.cloud_noise[2], weather_pos.x, weather_pos.z);

  weather.x = powf(fabsf(weather.x), __saturatef(remap(height * 3.0f, 0.7f, 0.8f, 1.0f, lerp(1.0f, 0.5f, device_scene.sky.cloud.anvil))));

  weather.x = __saturatef(
    remap(weather.x * device_scene.sky.cloud.coverage, 0.0f, 1.0f, __saturatef(device_scene.sky.cloud.coverage_min - 1.0f), 1.0f));

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
  return lerp(1.0f, 1.0f - device_scene.sky.cloud.wetness, __saturatef(weather.z));
}

__device__ float cloud_dual_lobe_henvey_greenstein(float cos_angle, float g0, float g1, float w) {
  const float mie0 = henvey_greenstein(cos_angle, g0);
  const float mie1 = henvey_greenstein(cos_angle, g1);
  return lerp(mie0, mie1, w);
}

__device__ float cloud_powder(const float density, const float step_size) {
  const float powder = 1.0f - expf(-density * step_size);
  return lerp(1.0f, __saturatef(powder), device_scene.sky.cloud.powder);
}

__device__ float2 cloud_get_intersection(const vec3 origin, const vec3 ray, const float limit) {
  const float h_min = world_to_sky_scale(device_scene.sky.cloud.height_min) + SKY_EARTH_RADIUS;
  const float h_max = world_to_sky_scale(device_scene.sky.cloud.height_max) + SKY_EARTH_RADIUS;

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
  shape_pos      = scale_vector(shape_pos, 0.4f * device_scene.sky.cloud.noise_shape_scale);

  float4 shape = tex3D<float4>(device.cloud_noise[0], shape_pos.x, shape_pos.y, shape_pos.z);

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
  curl_pos      = scale_vector(curl_pos, 3.0f * device_scene.sky.cloud.noise_curl_scale);

  const float4 curl = tex2D<float4>(device.cloud_noise[3], curl_pos.x, curl_pos.z);

  vec3 curl_shift = get_vector(curl.x, curl.y, curl.z);
  curl_shift      = sub_vector(curl_shift, get_vector(0.5f, 0.5f, 0.5f));
  curl_shift      = scale_vector(curl_shift, 2.0f * 0.55f * height);

  vec3 detail_pos = pos;
  detail_pos      = add_vector(detail_pos, curl_shift);
  detail_pos      = scale_vector(detail_pos, 2.0f * device_scene.sky.cloud.noise_detail_scale);

  const float4 detail = tex3D<float4>(device.cloud_noise[1], detail_pos.x, detail_pos.y, detail_pos.z);

  float detail_fbm = __saturatef(detail.x * 0.625f + detail.y * 0.25f + detail.z * 0.125f);

  float noise_modifier = lerp(1.0f - detail_fbm, detail_fbm, __saturatef(height * 10.0f));

  density = remap(density, noise_modifier * 0.2f, 1.0f, 0.0f, 1.0f);

  return density;
}

__device__ float cloud_density(vec3 pos, const float height, const vec3 weather) {
  pos.x += device_scene.sky.cloud.offset_x;
  pos.z += device_scene.sky.cloud.offset_z;

  float density = cloud_base_density(pos, height, weather);

  if (density > 0.0f) {
    density = cloud_erode_density(pos, density, height, weather);
  }

  return fmaxf(density * device_scene.sky.cloud.density, 0.0f);
}

////////////////////////////////////////////////////////////////////
// Integrator functions
////////////////////////////////////////////////////////////////////

__device__ float cloud_extinction(vec3 origin, vec3 ray) {
  float step_size  = CLOUD_EXTINCTION_STEP_SIZE;
  float reach      = step_size * 0.5f;
  float extinction = 0.0f;

  for (int i = 0; i < device_scene.sky.cloud.shadow_steps; i++) {
    const vec3 pos = add_vector(origin, scale_vector(ray, reach));

    const float height = cloud_height(pos);

    if (height > 1.0f)
      break;

    if (height < 0.0f) {
      const float h_min = world_to_sky_scale(device_scene.sky.cloud.height_min) + SKY_EARTH_RADIUS;
      reach += sph_ray_int_p0(ray, pos, h_min);
      continue;
    }

    const vec3 weather = cloud_weather(pos, height);

    if (weather.x > 0.05f) {
      const float density = CLOUD_EXTINCTION_DENSITY * cloud_density(pos, height, weather);

      extinction -= density * step_size;
    }

    step_size *= CLOUD_EXTINCTION_STEP_MULTIPLY;
    step_size = fminf(CLOUD_EXTINCTION_STEP_MAX, step_size);
    reach += step_size;
  }

  return expf(extinction);
}

__device__ RGBAF cloud_render(const vec3 origin, const vec3 ray, const float start, float dist) {
  dist = fminf(30.0f, dist);

  const int step_count = device_scene.sky.cloud.steps * __saturatef(dist / 15.0f);

  const int big_step_mult = 2;
  const float big_step    = big_step_mult;

  const float step_size = dist / step_count;

  float reach = start + (white_noise() + 0.1f) * step_size;

  float transmittance = 1.0f;
  RGBF scatteredLight = get_color(0.0f, 0.0f, 0.0f);

  const float sun_light_angle = sample_sphere_solid_angle(device_sun, SKY_SUN_RADIUS, add_vector(origin, scale_vector(ray, reach)));

  for (int i = 0; i < step_count; i++) {
    vec3 pos = add_vector(origin, scale_vector(ray, reach));

    const float height = cloud_height(pos);

    if (height < 0.0f || height > 1.0f) {
      break;
    }

    vec3 weather = cloud_weather(pos, height);

    if (weather.x < 0.05f) {
      i += big_step_mult - 1;
      reach += step_size * big_step;
      continue;
    }

    float density = cloud_density(pos, height, weather);

    if (density > 0.0f) {
      // Celestial light (prefer sun but use the moon if possible)
      const int sun_visible = !sph_ray_hit_p0(normalize_vector(sub_vector(device_sun, pos)), pos, SKY_EARTH_RADIUS);

      RGBF sun_color;

      if (sun_visible) {
        const vec3 ray_sun = sample_sphere(device_sun, SKY_SUN_RADIUS, pos);

        const float cos_angle_sun  = dot_product(ray, ray_sun);
        const float scattering_sun = cloud_dual_lobe_henvey_greenstein(
          cos_angle_sun, device_scene.sky.cloud.forward_scattering, device_scene.sky.cloud.backward_scattering,
          device_scene.sky.cloud.lobe_lerp);

        const float extinction_sun = cloud_extinction(pos, ray_sun);

        sun_color = sky_get_sun_color(pos, ray_sun);
        sun_color = scale_color(sun_color, 0.25f * ONE_OVER_PI * sun_light_angle * scattering_sun * extinction_sun);
      }
      else {
        sun_color = get_color(0.0f, 0.0f, 0.0f);
      }

      // Ambient light
      const float ambient_r1 = PI * white_noise();
      const float ambient_r2 = 2.0f * PI * white_noise();

      const vec3 ray_ambient = sample_hemisphere_basis(ambient_r2, ambient_r1, normalize_vector(pos));

      const float cos_angle_ambient  = dot_product(ray, ray_ambient);
      const float scattering_ambient = cloud_dual_lobe_henvey_greenstein(
        cos_angle_ambient, device_scene.sky.cloud.forward_scattering, device_scene.sky.cloud.backward_scattering,
        device_scene.sky.cloud.lobe_lerp);

      const float extinction_ambient = cloud_extinction(pos, ray_ambient);

      RGBF ambient_color = sky_get_color(pos, ray_ambient, FLT_MAX, false, device_scene.sky.steps / 2);
      ambient_color      = scale_color(ambient_color, scattering_ambient * extinction_ambient * 0.5f);

      const float scattering = density * CLOUD_SCATTERING_DENSITY;
      const float extinction = density * CLOUD_EXTINCTION_DENSITY;

      RGBF S           = add_color(sun_color, ambient_color);
      S                = scale_color(S, scattering);
      S                = scale_color(S, cloud_powder(scattering, step_size));
      S                = scale_color(S, cloud_weather_wetness(weather));
      float step_trans = expf(-extinction * step_size);
      S                = scale_color(sub_color(S, scale_color(S, step_trans)), 1.0f / extinction);
      scatteredLight   = add_color(scatteredLight, scale_color(S, transmittance));
      transmittance *= step_trans;

      if (transmittance < 0.005f) {
        transmittance = 0.0f;
        break;
      }
    }

    reach += step_size;
  }

  RGBAF result;
  result.r = scatteredLight.r;
  result.g = scatteredLight.g;
  result.b = scatteredLight.b;
  result.a = transmittance;

  return result;
}

////////////////////////////////////////////////////////////////////
// Wrapper
////////////////////////////////////////////////////////////////////

__device__ void trace_clouds(const vec3 origin, const vec3 ray, const float start, const float distance, ushort2 index) {
  int pixel = index.x + index.y * device_width;

  if (distance <= 0.0f)
    return;

  RGBAF result = cloud_render(origin, ray, start, distance);

  if ((result.r + result.g + result.b) != 0.0f) {
    RGBF color  = RGBAhalf_to_RGBF(device.frame_buffer[pixel]);
    RGBF record = RGBAhalf_to_RGBF(device_records[pixel]);

    color.r += result.r * record.r;
    color.g += result.g * record.g;
    color.b += result.b * record.b;

    device.frame_buffer[pixel] = RGBF_to_RGBAhalf(color);
  }

  if (result.a != 1.0f) {
    RGBAhalf record = load_RGBAhalf(device_records + pixel);
    record          = scale_RGBAhalf(record, __float2half(result.a));
    store_RGBAhalf(device_records + pixel, record);
  }
}

////////////////////////////////////////////////////////////////////
// Kernel
////////////////////////////////////////////////////////////////////

__global__ __launch_bounds__(THREADS_PER_BLOCK, 4) void clouds_render_tasks() {
  const int task_count = device_trace_count[threadIdx.x + blockIdx.x * blockDim.x];

  for (int i = 0; i < task_count; i++) {
    const int offset  = get_task_address(i);
    TraceTask task    = load_trace_task(device_trace_tasks + offset);
    const float depth = __ldcs((float*) (device.trace_results + offset));

    const vec3 sky_origin = world_to_sky_transform(task.origin);

    const float sky_max_dist = (depth == device_scene.camera.far_clip_distance) ? FLT_MAX : world_to_sky_scale(depth);
    const float2 params      = cloud_get_intersection(sky_origin, task.ray, sky_max_dist);

    const bool cloud_hit = (params.x < FLT_MAX && params.y > 0.0f);

    if (cloud_hit) {
      trace_clouds(sky_origin, task.ray, params.x, params.y, task.index);

      task.origin = add_vector(task.origin, scale_vector(task.ray, sky_to_world_scale(params.x)));
      store_trace_task(device_trace_tasks + offset, task);
    }
  }
}

#endif /* CU_CLOUDS_H */
