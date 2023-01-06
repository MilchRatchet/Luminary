#ifndef CU_CLOUDS_H
#define CU_CLOUDS_H

#include "math.cuh"
#include "sky.cuh"
#include "utils.cuh"

/*
 * The code of this file is based on the cloud rendering in https://github.com/turanszkij/WickedEngine.
 */

#define CLOUD_STEPS 96
#define CLOUD_EXTINCTION_STEP_SIZE 0.02f
#define CLOUD_EXTINCTION_STEP_MULTIPLY 1.5f
#define CLOUD_EXTINCTION_STEP_MAX 0.75f

__device__ float cloud_gradient(float4 gradient, float height) {
  return smoothstep(height, gradient.x, gradient.y) - smoothstep(height, gradient.z, gradient.w);
}

__device__ float cloud_height_fraction(const vec3 pos) {
  const float low  = SKY_EARTH_RADIUS + device_scene.sky.cloud.height_min * 0.001f;
  const float high = SKY_EARTH_RADIUS + device_scene.sky.cloud.height_max * 0.001f;
  return remap(get_length(pos), low, high, 0.0f, 1.0f);
}

__device__ vec3 cloud_weather(vec3 pos, float height) {
  pos.x += device_scene.sky.cloud.offset_x;
  pos.z += device_scene.sky.cloud.offset_z;

  const float weather_sample_scale = 0.025f * device_scene.sky.cloud.noise_weather_scale;

  float4 weather = tex2D<float4>(device.cloud_noise[2], pos.x * weather_sample_scale, pos.z * weather_sample_scale);

  weather.x = powf(fabsf(weather.x), __saturatef(remap(height * 3.0f, 0.7f, 0.8f, 1.0f, lerp(1.0f, 0.5f, device_scene.sky.cloud.anvil))));

  weather.x = __saturatef(
    remap(weather.x * device_scene.sky.cloud.coverage, 0.0f, 1.0f, __saturatef(device_scene.sky.cloud.coverage_min - 1.0f), 1.0f));

  return get_vector(weather.x, weather.y, weather.z);
}

__device__ float cloud_density_height_gradient(const float height, const vec3 weather) {
  float small  = 1.0f - __saturatef(weather.y * 2.0f);
  float medium = 1.0f - fabsf(weather.y - 0.5f) * 2.0f;
  float large  = __saturatef(weather.y - 0.5f) * 2.0f;

  float4 gradient = make_float4(
    0.02f * (small + medium + large), 0.07f * (small + medium + large), 0.12f * small + 0.39f * medium + 0.88f * large,
    0.28f * small + 0.59f * medium + large);

  return cloud_gradient(gradient, height);
}

__device__ float cloud_density(vec3 pos, const float height, const vec3 weather) {
  pos.x += device_scene.sky.cloud.offset_x;
  pos.z += device_scene.sky.cloud.offset_z;

  const float shape_sample_scale = 0.4f * device_scene.sky.cloud.noise_shape_scale;

  float4 shape = tex3D<float4>(device.cloud_noise[0], pos.x * shape_sample_scale, pos.y * shape_sample_scale, pos.z * shape_sample_scale);

  const vec3 gradient = get_vector(
    cloud_gradient(make_float4(0.02f, 0.07f, 0.12f, 0.28f), height), cloud_gradient(make_float4(0.02f, 0.07f, 0.39f, 0.59f), height),
    cloud_gradient(make_float4(0.02f, 0.07f, 0.88f, 1.0f), height));

  shape.y *= gradient.x * 0.2f;
  shape.z *= gradient.y * 0.2f;
  shape.w *= gradient.z * 0.2f;

  float density_gradient = cloud_density_height_gradient(height, weather);

  float density = (shape.x + shape.y + shape.z + shape.w) * 0.8f * density_gradient;
  density       = powf(fabsf(density), __saturatef(height * 6.0f));

  density = smoothstep(density, 0.25f, 1.1f);

  density = __saturatef(density - (1.0f - weather.x)) * weather.x;

  if (density > 0.0f) {
    const float curl_sample_scale = 3.0f * device_scene.sky.cloud.noise_curl_scale;

    float4 curl = tex2D<float4>(device.cloud_noise[3], pos.x * curl_sample_scale, pos.z * curl_sample_scale);

    pos.x += 2.0f * (curl.x - 0.5f) * height * 0.55f;
    pos.y += 2.0f * (curl.z - 0.5f) * height * 0.55f;
    pos.z += 2.0f * (curl.y - 0.5f) * height * 0.55f;

    const float detail_sample_scale = 2.0f * device_scene.sky.cloud.noise_detail_scale;

    float4 detail =
      tex3D<float4>(device.cloud_noise[1], pos.x * detail_sample_scale, pos.y * detail_sample_scale, pos.z * detail_sample_scale);

    float detail_fbm = __saturatef(detail.x * 0.625f + detail.y * 0.25f + detail.z * 0.125f);

    float noise_modifier = lerp(1.0f - detail_fbm, detail_fbm, __saturatef(height * 10.0f));

    density = remap(density, noise_modifier * 0.2f, 1.0f, 0.0f, 1.0f);
  }

  return fmaxf(density * device_scene.sky.cloud.density, 0.0f);
}

__device__ float cloud_extinction(vec3 origin, vec3 ray) {
  float step_size  = CLOUD_EXTINCTION_STEP_SIZE;
  float reach      = step_size * 0.5f;
  float extinction = 0.0f;

  for (int i = 0; i < device_scene.sky.cloud.shadow_steps; i++) {
    vec3 pos = add_vector(origin, scale_vector(ray, reach));

    float height = cloud_height_fraction(pos);
    if (height > 1.0f)
      break;

    if (height < 0.0f) {
      const float h_min = device_scene.sky.cloud.height_min * 0.001f + SKY_EARTH_RADIUS;
      reach += sph_ray_int_p0(ray, pos, h_min);
      continue;
    }

    height = __saturatef(height);

    vec3 weather = cloud_weather(pos, height);

    if (weather.x > 0.3f) {
      const float density = 1000.0f * 0.05f * 0.1f * cloud_density(pos, height, weather);

      extinction -= density * step_size;
    }

    step_size *= CLOUD_EXTINCTION_STEP_MULTIPLY;
    step_size = fminf(CLOUD_EXTINCTION_STEP_MAX, step_size);
    reach += step_size;
  }

  return expf(extinction);
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

__device__ RGBAF cloud_render(const vec3 origin, const vec3 ray, const float start, const float dist) {
  const int step_count = CLOUD_STEPS * (dist / (start + dist)) * (is_first_ray() ? 1.0f : 0.25f);

  const float big_step     = 5.0f;
  const float default_step = fminf(0.001f * (device_scene.sky.cloud.height_max - device_scene.sky.cloud.height_min), dist) / step_count;

  int zero_density_streak = 0;

  float reach = start;
  float step  = big_step;

  reach -= default_step * step * white_noise();

  float transmittance = 1.0f;
  RGBF scatteredLight = get_color(0.0f, 0.0f, 0.0f);

  for (int i = 0; i < step_count; i++) {
    vec3 pos = add_vector(origin, scale_vector(ray, reach));

    float height = cloud_height_fraction(pos);

    if (height < 0.0f || height > 1.0f) {
      zero_density_streak++;
      if (zero_density_streak > 3)
        break;
      reach += default_step * step;
      continue;
    }

    height = __saturatef(height);

    vec3 weather = cloud_weather(pos, height);

    if (weather.x < 0.3f) {
      zero_density_streak++;
      step = (zero_density_streak > 3) ? big_step : step;
      reach += default_step * step;
      continue;
    }

    float density = cloud_density(pos, height, weather);

    if (density > 0.0f) {
      zero_density_streak = 0;

      if (step > 1.0f) {
        reach -= default_step * (step - 1.0f);
        step = 1.0f;
        continue;
      }

      // Celestial light (prefer sun but use the moon if possible)
      int sun_visible  = !sph_ray_hit_p0(normalize_vector(sub_vector(device_sun, pos)), pos, SKY_EARTH_RADIUS);
      int moon_visible = !sph_ray_hit_p0(normalize_vector(sub_vector(device_moon, pos)), pos, SKY_EARTH_RADIUS);

      const vec3 celestial_pos     = (sun_visible || !moon_visible) ? device_sun : device_moon;
      const float celestial_radius = (sun_visible || !moon_visible) ? SKY_SUN_RADIUS : SKY_MOON_RADIUS;

      const vec3 ray_sun      = sample_sphere(celestial_pos, celestial_radius, pos);
      const float light_angle = sample_sphere_solid_angle(celestial_pos, celestial_radius, pos);

      const float cos_angle_sun  = dot_product(ray, ray_sun);
      const float scattering_sun = cloud_dual_lobe_henvey_greenstein(
        cos_angle_sun, device_scene.sky.cloud.forward_scattering, device_scene.sky.cloud.backward_scattering,
        device_scene.sky.cloud.lobe_lerp);

      RGBF sun_color = sky_get_color(pos, ray_sun, FLT_MAX, true);
      sun_color      = scale_color(sun_color, 0.25f * ONE_OVER_PI * light_angle * scattering_sun);

      // Ambient light
      const float ambient_r1 = (2.0f * PI * white_noise()) - PI;
      const float ambient_r2 = 2.0f * PI * white_noise();

      const vec3 ray_ambient = angles_to_direction(ambient_r1, ambient_r2);

      const float cos_angle_ambient  = dot_product(ray, ray_ambient);
      const float scattering_ambient = cloud_dual_lobe_henvey_greenstein(
        cos_angle_ambient, device_scene.sky.cloud.forward_scattering, device_scene.sky.cloud.backward_scattering,
        device_scene.sky.cloud.lobe_lerp);

      RGBF ambient_color = sky_get_color(pos, ray_ambient, FLT_MAX, false);
      ambient_color      = scale_color(ambient_color, scattering_ambient);

      const float scattering = density * 1000.0f * 0.05f;
      const float extinction = density * 1000.0f * 0.05f * 0.1f;

      const float extinction_sun     = cloud_extinction(pos, ray_sun);
      const float extinction_ambient = cloud_extinction(pos, ray_ambient);

      RGBF S           = scale_color(sun_color, extinction_sun);
      S                = add_color(S, scale_color(ambient_color, extinction_ambient));
      S                = scale_color(S, scattering);
      S                = scale_color(S, cloud_powder(scattering, step));
      S                = scale_color(S, cloud_weather_wetness(weather));
      float step_trans = expf(-extinction * step);
      S                = scale_color(sub_color(S, scale_color(S, step_trans)), 1.0f / extinction);
      scatteredLight   = add_color(scatteredLight, scale_color(S, transmittance));
      transmittance *= step_trans;

      if (transmittance <= 0.1f) {
        transmittance = 0.0f;
        break;
      }
      step = (zero_density_streak > 3) ? big_step : 1.0f;
    }
    else {
      zero_density_streak++;
      step = (zero_density_streak > 3) ? big_step : step;
    }

    reach += default_step * step;
  }

  RGBAF result;
  result.r = scatteredLight.r;
  result.g = scatteredLight.g;
  result.b = scatteredLight.b;
  result.a = transmittance;

  return result;
}

__device__ float2 cloud_get_intersection(const vec3 origin, const vec3 ray, const float limit) {
  const float h_min = device_scene.sky.cloud.height_min * 0.001f + SKY_EARTH_RADIUS;
  const float h_max = device_scene.sky.cloud.height_max * 0.001f + SKY_EARTH_RADIUS;

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

#endif /* CU_CLOUDS_H */
