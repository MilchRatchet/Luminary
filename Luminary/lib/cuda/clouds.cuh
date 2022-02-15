#ifndef CU_CLOUDS_H
#define CU_CLOUDS_H

#include "math.cuh"
#include "sky.cuh"
#include "utils.cuh"

/*
 * The code of this file is based on the cloud rendering in https://github.com/turanszkij/WickedEngine.
 */

__device__ float4 sample_noise_texture_2D(const uint8_t* texture, vec3 p, const int dim) {
  uchar4* src = (uchar4*) texture;

  p.x = dim * fractf(p.x);
  p.z = dim * fractf(p.z);

  int x0 = floorf(p.x);
  int z0 = floorf(p.z);
  int x1 = x0 + 1;
  int z1 = z0 + 1;

  x1 -= (x1 == dim) ? dim : 0;
  z1 -= (z1 == dim) ? dim : 0;

  float sx = p.x - x0;
  float sz = p.z - z0;

#define addr(x, z) (x + z * dim)

  uchar4 v00 = __ldg(src + addr(x0, z0));
  uchar4 v10 = __ldg(src + addr(x1, z0));
  uchar4 v01 = __ldg(src + addr(x0, z1));
  uchar4 v11 = __ldg(src + addr(x1, z1));

#undef addr

#define __floatify(a) make_float4(a.x, a.y, a.z, a.w)

  float4 f00 = __floatify(v00);
  float4 f10 = __floatify(v10);
  float4 f01 = __floatify(v01);
  float4 f11 = __floatify(v11);

#undef __floatify

#define __interp_vals(a, b, s) (make_float4(a.x + (b.x - a.x) * s, a.y + (b.y - a.y) * s, a.z + (b.z - a.z) * s, a.w + (b.w - a.w) * s))

  float4 xy0 = __interp_vals(f00, f10, sx);
  float4 xy1 = __interp_vals(f01, f11, sx);

  float4 xyz = __interp_vals(xy0, xy1, sz);

#undef __interp_vals

  xyz.x *= 1.0f / 255.0f;
  xyz.y *= 1.0f / 255.0f;
  xyz.z *= 1.0f / 255.0f;
  xyz.w *= 1.0f / 255.0f;

  return xyz;
}

__device__ float4 sample_noise_texture_3D(const uint8_t* texture, vec3 p, const int dim) {
  uchar4* src = (uchar4*) texture;

  p.x = dim * fractf(p.x);
  p.y = dim * fractf(p.y);
  p.z = dim * fractf(p.z);

  int x0 = floorf(p.x);
  int y0 = floorf(p.y);
  int z0 = floorf(p.z);
  int x1 = x0 + 1;
  int y1 = y0 + 1;
  int z1 = z0 + 1;

  x1 -= (x1 == dim) ? dim : 0;
  y1 -= (y1 == dim) ? dim : 0;
  z1 -= (z1 == dim) ? dim : 0;

  float sx = p.x - x0;
  float sy = p.y - y0;
  float sz = p.z - z0;

#define addr(x, y, z) (x + y * dim + z * dim * dim)

  uchar4 v000 = __ldg(src + addr(x0, y0, z0));
  uchar4 v100 = __ldg(src + addr(x1, y0, z0));
  uchar4 v010 = __ldg(src + addr(x0, y1, z0));
  uchar4 v110 = __ldg(src + addr(x1, y1, z0));
  uchar4 v001 = __ldg(src + addr(x0, y0, z1));
  uchar4 v101 = __ldg(src + addr(x1, y0, z1));
  uchar4 v011 = __ldg(src + addr(x0, y1, z1));
  uchar4 v111 = __ldg(src + addr(x1, y1, z1));

#undef addr

#define __floatify(a) make_float4(a.x, a.y, a.z, a.w)

  float4 f000 = __floatify(v000);
  float4 f100 = __floatify(v100);
  float4 f010 = __floatify(v010);
  float4 f110 = __floatify(v110);
  float4 f001 = __floatify(v001);
  float4 f101 = __floatify(v101);
  float4 f011 = __floatify(v011);
  float4 f111 = __floatify(v111);

#undef __floatify

#define __interp_vals(a, b, s) (make_float4(a.x + (b.x - a.x) * s, a.y + (b.y - a.y) * s, a.z + (b.z - a.z) * s, a.w + (b.w - a.w) * s))

  float4 x00 = __interp_vals(f000, f100, sx);
  float4 x10 = __interp_vals(f010, f110, sx);
  float4 x01 = __interp_vals(f001, f101, sx);
  float4 x11 = __interp_vals(f011, f111, sx);

  float4 xy0 = __interp_vals(x00, x10, sy);
  float4 xy1 = __interp_vals(x01, x11, sy);

  float4 xyz = __interp_vals(xy0, xy1, sz);

#undef __interp_vals

  xyz.x *= 1.0f / 255.0f;
  xyz.y *= 1.0f / 255.0f;
  xyz.z *= 1.0f / 255.0f;
  xyz.w *= 1.0f / 255.0f;

  return xyz;
}

#define CLOUD_STEPS 128
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

  float4 weather = sample_noise_texture_2D(
    device_scene.sky.cloud.weather_map, scale_vector(pos, 0.025f * device_scene.sky.cloud.noise_weather_scale), CLOUD_WEATHER_RES);

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

  float4 shape = sample_noise_texture_3D(
    device_scene.sky.cloud.shape_noise, scale_vector(pos, 0.4f * device_scene.sky.cloud.noise_shape_scale), CLOUD_SHAPE_RES);

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
    float4 curl = sample_noise_texture_2D(
      device_scene.sky.cloud.curl_noise, scale_vector(pos, 3.0f * device_scene.sky.cloud.noise_curl_scale), CLOUD_CURL_RES);

    pos.x += 2.0f * (curl.x - 0.5f) * height * 0.55f;
    pos.y += 2.0f * (curl.z - 0.5f) * height * 0.55f;
    pos.z += 2.0f * (curl.y - 0.5f) * height * 0.55f;

    float4 detail = sample_noise_texture_3D(
      device_scene.sky.cloud.detail_noise, scale_vector(pos, 2.0f * device_scene.sky.cloud.noise_detail_scale), CLOUD_DETAIL_RES);

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
  vec3 sun = scale_vector(device_sun, SKY_SUN_DISTANCE);
  sun.y -= SKY_EARTH_RADIUS;
  sun = sub_vector(sun, device_scene.sky.geometry_offset);

  const vec3 sun_diff     = sub_vector(sun, add_vector(origin, scale_vector(ray, start)));
  const vec3 ray_sun      = normalize_vector(sun_diff);
  const float light_angle = __saturatef(atanf(SKY_SUN_RADIUS / (get_length(sun_diff) + eps)));

  const float cos_angle_sun  = dot_product(ray, ray_sun);
  const float scattering_sun = cloud_dual_lobe_henvey_greenstein(
    cos_angle_sun, device_scene.sky.cloud.forward_scattering, device_scene.sky.cloud.backward_scattering, device_scene.sky.cloud.lobe_lerp);

  RGBF sun_color = sky_get_color(add_vector(origin, scale_vector(ray, start)), ray_sun, FLT_MAX, true);
  sun_color      = scale_color(sun_color, light_angle * scattering_sun);

  vec3 ray_ambient   = get_vector(0.0f, 1.0f, 0.0f);
  RGBF ambient_color = sky_get_color(add_vector(origin, scale_vector(ray, start)), ray_ambient, FLT_MAX, true);

  ambient_color = scale_color(ambient_color, 1.0f / (4.0f * PI));

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
    RGBF color  = device.frame_buffer[pixel];
    RGBF record = device_records[pixel];

    color.r += result.r * record.r;
    color.g += result.g * record.g;
    color.b += result.b * record.b;

    device.frame_buffer[pixel] = color;
  }

  if (result.a != 1.0f) {
    RGBF record           = device_records[pixel];
    record                = scale_color(record, result.a);
    device_records[pixel] = record;
  }
}

#endif /* CU_CLOUDS_H */
