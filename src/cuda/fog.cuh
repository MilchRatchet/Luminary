#ifndef CU_FOG_H
#define CU_FOG_H

#include "math.cuh"
#include "sky.cuh"

/*
 * Computes randomly a scattering point for fog.
 * @param origin Origin of ray.
 * @param ray Direction of ray.
 * @param depth Depth of ray without scattering.
 * @param random Random value in [0,1)
 * @result Two floats, the first containing the sampled depth, the second containing the probability for the scattering point.
 */
__device__ float2 get_intersection_fog(const vec3 origin, const vec3 ray, const float depth, const float random) {
  const float height = device_scene.fog.height + device_scene.fog.falloff;
  const float dist   = (fabsf(ray.y) < eps * eps) ? FLT_MAX : (height - origin.y) / ray.y;

  float max_dist = FLT_MAX;
  float min_dist = 0.0f;

  if (dist < 0.0f) {
    if (origin.y > height) {
      max_dist = -FLT_MAX;
    }
    else {
      max_dist = FLT_MAX;
    }
  }
  else {
    if (origin.y > height) {
      min_dist = dist;
    }
    else {
      max_dist = dist;
    }
  }

  max_dist = fminf(device_scene.fog.dist, max_dist);

  float t = min_dist - logf(random) / (0.001f * device_scene.fog.scattering);

  float p = 1.0f - (expf((min_dist - fminf(max_dist, depth)) * 0.001f * device_scene.fog.scattering));

  if (t < min_dist || t > max_dist) {
    t = FLT_MAX;
  }

  return make_float2(t, p);
}

__device__ float get_fog_depth(float y, float ry, float depth) {
  float height = device_scene.fog.height + device_scene.fog.falloff;

  if (y >= height && ry >= 0.0f)
    return 0.0f;

  if (y < height && ry <= 0.0f)
    return fminf(device_scene.fog.dist, depth);

  if (y < height) {
    return fminf(device_scene.fog.dist, fminf(((height - y) / ry), depth));
  }

  return fmaxf(0.0f, fminf(device_scene.fog.dist, depth - ((height - y) / ry)));
}

__device__ float get_fog_density(float base_density, float height) {
  if (height > device_scene.fog.height) {
    base_density = (height < device_scene.fog.height + device_scene.fog.falloff)
                     ? lerp(base_density, 0.0f, (height - device_scene.fog.height) / device_scene.fog.falloff)
                     : 0.0f;
  }

  return base_density;
}

__global__ __launch_bounds__(THREADS_PER_BLOCK, 9) void process_fog_tasks() {
  const int id = threadIdx.x + blockIdx.x * blockDim.x;

  const int task_count   = device.task_counts[id * 6 + 4];
  const int task_offset  = device.task_offsets[id * 5 + 4];
  int light_trace_count  = device.light_trace_count[id];
  int bounce_trace_count = device.bounce_trace_count[id];

  for (int i = 0; i < task_count; i++) {
    FogTask task    = load_fog_task(device_trace_tasks + get_task_address(task_offset + i));
    const int pixel = task.index.y * device_width + task.index.x;

    vec3 ray;
    ray.x = cosf(task.ray_xz) * cosf(task.ray_y);
    ray.y = sinf(task.ray_y);
    ray.z = sinf(task.ray_xz) * cosf(task.ray_y);

    const float density = get_fog_density(0.001f * device_scene.fog.scattering, task.position.y);

    task.state = (task.state & ~DEPTH_LEFT) | (((task.state & DEPTH_LEFT) - 1) & DEPTH_LEFT);

    RGBF record = RGBAhalf_to_RGBF(device_records[pixel]);

    {
      const vec3 bounce_ray = angles_to_direction(white_noise() * PI, white_noise() * 2.0f * PI);
      const float cos_angle = dot_product(ray, bounce_ray);
      const float phase     = henvey_greenstein(cos_angle, device_scene.fog.anisotropy);
      const float weight    = 4.0f * PI * phase * density * task.distance;

      device.bounce_records[pixel] = RGBF_to_RGBAhalf(scale_color(record, weight));

      TraceTask bounce_task;
      bounce_task.origin = task.position;
      bounce_task.ray    = bounce_ray;
      bounce_task.index  = task.index;
      bounce_task.state  = task.state;

      store_trace_task(device.bounce_trace + get_task_address(bounce_trace_count++), bounce_task);
    }

    const int light_occupied = (device.state_buffer[pixel] & STATE_LIGHT_OCCUPIED);

    if (!light_occupied) {
      LightSample light        = load_light_sample(device.light_samples, pixel);
      const float light_weight = brdf_light_sample_shading_weight(light) * light.solid_angle;

      uint32_t light_history_buffer_entry = LIGHT_ID_ANY;

      if (light_weight > 0.0f) {
        const vec3 light_ray  = brdf_sample_light_ray(light, task.position);
        const float cos_angle = dot_product(ray, light_ray);
        const float phase     = henvey_greenstein(cos_angle, device_scene.fog.anisotropy);
        const float weight    = light_weight * phase * density * task.distance;

        device.light_records[pixel] = RGBF_to_RGBAhalf(scale_color(record, weight));
        light_history_buffer_entry  = light.id;

        TraceTask light_task;
        light_task.origin = task.position;
        light_task.ray    = light_ray;
        light_task.index  = task.index;
        light_task.state  = task.state;

        store_trace_task(device.light_trace + get_task_address(light_trace_count++), light_task);
      }

      device.light_sample_history[pixel] = light_history_buffer_entry;
    }

    task.state = (task.state & ~RANDOM_INDEX) | (((task.state & RANDOM_INDEX) + 1) & RANDOM_INDEX);
  }

  device.light_trace_count[id]  = light_trace_count;
  device.bounce_trace_count[id] = bounce_trace_count;
}

__global__ __launch_bounds__(THREADS_PER_BLOCK, 12) void process_debug_fog_tasks() {
  const int id = threadIdx.x + blockIdx.x * blockDim.x;

  const int task_count  = device.task_counts[id * 6 + 4];
  const int task_offset = device.task_offsets[id * 5 + 4];

  for (int i = 0; i < task_count; i++) {
    FogTask task    = load_fog_task(device_trace_tasks + get_task_address(task_offset + i));
    const int pixel = task.index.y * device_width + task.index.x;

    if (device_shading_mode == SHADING_ALBEDO || device_shading_mode == SHADING_NORMAL) {
      device.frame_buffer[pixel] = RGBF_to_RGBAhalf(get_color(0.0f, 0.0f, 0.0f));
    }
    else if (device_shading_mode == SHADING_DEPTH) {
      const float value          = __saturatef((1.0f / task.distance) * 2.0f);
      device.frame_buffer[pixel] = RGBF_to_RGBAhalf(get_color(value, value, value));
    }
  }
}

#define FOG_DENSITY_SCALE 0.001f

__device__ RGBF fog_extinction(const vec3 origin, const vec3 ray, const float start, const float length) {
  if (length <= 0.0f)
    return get_color(1.0f, 1.0f, 1.0f);

  float density = -device_scene.fog.extinction * FOG_DENSITY_SCALE * length;

  return get_color(expf(density), expf(density), expf(density));
}

__global__ void process_fog_extinction_only() {
  const int task_count = device_trace_count[threadIdx.x + blockIdx.x * blockDim.x];

  for (int i = 0; i < task_count; i++) {
    const int offset     = get_task_address(i);
    const TraceTask task = load_trace_task(device_trace_tasks + offset);
    const float limit    = world_to_sky_scale(__ldca((float*) (device.trace_results + offset)));
    const int pixel      = task.index.x + task.index.y * device_width;

    const vec3 origin = world_to_sky_transform(task.origin);

    float2 path = sky_compute_path(origin, task.ray, SKY_EARTH_RADIUS, SKY_EARTH_RADIUS + device_scene.fog.height);

    if (path.y == -FLT_MAX)
      continue;

    const float start    = path.x;
    const float distance = fminf(path.y, limit - start);

    if (distance > 0.0f) {
      const RGBF extinction = fog_extinction(origin, task.ray, start, distance);

      store_RGBAhalf(device_records + pixel, mul_RGBAhalf(load_RGBAhalf(device_records + pixel), RGBF_to_RGBAhalf(extinction)));
    }
  }
}

__global__ void process_fog_intratasks() {
  const int task_count = device_trace_count[threadIdx.x + blockIdx.x * blockDim.x];

  for (int i = 0; i < task_count; i++) {
    const int offset     = get_task_address(i);
    const TraceTask task = load_trace_task(device_trace_tasks + offset);
    const float limit    = world_to_sky_scale(__ldca((float*) (device.trace_results + offset)));
    const int pixel      = task.index.x + task.index.y * device_width;

    const vec3 origin = world_to_sky_transform(task.origin);

    float2 path = sky_compute_path(origin, task.ray, SKY_EARTH_RADIUS, SKY_EARTH_RADIUS + device_scene.fog.height);

    if (path.y == -FLT_MAX)
      continue;

    const float start    = fmaxf(path.x, 0.0f);
    const float distance = fminf(path.y, limit - start);

    if (distance > 0.0f) {
      /*
       * Compute inscattering
       * Currently only unshadowed sun is being sampled
       */

      const float reach = start + distance * white_noise();

      const vec3 pos = add_vector(origin, scale_vector(task.ray, reach));

      const float light_angle = sample_sphere_solid_angle(device_sun, SKY_SUN_RADIUS, pos);
      const vec3 ray_scatter  = normalize_vector(sub_vector(device_sun, pos));

      const float scatter_distance = sph_ray_int_p0(ray_scatter, pos, SKY_EARTH_RADIUS + device_scene.fog.height);

      const RGBF extinction_light = fog_extinction(pos, ray_scatter, 0.0f, scatter_distance);

      const float cos_angle = fmaxf(0.0f, dot_product(task.ray, ray_scatter));
      const float phase_mie = henvey_greenstein(cos_angle, device_scene.fog.anisotropy);

      // Amount of light that reached pos
      RGBF S = scale_color(device_scene.sky.sun_color, device_scene.sky.sun_strength);
      S      = mul_color(S, extinction_light);

      // Amount of light that gets scattered towards camera at pos
      const float scattering = device_scene.fog.scattering * FOG_DENSITY_SCALE * light_angle * phase_mie;

      S = scale_color(S, scattering);

      // Amount of light that gets lost along this step
      const float transmittance = expf(-distance * device_scene.fog.extinction * FOG_DENSITY_SCALE);

      // Amount of light that gets scattered towards camera along this step
      S = scale_color(sub_color(S, scale_color(S, transmittance)), 1.0f / (device_scene.fog.extinction * FOG_DENSITY_SCALE));

      /*
       * Apply extinction to record
       */

      const RGBF extinction = fog_extinction(origin, task.ray, start, distance);

      RGBAhalf record = load_RGBAhalf(device_records + pixel);

      // Apply record before modifying the record
      const RGBAhalf fog_inscattered_light = mul_RGBAhalf(RGBF_to_RGBAhalf(S), record);
      store_RGBAhalf(device.frame_buffer + pixel, add_RGBAhalf(load_RGBAhalf(device.frame_buffer + pixel), fog_inscattered_light));

      record = mul_RGBAhalf(record, RGBF_to_RGBAhalf(extinction));
      store_RGBAhalf(device_records + pixel, record);
    }
  }
}

#endif /* CU_FOG_H */
