#ifndef CU_FOG_H
#define CU_FOG_H

#include "math.cuh"
#include "sky.cuh"

#define FOG_DENSITY_SCALE 0.001f

/*
 * Computes the start and length of a ray path through fog.
 * @param origin Start point of ray in world space.
 * @param ray Direction of ray
 * @param limit Max distance ray may travel
 * @result Two floats:
 *                  - [x] = Start in sky space
 *                  - [y] = Distance through fog in sky space
 */
__device__ float2 fog_compute_path(const vec3 origin, const vec3 ray, const float limit) {
  const float sky_limit = world_to_sky_scale(limit);
  const vec3 sky_origin = world_to_sky_transform(origin);
  const float height    = world_to_sky_scale(device_scene.fog.height);
  const float2 path     = sky_compute_path(sky_origin, ray, SKY_EARTH_RADIUS, SKY_EARTH_RADIUS + height);

  if (path.y == -FLT_MAX)
    return make_float2(-FLT_MAX, -FLT_MAX);

  const float start    = fmaxf(path.x, 0.0f);
  const float distance = fmaxf(fminf(fminf(path.y, sky_limit - start), world_to_sky_scale(device_scene.fog.dist)), 0.0f);

  return make_float2(start, distance);
}

/*
 * Computes randomly a scattering point for fog.
 * @param origin Origin of ray in world space.
 * @param ray Direction of ray.
 * @param limit Depth of ray without scattering in world space.
 * @result Two floats:
 *                  - [x] = Sampled Depth in world space
 *                  - [y] = Weight associated with this sample
 */
__device__ float2 fog_get_intersection(const vec3 origin, const vec3 ray, const float limit) {
  const float2 path = fog_compute_path(origin, ray, limit);

  if (path.y == -FLT_MAX)
    return make_float2(FLT_MAX, 1.0f);

  const float start    = sky_to_world_scale(path.x);
  const float distance = sky_to_world_scale(path.y);

  if (start > limit)
    return make_float2(FLT_MAX, 1.0f);

  float t = start + 2.0f * white_noise() * distance;

  float weight = 2.0f * distance * expf(-(t - start) * FOG_DENSITY_SCALE * device_scene.fog.extinction);

  if (t > start + distance) {
    t      = FLT_MAX;
    weight = 2.0f * expf(-distance * FOG_DENSITY_SCALE * device_scene.fog.extinction);
  }

  return make_float2(t, weight);
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

    task.state = (task.state & ~DEPTH_LEFT) | (((task.state & DEPTH_LEFT) - 1) & DEPTH_LEFT);

    RGBF record = RGBAhalf_to_RGBF(device_records[pixel]);

    {
      const vec3 bounce_ray = angles_to_direction(white_noise() * PI, white_noise() * 2.0f * PI);
      const float cos_angle = dot_product(ray, bounce_ray);
      const float phase     = henvey_greenstein(cos_angle, device_scene.fog.anisotropy);
      const float weight    = 4.0f * PI * phase * FOG_DENSITY_SCALE * device_scene.fog.scattering;

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
        const float weight    = light_weight * phase * FOG_DENSITY_SCALE * device_scene.fog.scattering;

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
       * Currently only unshadowed bogus random light stuff idk
       */

      float step_size = 0.2f;
      float reach     = start + step_size * white_noise();

      vec3 light_pos;

      while (reach < distance) {
        const vec3 pos = add_vector(origin, scale_vector(task.ray, reach));
      }

      float sampling_weight = 2.0f;
      float light_angle;
      vec3 pos, ray_scatter;
      RGBF light_color;

      if (white_noise() < 0.5f) {
        // Sample sun
        const float reach = start + distance * white_noise();

        pos = add_vector(origin, scale_vector(task.ray, reach));

        light_angle = sample_sphere_solid_angle(device_sun, SKY_SUN_RADIUS, pos);
        ray_scatter = normalize_vector(sub_vector(device_sun, pos));

        light_color = scale_color(device_scene.sky.sun_color, device_scene.sky.sun_strength);
      }
      else {
        const vec3 sphere_center = world_to_sky_transform(device_scene.toy.position);

        // const float reach = (dot_product(task.ray, origin) - dot_product(task.ray, triangle_center)) / dot_product(task.ray, task.ray);
        const float reach = start + distance * white_noise();

        pos = add_vector(origin, scale_vector(task.ray, reach));

        light_angle = sample_sphere_solid_angle(sphere_center, world_to_sky_scale(device_scene.toy.scale), pos);
        ray_scatter = normalize_vector(sub_vector(sphere_center, pos));

        light_color = get_color(10.0f, 0.0f, 10.0f);
      }

      const float scatter_distance = sph_ray_int_p0(ray_scatter, pos, SKY_EARTH_RADIUS + device_scene.fog.height);

      const RGBF extinction_light = fog_extinction(pos, ray_scatter, 0.0f, scatter_distance);

      const float cos_angle = fmaxf(0.0f, dot_product(task.ray, ray_scatter));
      const float phase_mie = henvey_greenstein(cos_angle, device_scene.fog.anisotropy);

      // Amount of light that reached pos
      RGBF S = mul_color(light_color, extinction_light);

      // Amount of light that gets scattered towards camera at pos
      const float scattering = device_scene.fog.scattering * FOG_DENSITY_SCALE * light_angle * phase_mie;

      S = scale_color(S, scattering);

      // Amount of light that gets lost along this step
      const float transmittance = expf(-distance * device_scene.fog.extinction * FOG_DENSITY_SCALE);

      // Amount of light that gets scattered towards camera along this step
      S = scale_color(sub_color(S, scale_color(S, transmittance)), 1.0f / (device_scene.fog.extinction * FOG_DENSITY_SCALE));

      S = scale_color(S, sampling_weight);

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
