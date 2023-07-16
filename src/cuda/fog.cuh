#ifndef CU_FOG_H
#define CU_FOG_H

#include "math.cuh"
#include "ocean.cuh"
#include "sky.cuh"
#include "sky_utils.cuh"

#define FOG_DENSITY (0.0001f * device.scene.fog.density)

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
  const float height    = world_to_sky_scale(device.scene.fog.height);
  const float2 path     = sky_compute_path(sky_origin, ray, SKY_EARTH_RADIUS, SKY_EARTH_RADIUS + height);

  if (path.y == -FLT_MAX)
    return make_float2(-FLT_MAX, -FLT_MAX);

  const float start    = fmaxf(path.x, 0.0f);
  const float distance = fmaxf(fminf(fminf(path.y, sky_limit - start), world_to_sky_scale(device.scene.fog.dist)), 0.0f);

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

  float weight = 2.0f * distance * expf(-(t - start) * FOG_DENSITY);

  if (t > start + distance) {
    t      = FLT_MAX;
    weight = 2.0f * expf(-distance * FOG_DENSITY);
  }

  return make_float2(t, weight);
}

////////////////////////////////////////////////////////////////////
// Kernel
////////////////////////////////////////////////////////////////////

__global__ void fog_generate_light_eval_data() {
  const int id = threadIdx.x + blockIdx.x * blockDim.x;

  const int task_count  = device.ptrs.task_counts[id * 6 + 4];
  const int task_offset = device.ptrs.task_offsets[id * 5 + 4];

  for (int i = 0; i < task_count; i++) {
    FogTask task    = load_fog_task(device.trace_tasks + get_task_address(task_offset + i));
    const int pixel = task.index.y * device.width + task.index.x;

    vec3 ray;
    ray.x = cosf(task.ray_xz) * cosf(task.ray_y);
    ray.y = sinf(task.ray_y);
    ray.z = sinf(task.ray_xz) * cosf(task.ray_y);

    LightEvalData data;
    data.flags     = LIGHT_EVAL_DATA_REQUIRES_SAMPLING | LIGHT_EVAL_DATA_VOLUME_HIT;
    data.normal    = get_vector(0.0f, 0.0f, 0.0f);
    data.position  = task.position;
    data.V         = scale_vector(ray, -1.0f);
    data.roughness = 1.0f;
    data.metallic  = 0.0f;

    store_light_eval_data(data, pixel);
  }
}

__global__ __launch_bounds__(THREADS_PER_BLOCK, 9) void process_fog_tasks() {
  const int id = threadIdx.x + blockIdx.x * blockDim.x;

  const int task_count   = device.ptrs.task_counts[id * 6 + 4];
  const int task_offset  = device.ptrs.task_offsets[id * 5 + 4];
  int light_trace_count  = device.ptrs.light_trace_count[id];
  int bounce_trace_count = device.ptrs.bounce_trace_count[id];

  for (int i = 0; i < task_count; i++) {
    FogTask task    = load_fog_task(device.trace_tasks + get_task_address(task_offset + i));
    const int pixel = task.index.y * device.width + task.index.x;

    vec3 ray;
    ray.x = cosf(task.ray_xz) * cosf(task.ray_y);
    ray.y = sinf(task.ray_y);
    ray.z = sinf(task.ray_xz) * cosf(task.ray_y);

    RGBF record = device.records[pixel];

    {
      const vec3 bounce_ray = angles_to_direction(white_noise() * PI, white_noise() * 2.0f * PI);
      const float cos_angle = dot_product(ray, bounce_ray);
      const float phase     = jendersie_eon_phase_function(cos_angle, device.scene.fog.droplet_diameter);

      // solid angle is 4.0 * PI
      const float S      = 4.0f * PI * phase * FOG_DENSITY;
      const float weight = (S - S * expf(-task.distance * FOG_DENSITY)) / FOG_DENSITY;

      device.ptrs.bounce_records[pixel] = scale_color(record, weight);

      TraceTask bounce_task;
      bounce_task.origin = task.position;
      bounce_task.ray    = bounce_ray;
      bounce_task.index  = task.index;

      store_trace_task(device.ptrs.bounce_trace + get_task_address(bounce_trace_count++), bounce_task);
    }

    const int light_occupied = (device.ptrs.state_buffer[pixel] & STATE_LIGHT_OCCUPIED);

    if (!light_occupied) {
      LightSample light = load_light_sample(device.ptrs.light_samples, pixel);

      uint32_t light_history_buffer_entry = LIGHT_ID_ANY;

      BRDFInstance brdf = brdf_get_instance_scattering();

      if (light.weight > 0.0f) {
        BRDFInstance brdf_sample = brdf_apply_sample_scattering(brdf, light, task.position, device.scene.fog.droplet_diameter);

        const float weight = (FOG_DENSITY - FOG_DENSITY * expf(-task.distance * FOG_DENSITY)) / FOG_DENSITY;

        device.ptrs.light_records[pixel] = mul_color(record, scale_color(brdf_sample.term, weight));
        light_history_buffer_entry       = light.id;

        TraceTask light_task;
        light_task.origin = task.position;
        light_task.ray    = brdf_sample.L;
        light_task.index  = task.index;

        store_trace_task(device.ptrs.light_trace + get_task_address(light_trace_count++), light_task);
      }

      device.ptrs.light_sample_history[pixel] = light_history_buffer_entry;
    }
  }

  device.ptrs.light_trace_count[id]  = light_trace_count;
  device.ptrs.bounce_trace_count[id] = bounce_trace_count;
}

__global__ __launch_bounds__(THREADS_PER_BLOCK, 12) void process_debug_fog_tasks() {
  const int id = threadIdx.x + blockIdx.x * blockDim.x;

  const int task_count  = device.ptrs.task_counts[id * 6 + 4];
  const int task_offset = device.ptrs.task_offsets[id * 5 + 4];

  for (int i = 0; i < task_count; i++) {
    FogTask task    = load_fog_task(device.trace_tasks + get_task_address(task_offset + i));
    const int pixel = task.index.y * device.width + task.index.x;

    if (device.shading_mode == SHADING_ALBEDO || device.shading_mode == SHADING_NORMAL) {
      device.ptrs.frame_buffer[pixel] = RGBF_to_RGBAhalf(get_color(0.0f, 0.0f, 0.0f));
    }
    else if (device.shading_mode == SHADING_DEPTH) {
      const float value               = __saturatef((1.0f / task.distance) * 2.0f);
      device.ptrs.frame_buffer[pixel] = RGBF_to_RGBAhalf(get_color(value, value, value));
    }
  }
}

__global__ __launch_bounds__(THREADS_PER_BLOCK, 10) void fog_preprocess_tasks() {
  const int task_count = device.trace_count[threadIdx.x + blockIdx.x * blockDim.x];

  for (int i = 0; i < task_count; i++) {
    const int offset    = get_task_address(i);
    TraceTask task      = load_trace_task(device.trace_tasks + offset);
    const float2 result = __ldcs((float2*) (device.ptrs.trace_results + offset));

    float depth     = result.x;
    uint32_t hit_id = __float_as_uint(result.y);

    const int pixel = task.index.x + task.index.y * device.width;
    RGBF record     = load_RGBF(device.records + pixel);

    if (device.scene.fog.active && device.iteration_type != TYPE_LIGHT) {
      const float2 fog = fog_get_intersection(task.origin, task.ray, depth);

      const float weight = fog.y;

      if (fog.x < depth) {
        depth  = fog.x;
        hit_id = FOG_HIT;
        __stcs((float2*) (device.ptrs.trace_results + offset), make_float2(depth, __uint_as_float(hit_id)));
      }

      record = scale_color(record, weight);
    }

    if (device.scene.fog.active) {
      const float t      = fog_compute_path(task.origin, task.ray, depth).y;
      const float weight = expf(-t * FOG_DENSITY);

      record = scale_color(record, weight);
    }

    if (device.scene.ocean.active) {
      const float underwater_dist = ocean_ray_underwater_length(task.origin, task.ray, depth);

      RGBF extinction = ocean_get_extinction();

      RGBF path_extinction;
      path_extinction.r = expf(-underwater_dist * extinction.r);
      path_extinction.g = expf(-underwater_dist * extinction.g);
      path_extinction.b = expf(-underwater_dist * extinction.b);

      record = mul_color(record, path_extinction);
    }

    store_RGBF(device.records + pixel, record);
  }
}

#endif /* CU_FOG_H */
