/*
 * Wave generation based on a shadertoy by Alexander Alekseev aka TDM (2014)
 * The shadertoy can be found on https://www.shadertoy.com/view/Ms2SD1
 */
#ifndef CU_OCEAN_H
#define CU_OCEAN_H

#include "math.cuh"

#define FAST_ITERATIONS 3
#define SLOW_ITERATIONS 6

__device__ float ocean_hash(const float2 p) {
  const float x = p.x * 127.1f + p.y * 311.7f;
  return fractf(sinf(x) * 43758.5453123f);
}

__device__ float ocean_noise(const float2 p) {
  float2 integral;
  integral.x = floorf(p.x);
  integral.y = floorf(p.y);

  float2 fractional;
  fractional.x = fractf(p.x);
  fractional.y = fractf(p.y);

  fractional.x *= fractional.x * (3.0f - 2.0f * fractional.x);
  fractional.y *= fractional.y * (3.0f - 2.0f * fractional.y);

  const float hash1 = ocean_hash(integral);
  integral.x += 1.0f;
  const float hash2 = ocean_hash(integral);
  integral.y += 1.0f;
  const float hash4 = ocean_hash(integral);
  integral.x -= 1.0f;
  const float hash3 = ocean_hash(integral);

  const float a = lerp(hash1, hash2, fractional.x);
  const float b = lerp(hash3, hash4, fractional.x);

  return -1.0f + 2.0f * lerp(a, b, fractional.y);
}

__device__ float ocean_octave(float2 p, const float choppyness) {
  const float offset = ocean_noise(p);
  p.x += offset;
  p.y += offset;

  float2 wave1;
  wave1.x = 1.0f - fabsf(sinf(p.x));
  wave1.y = 1.0f - fabsf(sinf(p.y));

  float2 wave2;
  wave2.x = fabsf(cosf(p.x));
  wave2.y = fabsf(cosf(p.y));

  wave1.x = lerp(wave1.x, wave2.x, wave1.x);
  wave1.y = lerp(wave1.y, wave2.y, wave1.y);

  return powf(1.0f - powf(wave1.x * wave1.y, 0.65f), choppyness);
}

__device__ float ocean_get_height(const vec3 p, const int steps) {
  float amplitude  = device_scene.ocean.amplitude;
  float choppyness = device_scene.ocean.choppyness;
  float frequency  = device_scene.ocean.frequency;

  float2 q = make_float2(p.x * 0.75f, p.z);

  float d = 0.0f;
  float h = 0.0f;

  float t = 1.0f + device_scene.ocean.time * device_scene.ocean.speed;

  for (int i = 0; i < steps; i++) {
    float2 a;
    a.x = (q.x + t) * frequency;
    a.y = (q.y + t) * frequency;
    d   = ocean_octave(a, choppyness);

    float2 b;
    b.x = (q.x - t) * frequency;
    b.y = (q.y - t) * frequency;
    d += ocean_octave(b, choppyness);

    h += d * amplitude;

    const float u = q.x;
    const float v = q.y;
    q.x           = 1.6f * u - 1.2f * v;
    q.y           = 1.2f * u + 1.6f * v;

    frequency *= 1.9f;
    amplitude *= 0.22f;
    choppyness = lerp(choppyness, 1.0f, 0.2f);
  }

  return p.y - h - device_scene.ocean.height;
}

__device__ vec3 ocean_get_normal(vec3 p, const float diff) {
  vec3 normal;
  normal.y = ocean_get_height(p, SLOW_ITERATIONS);
  p.x += diff;
  normal.x = ocean_get_height(p, SLOW_ITERATIONS) - normal.y;
  p.x -= diff;
  p.z += diff;
  normal.z = ocean_get_height(p, SLOW_ITERATIONS) - normal.y;
  normal.y = diff;

  return normalize_vector(normal);
}

__device__ float ocean_far_distance(const vec3 origin, const vec3 ray) {
  if (ray.y > -eps)
    return FLT_MAX;

  if (origin.y < device_scene.ocean.height)
    return FLT_MAX;

  const float depth = (device_scene.ocean.height - origin.y) / ray.y;

  return (depth >= eps) ? depth : FLT_MAX;
}

__device__ float ocean_short_distance(const vec3 origin, const vec3 ray) {
  if (ray.y > -eps)
    return FLT_MAX;

  if (origin.y < device_scene.ocean.height)
    return FLT_MAX;

  return (device_scene.ocean.height + 3.0f * device_scene.ocean.amplitude - origin.y) / ray.y;
}

__device__ float ocean_intersection_distance(const vec3 origin, const vec3 ray, float max) {
  float min = 0.0f;

  vec3 p = add_vector(origin, scale_vector(ray, max));

  float height_at_max = ocean_get_height(p, FAST_ITERATIONS);
  if (height_at_max > 0.0f)
    return FLT_MAX;

  float height_at_min = ocean_get_height(origin, FAST_ITERATIONS);
  if (height_at_min < 0.0f)
    return FLT_MAX;

  float mid = 0.0f;

  for (int i = 0; i < 8; i++) {
    mid = lerp(min, max, height_at_min / (height_at_min - height_at_max));
    p.x = origin.x + mid * ray.x;
    p.y = origin.y + mid * ray.y;
    p.z = origin.z + mid * ray.z;

    float height_at_mid = ocean_get_height(p, FAST_ITERATIONS);

    if (height_at_mid < 0.0f) {
      max           = mid;
      height_at_max = height_at_mid;
    }
    else {
      min           = mid;
      height_at_min = height_at_mid;
    }
  }

  return mid;
}

__global__ __launch_bounds__(THREADS_PER_BLOCK, 8) void process_ocean_tasks() {
  const int id = threadIdx.x + blockIdx.x * blockDim.x;

  const int task_count   = device.task_counts[id * 6 + 1];
  const int task_offset  = device.task_offsets[id * 5 + 1];
  int light_trace_count  = device.light_trace_count[id];
  int bounce_trace_count = device.bounce_trace_count[id];

  for (int i = 0; i < task_count; i++) {
    OceanTask task  = load_ocean_task(device_trace_tasks + get_task_address(task_offset + i));
    const int pixel = task.index.y * device_width + task.index.x;

    vec3 ray;
    ray.x = cosf(task.ray_xz) * cosf(task.ray_y);
    ray.y = sinf(task.ray_y);
    ray.z = sinf(task.ray_xz) * cosf(task.ray_y);

    task.state = (task.state & ~DEPTH_LEFT) | (((task.state & DEPTH_LEFT) - 1) & DEPTH_LEFT);

    const vec3 normal = ocean_get_normal(task.position, fmaxf(eps, task.distance / device_width));

    RGBAF albedo = device_scene.ocean.albedo;
    RGBF record  = RGBAhalf_to_RGBF(device_records[pixel]);

    if (device_scene.ocean.emissive) {
      RGBF emission = get_color(albedo.r, albedo.g, albedo.b);

      write_albedo_buffer(emission, pixel);

      emission.r *= 2.0f * record.r;
      emission.g *= 2.0f * record.g;
      emission.b *= 2.0f * record.b;

      device.frame_buffer[pixel] = RGBF_to_RGBAhalf(emission);
    }

    write_normal_buffer(normal, pixel);

    const vec3 V      = scale_vector(ray, -1.0f);
    BRDFInstance brdf = brdf_get_instance(RGBAF_to_RGBAhalf(albedo), V, normal, 0.0f, 1.0f);

    if (white_noise() > albedo.a) {
      task.position = add_vector(task.position, scale_vector(ray, 2.0f * eps));

      const float refraction_index = 1.0f / device_scene.ocean.refractive_index;

      brdf = brdf_sample_ray_refraction(brdf, refraction_index, 0.0f, 0.0f);

      RGBAhalf alpha_record = RGBF_to_RGBAhalf(record);

      TraceTask new_task;
      new_task.origin = task.position;
      new_task.ray    = brdf.L;
      new_task.index  = task.index;
      new_task.state  = task.state;

      switch (device_iteration_type) {
        case TYPE_CAMERA:
        case TYPE_BOUNCE:
          store_RGBAhalf(device.bounce_records + pixel, alpha_record);
          store_trace_task(device.bounce_trace + get_task_address(bounce_trace_count++), new_task);
          break;
        case TYPE_LIGHT:
          if (white_noise() > 0.5f)
            break;
          store_RGBAhalf(device.light_records + pixel, scale_RGBAhalf(alpha_record, 2.0f));
          store_trace_task(device.light_trace + get_task_address(light_trace_count++), new_task);
          device.state_buffer[pixel] |= STATE_LIGHT_OCCUPIED;
          break;
      }
    }
    else if (device_iteration_type != TYPE_LIGHT) {
      task.position = add_vector(task.position, scale_vector(normal, 8.0f * eps));
      task.state    = (task.state & ~RANDOM_INDEX) | (((task.state & RANDOM_INDEX) + 1) & RANDOM_INDEX);

      uint32_t light_history_buffer_entry = LIGHT_ID_ANY_NO_SUN;

      {
        LightSample light = load_light_sample(device.light_samples, pixel);

        const float light_weight = brdf_light_sample_shading_weight(light) * light.solid_angle;

        if (light_weight > 0.0f) {
          brdf.L = brdf_sample_light_ray(light, task.position);

          const RGBAhalf light_record = mul_RGBAhalf(RGBF_to_RGBAhalf(record), scale_RGBAhalf(brdf_evaluate(brdf).term, light_weight));

          TraceTask light_task;
          light_task.origin = task.position;
          light_task.ray    = brdf.L;
          light_task.index  = task.index;
          light_task.state  = task.state;

          if (any_RGBAhalf(light_record)) {
            store_RGBAhalf(device.light_records + pixel, light_record);
            light_history_buffer_entry = light.id;
            store_trace_task(device.light_trace + get_task_address(light_trace_count++), light_task);
          }
        }
      }

      if (!(device.state_buffer[pixel] & STATE_LIGHT_OCCUPIED))
        device.light_sample_history[pixel] = light_history_buffer_entry;

      brdf = brdf_sample_ray(brdf, task.index, task.state);

      RGBAhalf bounce_record = mul_RGBAhalf(RGBF_to_RGBAhalf(record), brdf.term);

      TraceTask bounce_task;
      bounce_task.origin = task.position;
      bounce_task.ray    = brdf.L;
      bounce_task.index  = task.index;
      bounce_task.state  = task.state;

      store_RGBAhalf(device.bounce_records + pixel, bounce_record);
      store_trace_task(device.bounce_trace + get_task_address(bounce_trace_count++), bounce_task);
    }
  }

  device.light_trace_count[id]  = light_trace_count;
  device.bounce_trace_count[id] = bounce_trace_count;
}

__global__ __launch_bounds__(THREADS_PER_BLOCK, 10) void process_debug_ocean_tasks() {
  const int id = threadIdx.x + blockIdx.x * blockDim.x;

  const int task_count  = device.task_counts[id * 6 + 1];
  const int task_offset = device.task_offsets[id * 5 + 1];

  for (int i = 0; i < task_count; i++) {
    OceanTask task  = load_ocean_task(device_trace_tasks + get_task_address(task_offset + i));
    const int pixel = task.index.y * device_width + task.index.x;

    if (device_shading_mode == SHADING_ALBEDO) {
      RGBAF albedo               = device_scene.ocean.albedo;
      device.frame_buffer[pixel] = RGBF_to_RGBAhalf(get_color(albedo.r, albedo.g, albedo.b));
    }
    else if (device_shading_mode == SHADING_DEPTH) {
      const float value          = __saturatef((1.0f / task.distance) * 2.0f);
      device.frame_buffer[pixel] = RGBF_to_RGBAhalf(get_color(value, value, value));
    }
    else if (device_shading_mode == SHADING_NORMAL) {
      vec3 normal = ocean_get_normal(task.position, fmaxf(eps, task.distance / device_width));

      normal.x = 0.5f * normal.x + 0.5f;
      normal.y = 0.5f * normal.y + 0.5f;
      normal.z = 0.5f * normal.z + 0.5f;

      device.frame_buffer[pixel] = RGBF_to_RGBAhalf(get_color(__saturatef(normal.x), __saturatef(normal.y), __saturatef(normal.z)));
    }
    else if (device_shading_mode == SHADING_WIREFRAME) {
      int a = fabsf(floorf(task.position.x) - task.position.x) < 0.001f;
      int b = fabsf(floorf(task.position.z) - task.position.z) < 0.001f;
      int c = fabsf(ceilf(task.position.x) - task.position.x) < 0.001f;
      int d = fabsf(ceilf(task.position.z) - task.position.z) < 0.001f;

      float light = (a || b || c || d) ? 1.0f : 0.0f;

      device.frame_buffer[pixel] = RGBF_to_RGBAhalf(get_color(0.0f, 0.5f * light, light));
    }
  }
}

#endif /* CU_OCEAN_H */
