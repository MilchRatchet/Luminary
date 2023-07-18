/*
 * Wave generation based on a shadertoy by Alexander Alekseev aka TDM (2014)
 * The shadertoy can be found on https://www.shadertoy.com/view/Ms2SD1
 */
#ifndef CU_OCEAN_H
#define CU_OCEAN_H

#include "brdf.cuh"
#include "math.cuh"
#include "memory.cuh"

#define OCEAN_POLLUTION (device.scene.ocean.pollution * 0.01f)
#define OCEAN_SCATTERING (scale_color(device.scene.ocean.scattering, OCEAN_POLLUTION))
#define OCEAN_ABSORPTION (scale_color(device.scene.ocean.absorption, device.scene.ocean.absorption_strength * 0.02f))
#define OCEAN_EXTINCTION (add_color(OCEAN_SCATTERING, OCEAN_ABSORPTION))

#define OCEAN_MAX_HEIGHT (device.scene.ocean.height + 3.0f * device.scene.ocean.amplitude)
#define OCEAN_MIN_HEIGHT (device.scene.ocean.height)

#define OCEAN_ITERATIONS_INTERSECTION 4
#define OCEAN_ITERATIONS_NORMAL 6

__device__ float ocean_get_normal_granularity(const float distance) {
  return fmaxf(eps, distance / device.width);
}

__device__ float ocean_ray_underwater_length(const vec3 origin, const vec3 ray, const float limit) {
  const float max_ocean_height = OCEAN_MAX_HEIGHT;

  if (origin.y > max_ocean_height) {
    const float ref_height = get_length(world_to_sky_transform(get_vector(0.0f, 0.0f, 0.0f)));

    if (!sph_ray_hit_p0(ray, world_to_sky_transform(origin), world_to_sky_scale(max_ocean_height) + ref_height)) {
      return 0.0f;
    }
  }

  if (origin.y < device.scene.ocean.height) {
    if (ray.y < eps) {
      return limit;
    }
    else {
      return fminf(limit, (device.scene.ocean.height - origin.y) / ray.y);
    }
  }

  if (ray.y > -eps) {
    return 0.0f;
  }

  return fmaxf(0.0f, limit - (device.scene.ocean.height - origin.y) / ray.y);
}

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
  float amplitude  = device.scene.ocean.amplitude;
  float choppyness = device.scene.ocean.choppyness;
  float frequency  = device.scene.ocean.frequency;

  float2 q = make_float2(p.x * 0.75f, p.z);

  float d = 0.0f;
  float h = 0.0f;

  float t = 1.0f + device.scene.ocean.time * device.scene.ocean.speed;

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

  return p.y - h - device.scene.ocean.height;
}

__device__ vec3 ocean_get_normal(vec3 p, const float diff) {
  // Sobel filter
  float h[8];
  h[0] = ocean_get_height(add_vector(p, get_vector(-diff, 0.0f, diff)), OCEAN_ITERATIONS_NORMAL);
  h[1] = ocean_get_height(add_vector(p, get_vector(0.0f, 0.0f, diff)), OCEAN_ITERATIONS_NORMAL);
  h[2] = ocean_get_height(add_vector(p, get_vector(diff, 0.0f, diff)), OCEAN_ITERATIONS_NORMAL);
  h[3] = ocean_get_height(add_vector(p, get_vector(-diff, 0.0f, 0.0f)), OCEAN_ITERATIONS_NORMAL);
  h[4] = ocean_get_height(add_vector(p, get_vector(diff, 0.0f, 0.0f)), OCEAN_ITERATIONS_NORMAL);
  h[5] = ocean_get_height(add_vector(p, get_vector(-diff, 0.0f, -diff)), OCEAN_ITERATIONS_NORMAL);
  h[6] = ocean_get_height(add_vector(p, get_vector(0.0f, 0.0f, -diff)), OCEAN_ITERATIONS_NORMAL);
  h[7] = ocean_get_height(add_vector(p, get_vector(diff, 0.0f, -diff)), OCEAN_ITERATIONS_NORMAL);

  vec3 normal;
  normal.x = ((h[7] + 2.0f * h[4] + h[2]) - (h[5] + 2.0f * h[3] + h[0])) / 8.0f;
  normal.y = diff;
  normal.z = ((h[0] + 2.0f * h[1] + h[2]) - (h[5] + 2.0f * h[6] + h[7])) / 8.0f;

  return normalize_vector(normal);
}

__device__ float ocean_far_distance(const vec3 origin, const vec3 ray) {
  const float ref_height = get_length(world_to_sky_transform(get_vector(0.0f, 0.0f, 0.0f)));

  if (!sph_ray_hit_p0(ray, world_to_sky_transform(origin), world_to_sky_scale(OCEAN_MAX_HEIGHT) + ref_height)) {
    return FLT_MAX;
  }

  const float d1 = OCEAN_MIN_HEIGHT - origin.y;
  const float d2 = d1 + 3.0f * device.scene.ocean.amplitude;

  const float s1 = d1 / ray.y;
  const float s2 = d2 / ray.y;

  // inbetween top and bottom is inconclusive
  if (s1 * s2 < 0.0f)
    return FLT_MAX;

  const float s = fmaxf(s1, s2);

  return (s >= eps) ? s : FLT_MAX;
}

__device__ float ocean_short_distance(const vec3 origin, const vec3 ray) {
  const float ref_height = get_length(world_to_sky_transform(get_vector(0.0f, 0.0f, 0.0f)));

  if (!sph_ray_hit_p0(ray, world_to_sky_transform(origin), world_to_sky_scale(OCEAN_MAX_HEIGHT) + ref_height)) {
    return FLT_MAX;
  }

  const float d1 = OCEAN_MIN_HEIGHT - origin.y;
  const float d2 = d1 + 3.0f * device.scene.ocean.amplitude;

  const float s1 = d1 / ray.y;
  const float s2 = d2 / ray.y;

  // inbetween top and bottom is inconclusive
  if (s1 * s2 < 0.0f)
    return 0.0f;

  return fabsf(fminf(s1, s2));
}

__device__ bool ocean_is_underwater(const vec3 origin) {
  const float ref_height    = get_length(world_to_sky_transform(get_vector(0.0f, 0.0f, 0.0f)));
  const float origin_height = get_length(world_to_sky_transform(origin));

  return (origin_height < ref_height + world_to_sky_scale(OCEAN_MAX_HEIGHT));
}

__device__ float ocean_intersection_distance(const vec3 origin, const vec3 ray, float max) {
  float min = 0.0f;

  vec3 p = add_vector(origin, scale_vector(ray, max));

  float height_at_max = ocean_get_height(p, OCEAN_ITERATIONS_INTERSECTION);

  float height_at_min = ocean_get_height(origin, OCEAN_ITERATIONS_INTERSECTION);

  float mid = 0.0f;

  for (int i = 0; i < 8; i++) {
    mid = lerp(min, max, height_at_min / (height_at_min - height_at_max));
    p   = add_vector(origin, scale_vector(ray, mid));

    float height_at_mid = ocean_get_height(p, OCEAN_ITERATIONS_INTERSECTION);

    if (height_at_mid < 0.0f) {
      max           = mid;
      height_at_max = height_at_mid;
    }
    else {
      min           = mid;
      height_at_min = height_at_mid;
    }
  }

  return mid < 0.0f ? FLT_MAX : mid;
}

__global__ void ocean_generate_light_eval_data() {
  const int task_count  = device.ptrs.task_counts[(threadIdx.x + blockIdx.x * blockDim.x) * 6 + 1];
  const int task_offset = device.ptrs.task_offsets[(threadIdx.x + blockIdx.x * blockDim.x) * 5 + 1];

  for (int i = 0; i < task_count; i++) {
    OceanTask task  = load_ocean_task(device.trace_tasks + get_task_address(task_offset + i));
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

__global__ __launch_bounds__(THREADS_PER_BLOCK, 7) void process_ocean_tasks() {
  const int id = threadIdx.x + blockIdx.x * blockDim.x;

  const int task_count   = device.ptrs.task_counts[id * 6 + 1];
  const int task_offset  = device.ptrs.task_offsets[id * 5 + 1];
  int light_trace_count  = device.ptrs.light_trace_count[id];
  int bounce_trace_count = device.ptrs.bounce_trace_count[id];

  for (int i = 0; i < task_count; i++) {
    OceanTask task  = load_ocean_task(device.trace_tasks + get_task_address(task_offset + i));
    const int pixel = task.index.y * device.width + task.index.x;

    vec3 ray;
    ray.x = cosf(task.ray_xz) * cosf(task.ray_y);
    ray.y = sinf(task.ray_y);
    ray.z = sinf(task.ray_xz) * cosf(task.ray_y);

    vec3 normal = ocean_get_normal(task.position, ocean_get_normal_granularity(task.distance));

    if (ray.y > 0.0f) {
      normal = scale_vector(normal, -1.0f);
    }

    RGBAF albedo = device.scene.ocean.albedo;
    RGBF record  = device.records[pixel];

    if (device.scene.ocean.emissive) {
      RGBF emission = get_color(albedo.r, albedo.g, albedo.b);

      write_albedo_buffer(emission, pixel);

      emission.r *= record.r;
      emission.g *= record.g;
      emission.b *= record.b;

      device.ptrs.frame_buffer[pixel] = RGBF_to_RGBAhalf(emission);
    }

    write_normal_buffer(normal, pixel);

    const vec3 V      = scale_vector(ray, -1.0f);
    BRDFInstance brdf = brdf_get_instance(RGBAF_to_RGBAhalf(albedo), V, normal, 0.0f, 1.0f);
    brdf.specular_f0  = get_RGBAhalf(0.02f, 0.02f, 0.02f, 0.02f);

    if (white_noise() > albedo.a) {
      task.position = add_vector(task.position, scale_vector(ray, eps * get_length(task.position)));

      const float refraction_index = 1.0f / device.scene.ocean.refractive_index;

      brdf = brdf_sample_ray_refraction(brdf, refraction_index, 0.0f, 0.0f);

      RGBF alpha_record = record;

      TraceTask new_task;
      new_task.origin = task.position;
      new_task.ray    = brdf.L;
      new_task.index  = task.index;

      switch (device.iteration_type) {
        case TYPE_CAMERA:
        case TYPE_BOUNCE:
          store_RGBF(device.ptrs.bounce_records + pixel, alpha_record);
          store_trace_task(device.ptrs.bounce_trace + get_task_address(bounce_trace_count++), new_task);
          break;
        case TYPE_LIGHT:
          if (white_noise() > 0.5f)
            break;
          store_RGBF(device.ptrs.light_records + pixel, scale_color(alpha_record, 2.0f));
          store_trace_task(device.ptrs.light_trace + get_task_address(light_trace_count++), new_task);
          device.ptrs.state_buffer[pixel] |= STATE_LIGHT_OCCUPIED;
          break;
      }
    }
    else if (device.iteration_type != TYPE_LIGHT) {
      const float scattering_prob = (ray.y > 0.0f) ? 0.0f : (1.0f - albedo.a);
      const int scattering_pass   = white_noise() < scattering_prob && !(device.ptrs.state_buffer[pixel] & STATE_LIGHT_OCCUPIED);

      task.position = add_vector(task.position, scale_vector(ray, -eps * get_length(task.position)));

      uint32_t light_history_buffer_entry = LIGHT_ID_ANY;

      // TODO: change this so that this coincides with the transparent pass from above, this here has nothing to do with surface bounces!
      if (scattering_pass) {
        LightSample light = load_light_sample(device.ptrs.light_samples, pixel);

        float underwater_sample;
        vec3 light_pos;

        underwater_sample            = 10.0f * white_noise();
        const float refraction_index = 1.0f / device.scene.ocean.refractive_index;
        BRDFInstance brdf2           = brdf_sample_ray_refraction(brdf, refraction_index, 0.0f, 0.0f);
        light_pos                    = add_vector(task.position, scale_vector(brdf2.L, underwater_sample));

        if (light.weight > 0.0f) {
          RGBF light_record;

          BRDFInstance brdf_sample = brdf_apply_sample_scattering(brdf, light, light_pos, 50.0f);

          const RGBF S = mul_color(device.scene.ocean.scattering, scale_color(brdf_sample.term, OCEAN_POLLUTION));

          RGBF extinction = OCEAN_EXTINCTION;

          // Amount of light that gets lost along this step
          RGBF step_transmittance;
          step_transmittance.r = expf(-underwater_sample * extinction.r);
          step_transmittance.g = expf(-underwater_sample * extinction.g);
          step_transmittance.b = expf(-underwater_sample * extinction.b);

          const RGBF weight = mul_color(sub_color(S, mul_color(S, step_transmittance)), inv_color(extinction));

          light_record = mul_color(record, weight);
          light_record = scale_color(light_record, 1.0f / scattering_prob);

          TraceTask light_task;
          light_task.origin = light_pos;
          light_task.ray    = brdf_sample.L;
          light_task.index  = task.index;

          if (luminance(light_record) > 0.0f) {
            store_RGBF(device.ptrs.light_records + pixel, light_record);
            light_history_buffer_entry = light.id;
            store_trace_task(device.ptrs.light_trace + get_task_address(light_trace_count++), light_task);
          }
        }
      }

      if (!(device.ptrs.state_buffer[pixel] & STATE_LIGHT_OCCUPIED))
        device.ptrs.light_sample_history[pixel] = light_history_buffer_entry;

      brdf = brdf_sample_ray(brdf, task.index);

      RGBF bounce_record = mul_color(record, brdf.term);

      TraceTask bounce_task;
      bounce_task.origin = task.position;
      bounce_task.ray    = brdf.L;
      bounce_task.index  = task.index;

      store_RGBF(device.ptrs.bounce_records + pixel, bounce_record);
      store_trace_task(device.ptrs.bounce_trace + get_task_address(bounce_trace_count++), bounce_task);
    }
  }

  device.ptrs.light_trace_count[id]  = light_trace_count;
  device.ptrs.bounce_trace_count[id] = bounce_trace_count;
}

__global__ __launch_bounds__(THREADS_PER_BLOCK, 10) void process_debug_ocean_tasks() {
  const int id = threadIdx.x + blockIdx.x * blockDim.x;

  const int task_count  = device.ptrs.task_counts[id * 6 + 1];
  const int task_offset = device.ptrs.task_offsets[id * 5 + 1];

  for (int i = 0; i < task_count; i++) {
    OceanTask task  = load_ocean_task(device.trace_tasks + get_task_address(task_offset + i));
    const int pixel = task.index.y * device.width + task.index.x;

    if (device.shading_mode == SHADING_ALBEDO) {
      RGBAF albedo                    = device.scene.ocean.albedo;
      device.ptrs.frame_buffer[pixel] = RGBF_to_RGBAhalf(get_color(albedo.r, albedo.g, albedo.b));
    }
    else if (device.shading_mode == SHADING_DEPTH) {
      const float value               = __saturatef((1.0f / task.distance) * 2.0f);
      device.ptrs.frame_buffer[pixel] = RGBF_to_RGBAhalf(get_color(value, value, value));
    }
    else if (device.shading_mode == SHADING_NORMAL) {
      vec3 normal = ocean_get_normal(task.position, ocean_get_normal_granularity(task.distance));

      normal.x = 0.5f * normal.x + 0.5f;
      normal.y = 0.5f * normal.y + 0.5f;
      normal.z = 0.5f * normal.z + 0.5f;

      device.ptrs.frame_buffer[pixel] = RGBF_to_RGBAhalf(get_color(__saturatef(normal.x), __saturatef(normal.y), __saturatef(normal.z)));
    }
    else if (device.shading_mode == SHADING_IDENTIFICATION) {
      const RGBAhalf color = get_RGBAhalf(0.0f, 0.0f, 1.0f, 0.0f);

      store_RGBAhalf(device.ptrs.frame_buffer + pixel, color);
    }
  }
}

#endif /* CU_OCEAN_H */
