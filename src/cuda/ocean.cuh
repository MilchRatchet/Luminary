#ifndef CU_OCEAN_H
#define CU_OCEAN_H

#include "brdf.cuh"
#include "math.cuh"
#include "memory.cuh"
#include "ocean_utils.cuh"

//
// In this ocean implementation the surface shape is defined by a function based on the shadertoy by
// Alexander Alekseev aka TDM (https://www.shadertoy.com/view/Ms2SD1).
// The intersection of the ray with the surface is handled through a ray marcher that uses an
// approximate Lipschitz factor of the surface function to obtain a function similar to an SDF.
// The shading of the ocean and the water beneath is based on
// M. Droske, J. Hanika, J. Vorba, A. Weidlich, M. Sabbadin, "Path Tracing in Production: The Path of Water", ACM SIGGRAPH 2023 Courses,
// 2023.
// The water is handled by the volume implementation.
//

#define OCEAN_LIPSCHITZ (device.scene.ocean.amplitude * 4.0f)

#define OCEAN_ITERATIONS_INTERSECTION 5
#define OCEAN_ITERATIONS_NORMAL 8

__device__ float ocean_hash(const float2 p) {
  const float x = p.x * 127.1f + p.y * 311.7f;
  return fractf(sinf(x) * 43758.5453123f);
}

__device__ float ocean_noise(const float2 p) {
  float2 integral;
  integral.x = floorf(p.x);
  integral.y = floorf(p.y);

  float2 fractional = make_float2(p.x - integral.x, p.y - integral.y);

  fractional.x = fractional.x * fractional.x * (3.0f - 2.0f * fractional.x);
  fractional.y = fractional.y * fractional.y * (3.0f - 2.0f * fractional.y);

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

  float h = 0.0f;

  for (int i = 0; i < steps; i++) {
    float2 a = make_float2(q.x * frequency, q.y * frequency);
    h += ocean_octave(a, choppyness) * amplitude;

    const float u = q.x;
    const float v = q.y;
    q.x           = 1.6f * u - 1.2f * v;
    q.y           = 1.2f * u + 1.6f * v;

    frequency *= 1.9f;
    amplitude *= 0.22f;
    choppyness = lerp(choppyness, 1.0f, 0.2f);
  }

  h *= 2.0f;

  return h;
}

__device__ float ocean_get_relative_height(const vec3 p, const int steps) {
  return (p.y - device.scene.ocean.height) - ocean_get_height(p, steps);
}

__device__ vec3 ocean_get_normal(const vec3 p) {
  const float d = (OCEAN_LIPSCHITZ + get_length(p)) * eps;

  // Sobel filter
  float h[8];
  h[0] = ocean_get_height(add_vector(p, get_vector(-d, 0.0f, d)), OCEAN_ITERATIONS_NORMAL);
  h[1] = ocean_get_height(add_vector(p, get_vector(0.0f, 0.0f, d)), OCEAN_ITERATIONS_NORMAL);
  h[2] = ocean_get_height(add_vector(p, get_vector(d, 0.0f, d)), OCEAN_ITERATIONS_NORMAL);
  h[3] = ocean_get_height(add_vector(p, get_vector(-d, 0.0f, 0.0f)), OCEAN_ITERATIONS_NORMAL);
  h[4] = ocean_get_height(add_vector(p, get_vector(d, 0.0f, 0.0f)), OCEAN_ITERATIONS_NORMAL);
  h[5] = ocean_get_height(add_vector(p, get_vector(-d, 0.0f, -d)), OCEAN_ITERATIONS_NORMAL);
  h[6] = ocean_get_height(add_vector(p, get_vector(0.0f, 0.0f, -d)), OCEAN_ITERATIONS_NORMAL);
  h[7] = ocean_get_height(add_vector(p, get_vector(d, 0.0f, -d)), OCEAN_ITERATIONS_NORMAL);

  vec3 normal;
  normal.x = -((h[7] + 2.0f * h[4] + h[2]) - (h[5] + 2.0f * h[3] + h[0])) / 8.0f;
  normal.y = d;
  normal.z = -((h[0] + 2.0f * h[1] + h[2]) - (h[5] + 2.0f * h[6] + h[7])) / 8.0f;

  return normalize_vector(normal);
}

// FLT_MAX signals no hit.
__device__ float ocean_far_distance(const vec3 origin, const vec3 ray) {
  if (!sph_ray_hit_p0(ray, world_to_sky_transform(origin), world_to_sky_scale(OCEAN_MAX_HEIGHT) + SKY_WORLD_REFERENCE_HEIGHT)) {
    return FLT_MAX;
  }

  if (fabsf(ray.y) < eps) {
    return FLT_MAX;
  }

  const float d1 = OCEAN_MIN_HEIGHT - origin.y;
  const float d2 = OCEAN_MAX_HEIGHT - origin.y;

  const float s1 = d1 / ray.y;
  const float s2 = d2 / ray.y;

  const float s = fmaxf(s1, s2);

  return (s >= eps) ? s : FLT_MAX;
}

// FLT_MAX signals no hit.
__device__ float ocean_short_distance(const vec3 origin, const vec3 ray) {
  if (!sph_ray_hit_p0(ray, world_to_sky_transform(origin), world_to_sky_scale(OCEAN_MAX_HEIGHT) + SKY_WORLD_REFERENCE_HEIGHT)) {
    return FLT_MAX;
  }

  if (fabsf(ray.y) < eps) {
    return (origin.y >= OCEAN_MIN_HEIGHT && origin.y <= OCEAN_MAX_HEIGHT) ? 0.0f : FLT_MAX;
  }

  const float d1 = OCEAN_MIN_HEIGHT - origin.y;
  const float d2 = OCEAN_MAX_HEIGHT - origin.y;

  const float s1 = d1 / ray.y;
  const float s2 = d2 / ray.y;

  if (s1 < 0.0f && s2 < 0.0f)
    return FLT_MAX;

  return (s1 * s2 < 0.0f) ? fmaxf(s1, s2) : fminf(s1, s2);
}

__device__ float ocean_intersection_solver(const vec3 origin, const vec3 ray, const float start, const float limit) {
  const float target_residual = (1.0f + fabsf(device.scene.ocean.height) + start / 10.0f) * 0.5f * eps;

  float t                       = start;
  float last_residual           = 0.0f;
  float slope_confidence_factor = 6.0f / (OCEAN_LIPSCHITZ + fabsf(ray.y));

  for (int i = 0; i < 200; i++) {
    const vec3 p = add_vector(origin, scale_vector(ray, t));

    const float residual_at_t = ocean_get_relative_height(p, OCEAN_ITERATIONS_INTERSECTION);
    const float res_abs       = fabsf(residual_at_t);

    if (res_abs < target_residual)
      return t;

    if (last_residual * residual_at_t < 0.0f) {
      slope_confidence_factor *= -0.5f;
    }

    last_residual = residual_at_t;

    const float step_size = fminf(0.1f * (limit - start), res_abs * fabsf(slope_confidence_factor));

    t += copysignf(step_size, slope_confidence_factor);

    // Sometimes we may overstep beyond the limit and then require to backtrack, hence we abort
    // only if we are far beyond the limit.
    if (t >= limit + 0.2f * (limit - start)) {
      break;
    }
    else if (t <= start - 0.2f * (limit - start)) {
      break;
    }
  }

  return FLT_MAX;
}

__device__ float ocean_intersection_distance(const vec3 origin, const vec3 ray, const float limit) {
  float start = 0.0f;

  if (origin.y < OCEAN_MIN_HEIGHT || origin.y > OCEAN_MAX_HEIGHT) {
    const float short_distance = ocean_short_distance(origin, ray);

    if (short_distance == FLT_MAX) {
      return FLT_MAX;
    }

    start = short_distance;
  }

  if (device.scene.ocean.amplitude == 0.0f) {
    return start;
  }

  const float end = fminf(limit, ocean_far_distance(origin, ray));

  return ocean_intersection_solver(origin, ray, start, end);
}

__device__ RGBF ocean_brdf(const vec3 ray, const vec3 normal) {
  const float dot = dot_product(ray, normal);

  if (dot < 0.0f) {
    return get_color(1.0f, 1.0f, 1.0f);
  }

  RGBF specular_f0 = get_color(0.02f, 0.02f, 0.02f);
  switch (device.scene.material.fresnel) {
    case SCHLICK:
      return brdf_fresnel_schlick(specular_f0, brdf_shadowed_F90(specular_f0), dot);
      break;
    case FDEZ_AGUERA:
    default:
      return brdf_fresnel_roughness(specular_f0, 0.0f, dot);
      break;
  }
}

////////////////////////////////////////////////////////////////////
// Kernel
////////////////////////////////////////////////////////////////////

__global__ __launch_bounds__(THREADS_PER_BLOCK, 12) void ocean_depth_trace_tasks() {
  const int task_count = device.trace_count[THREAD_ID];

  for (int i = 0; i < task_count; i++) {
    const int offset  = get_task_address(i);
    TraceTask task    = load_trace_task(device.trace_tasks + offset);
    const float depth = __ldcs(&((device.ptrs.trace_results + offset)->depth));

    if (device.iteration_type != TYPE_LIGHT) {
      bool intersection_possible = true;
      if (task.origin.y < OCEAN_MIN_HEIGHT || task.origin.y > OCEAN_MAX_HEIGHT) {
        const float short_distance = ocean_short_distance(task.origin, task.ray);
        intersection_possible      = (short_distance != FLT_MAX) && (short_distance <= depth);
      }

      if (intersection_possible) {
        const float ocean_depth = ocean_intersection_distance(task.origin, task.ray, depth);

        if (ocean_depth < depth) {
          float2 result;
          result.x = ocean_depth;
          result.y = __uint_as_float(HIT_TYPE_OCEAN);
          __stcs((float2*) (device.ptrs.trace_results + offset), result);
        }
      }
    }
  }
}

__global__ __launch_bounds__(THREADS_PER_BLOCK, 7) void process_ocean_tasks() {
  const int task_count   = device.ptrs.task_counts[THREAD_ID * TASK_ADDRESS_COUNT_STRIDE + TASK_ADDRESS_OFFSET_OCEAN];
  const int task_offset  = device.ptrs.task_offsets[THREAD_ID * TASK_ADDRESS_OFFSET_STRIDE + TASK_ADDRESS_OFFSET_OCEAN];
  int light_trace_count  = device.ptrs.light_trace_count[THREAD_ID];
  int bounce_trace_count = device.ptrs.bounce_trace_count[THREAD_ID];

  for (int i = 0; i < task_count; i++) {
    OceanTask task  = load_ocean_task(device.trace_tasks + get_task_address(task_offset + i));
    const int pixel = task.index.y * device.width + task.index.x;

    vec3 normal = ocean_get_normal(task.position);

    float refraction_index_ratio;
    if (dot_product(task.ray, normal) > 0.0f) {
      normal                 = scale_vector(normal, -1.0f);
      refraction_index_ratio = device.scene.ocean.refractive_index;
    }
    else {
      refraction_index_ratio = 1.0f / device.scene.ocean.refractive_index;
    }

    write_normal_buffer(normal, pixel);

    if (white_noise() > 0.5f) {
      task.ray      = refract_ray(task.ray, normal, refraction_index_ratio);
      task.position = add_vector(task.position, scale_vector(task.ray, 2.0f * eps * (1.0f + get_length(task.position))));
    }
    else {
      task.position = add_vector(task.position, scale_vector(task.ray, -2.0f * eps * (1.0f + get_length(task.position))));
      task.ray      = reflect_vector(task.ray, normal);
    }

    RGBF record = load_RGBF(device.records + pixel);
    record      = mul_color(record, ocean_brdf(task.ray, normal));
    record      = scale_color(record, 2.0f);  // 1.0 / probability of refraction/reflection

    TraceTask new_task;
    new_task.origin = task.position;
    new_task.ray    = task.ray;
    new_task.index  = task.index;

    device.ptrs.mis_buffer[pixel] = 1.0f;
    store_RGBF(device.ptrs.bounce_records + pixel, record);
    store_trace_task(device.ptrs.bounce_trace + get_task_address(bounce_trace_count++), new_task);

    if (!state_peek(pixel, STATE_FLAG_LIGHT_OCCUPIED)) {
      // If no light ray is queued yet, make sure that this bounce ray is allowed to gather energy from any light it hits.
      device.ptrs.light_sample_history[pixel] = LIGHT_ID_ANY;
    }
  }

  device.ptrs.light_trace_count[THREAD_ID]  = light_trace_count;
  device.ptrs.bounce_trace_count[THREAD_ID] = bounce_trace_count;
}

__global__ __launch_bounds__(THREADS_PER_BLOCK, 10) void process_debug_ocean_tasks() {
  const int task_count  = device.ptrs.task_counts[THREAD_ID * TASK_ADDRESS_COUNT_STRIDE + TASK_ADDRESS_OFFSET_OCEAN];
  const int task_offset = device.ptrs.task_offsets[THREAD_ID * TASK_ADDRESS_OFFSET_STRIDE + TASK_ADDRESS_OFFSET_OCEAN];

  for (int i = 0; i < task_count; i++) {
    OceanTask task  = load_ocean_task(device.trace_tasks + get_task_address(task_offset + i));
    const int pixel = task.index.y * device.width + task.index.x;

    if (device.shading_mode == SHADING_ALBEDO) {
      device.ptrs.frame_buffer[pixel] = get_color(0.0f, 0.0f, 0.0f);
    }
    else if (device.shading_mode == SHADING_DEPTH) {
      const float dist                = get_length(sub_vector(device.scene.camera.pos, task.position));
      const float value               = __saturatef((1.0f / dist) * 2.0f);
      device.ptrs.frame_buffer[pixel] = get_color(value, value, value);
    }
    else if (device.shading_mode == SHADING_NORMAL) {
      vec3 normal = ocean_get_normal(task.position);

      normal.x = 0.5f * normal.x + 0.5f;
      normal.y = 0.5f * normal.y + 0.5f;
      normal.z = 0.5f * normal.z + 0.5f;

      device.ptrs.frame_buffer[pixel] = get_color(__saturatef(normal.x), __saturatef(normal.y), __saturatef(normal.z));
    }
    else if (device.shading_mode == SHADING_IDENTIFICATION) {
      store_RGBF(device.ptrs.frame_buffer + pixel, get_color(0.0f, 0.0f, 1.0f));
    }
  }
}

#endif /* CU_OCEAN_H */
