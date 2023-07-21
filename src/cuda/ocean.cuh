/*
 * Wave generation based on a shadertoy by Alexander Alekseev aka TDM (2014)
 * The shadertoy can be found on https://www.shadertoy.com/view/Ms2SD1
 */
#ifndef CU_OCEAN_H
#define CU_OCEAN_H

#include "brdf.cuh"
#include "math.cuh"
#include "memory.cuh"

#define OCEAN_POLLUTION (device.scene.ocean.pollution * 0.1f)
#define OCEAN_SCATTERING (scale_color(device.scene.ocean.scattering, OCEAN_POLLUTION))
#define OCEAN_ABSORPTION (scale_color(device.scene.ocean.absorption, device.scene.ocean.absorption_strength * 0.1f))
#define OCEAN_EXTINCTION (add_color(OCEAN_SCATTERING, OCEAN_ABSORPTION))

#define OCEAN_MAX_HEIGHT (device.scene.ocean.height + 2.66f * device.scene.ocean.amplitude)
#define OCEAN_MIN_HEIGHT (device.scene.ocean.height)

#define OCEAN_LIPSCHITZ (device.scene.ocean.amplitude * 3.0f)

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

  const float d1 = OCEAN_MIN_HEIGHT - origin.y;
  const float d2 = OCEAN_MAX_HEIGHT - origin.y;

  const float s1 = d1 / ray.y;
  const float s2 = d2 / ray.y;

  if (s1 < 0.0f && s2 < 0.0f)
    return FLT_MAX;

  return (s1 * s2 < 0.0f) ? fmaxf(s1, s2) : fminf(s1, s2);
}

__device__ float ocean_intersection_solver(const vec3 origin, const vec3 ray, const float start, const float limit) {
  const float target_residual = 400.0f * device.scene.ocean.amplitude * eps;

  float min = start + target_residual;
  float max = limit;

  const float far_distance = ocean_far_distance(origin, ray);

  if (far_distance == FLT_MAX) {
    return FLT_MAX;
  }

  max = fminf(max, far_distance);

  vec3 p;

  p                     = add_vector(origin, scale_vector(ray, max));
  float residual_at_max = p.y - ocean_get_height(p, OCEAN_ITERATIONS_INTERSECTION) - device.scene.ocean.height;

  p                     = add_vector(origin, scale_vector(ray, min));
  float residual_at_min = p.y - ocean_get_height(p, OCEAN_ITERATIONS_INTERSECTION) - device.scene.ocean.height;

  float mid = 0.0f;

  for (int i = 0; i < 20; i++) {
    mid = lerp(min, max, residual_at_min / (residual_at_min - residual_at_max));
    p   = add_vector(origin, scale_vector(ray, mid));

    float residual_at_mid = p.y - ocean_get_height(p, OCEAN_ITERATIONS_INTERSECTION) - device.scene.ocean.height;
    if (residual_at_mid < 0.0f) {
      if (ray.y > 0.0f) {
        min             = mid;
        residual_at_min = residual_at_mid;
      }
      else {
        max             = mid;
        residual_at_max = residual_at_mid;
      }
    }
    else {
      if (ray.y > 0.0f) {
        max             = mid;
        residual_at_max = residual_at_mid;
      }
      else {
        min             = mid;
        residual_at_min = residual_at_mid;
      }
    }
  }

  return (mid < target_residual) ? FLT_MAX : mid;
}

__device__ float ocean_ray_marcher(const vec3 origin, const vec3 ray, const float start, const float limit) {
  const float target_residual = 400.0f * device.scene.ocean.amplitude * eps;

  float t = start + target_residual;

  const float slope_confidence_factor = 1.0f / (OCEAN_LIPSCHITZ + fabsf(ray.y));

  for (int i = 0; i < 500; i++) {
    const vec3 p = add_vector(origin, scale_vector(ray, t));

    const float residual_at_t = fabsf(p.y - ocean_get_height(p, OCEAN_ITERATIONS_INTERSECTION) - device.scene.ocean.height);

    if (residual_at_t < target_residual)
      return t;

    t += residual_at_t * slope_confidence_factor;

    if (t >= limit)
      break;
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

  if (device.iteration_type != TYPE_LIGHT) {
    if (get_length(sub_vector(add_vector(origin, scale_vector(ray, start)), device.scene.camera.pos)) > 30.0f) {
      return ocean_intersection_solver(origin, ray, start, limit);
    }
    else {
      return ocean_ray_marcher(origin, ray, start, limit);
    }
  }
  else {
    return ocean_intersection_solver(origin, ray, start, limit);
  }

  return FLT_MAX;
}

__device__ RGBF ocean_brdf(const vec3 ray, const vec3 normal) {
  const float dot = dot_product(ray, normal);

  if (dot < 0.0f) {
    return get_color(1.0f, 1.0f, 1.0f);
  }

  RGBAhalf specular_f0 = get_RGBAhalf(0.02f, 0.02f, 0.02f, 0.0f);
  switch (device.scene.material.fresnel) {
    case SCHLICK:
      return RGBAhalf_to_RGBF(brdf_fresnel_schlick(specular_f0, brdf_shadowed_F90(specular_f0), dot));
      break;
    case FDEZ_AGUERA:
    default:
      return RGBAhalf_to_RGBF(brdf_fresnel_roughness(specular_f0, 0.0f, dot));
      break;
  }
}

////////////////////////////////////////////////////////////////////
// Kernel
////////////////////////////////////////////////////////////////////

__global__ __launch_bounds__(THREADS_PER_BLOCK, 12) void ocean_depth_trace_tasks() {
  const int task_count = device.trace_count[threadIdx.x + blockIdx.x * blockDim.x];

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
          result.y = __uint_as_float(OCEAN_HIT);
          __stcs((float2*) (device.ptrs.trace_results + offset), result);
        }
      }
    }
    else if (device.scene.ocean.albedo.a > 0.0f) {
      const float ocean_depth = ocean_intersection_distance(task.origin, task.ray, depth);

      if (ocean_depth < depth) {
        const int pixel = task.index.y * device.width + task.index.x;
        store_RGBF(device.records + pixel, scale_color(load_RGBF(device.records + pixel), 1.0f - device.scene.ocean.albedo.a));
      }
    }
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

    vec3 normal = ocean_get_normal(task.position);

    float refraction_index_ratio;
    if (dot_product(ray, normal) > 0.0f) {
      normal                 = scale_vector(normal, -1.0f);
      refraction_index_ratio = device.scene.ocean.refractive_index;
    }
    else {
      refraction_index_ratio = 1.0f / device.scene.ocean.refractive_index;
    }

    RGBF record = device.records[pixel];

    if (device.scene.ocean.emissive) {
      RGBF emission = get_color(device.scene.ocean.albedo.r, device.scene.ocean.albedo.g, device.scene.ocean.albedo.b);

      write_albedo_buffer(emission, pixel);

      emission.r *= record.r;
      emission.g *= record.g;
      emission.b *= record.b;

      // TODO: This is wrong.
      device.ptrs.frame_buffer[pixel] = RGBF_to_RGBAhalf(emission);
    }

    write_normal_buffer(normal, pixel);

    if (white_noise() > device.scene.ocean.albedo.a) {
      ray           = refract_ray(ray, normal, refraction_index_ratio);
      task.position = add_vector(task.position, scale_vector(ray, eps * get_length(task.position)));
    }
    else {
      task.position = add_vector(task.position, scale_vector(ray, -eps * get_length(task.position)));
      ray           = reflect_vector(ray, normal);
    }

    record = mul_color(record, ocean_brdf(ray, normal));

    TraceTask new_task;
    new_task.origin = task.position;
    new_task.ray    = ray;
    new_task.index  = task.index;

    store_RGBF(device.ptrs.bounce_records + pixel, record);
    store_trace_task(device.ptrs.bounce_trace + get_task_address(bounce_trace_count++), new_task);
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
      vec3 normal = ocean_get_normal(task.position);

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
