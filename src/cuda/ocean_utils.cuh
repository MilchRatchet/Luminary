#ifndef CU_OCEAN_UTILS_H
#define CU_OCEAN_UTILS_H

#include "sky_utils.cuh"
#include "utils.cuh"

#define OCEAN_MAX_HEIGHT (device.scene.ocean.height + 2.66f * device.scene.ocean.amplitude)
#define OCEAN_MIN_HEIGHT (device.scene.ocean.height)
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

  return powf(1.0f - sqrtf(wave1.x * wave1.y), choppyness);
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

// Coefficients taken from
// M. Droske, J. Hanika, J. Vorba, A. Weidlich, M. Sabbadin, _Path Tracing in Production: The Path of Water_, ACM SIGGRAPH 2023 Courses,
// 2023.

__device__ RGBF ocean_jerlov_scattering_coefficient(const JerlovWaterType type) {
  switch (type) {
    case JERLOV_WATER_TYPE_I:
      return get_color(0.001f, 0.002f, 0.004f);
    case JERLOV_WATER_TYPE_IA:
      return get_color(0.002f, 0.004f, 0.007f);
    case JERLOV_WATER_TYPE_IB:
      return get_color(0.045f, 0.054f, 0.07f);
    case JERLOV_WATER_TYPE_II:
      return get_color(0.27f, 0.365f, 0.516f);
    case JERLOV_WATER_TYPE_III:
      return get_color(0.737f, 0.998f, 1.413f);
    case JERLOV_WATER_TYPE_1C:
      return get_color(0.274f, 0.372f, 0.526f);
    case JERLOV_WATER_TYPE_3C:
      return get_color(0.904f, 1.071f, 1.532f);
    case JERLOV_WATER_TYPE_5C:
      return get_color(3.589f, 1.382f, 1.857f);
    case JERLOV_WATER_TYPE_7C:
      return get_color(1.772f, 2.394f, 3.376f);
    case JERLOV_WATER_TYPE_9C:
      return get_color(2.347f, 3.18f, 4.496f);
  }

  return get_color(0.0f, 0.0f, 0.0f);
}

__device__ RGBF ocean_jerlov_absorption_coefficient(const JerlovWaterType type) {
  switch (type) {
    case JERLOV_WATER_TYPE_I:
      return get_color(0.309f, 0.053f, 0.009f);
    case JERLOV_WATER_TYPE_IA:
      return get_color(0.309f, 0.054f, 0.014f);
    case JERLOV_WATER_TYPE_IB:
      return get_color(0.309f, 0.054f, 0.015f);
    case JERLOV_WATER_TYPE_II:
      return get_color(0.31f, 0.054f, 0.016f);
    case JERLOV_WATER_TYPE_III:
      return get_color(0.31f, 0.056f, 0.031f);
    case JERLOV_WATER_TYPE_1C:
      return get_color(0.316f, 0.067f, 0.105f);
    case JERLOV_WATER_TYPE_3C:
      return get_color(0.508f, 0.052f, 0.161f);
    case JERLOV_WATER_TYPE_5C:
      return get_color(4.638f, 0.222f, 0.216f);
    case JERLOV_WATER_TYPE_7C:
      return get_color(0.351f, 0.188f, 0.574f);
    case JERLOV_WATER_TYPE_9C:
      return get_color(0.398f, 0.349f, 0.995f);
  }

  return get_color(0.0f, 0.0f, 0.0f);
}

__device__ float ocean_molecular_weight(const JerlovWaterType type) {
  switch (type) {
    case JERLOV_WATER_TYPE_I:
      return 0.93f;
    case JERLOV_WATER_TYPE_IA:
      return 0.44f;
    case JERLOV_WATER_TYPE_IB:
      return 0.06f;
    case JERLOV_WATER_TYPE_II:
      return 0.007f;
    case JERLOV_WATER_TYPE_III:
      return 0.003f;
    case JERLOV_WATER_TYPE_1C:
      return 0.005f;
    case JERLOV_WATER_TYPE_3C:
      return 0.003f;
    case JERLOV_WATER_TYPE_5C:
      return 0.001f;
    case JERLOV_WATER_TYPE_7C:
      return 0.0f;
    case JERLOV_WATER_TYPE_9C:
      return 0.0f;
  }

  return 0.0f;
}

// Henyey Greenstein importance sampling for g = 0
// pbrt v3 - Light Transport II: Volume Rendering - Sampling Volume Scattering
__device__ float ocean_molecular_phase_sampling_cosine(const vec3 ray, const float r) {
  return 2.0f * r - 1.0f;
}

// Henyey Greenstein importance sampling for g != 0
// pbrt v3 - Light Transport II: Volume Rendering - Sampling Volume Scattering
__device__ float ocean_particle_phase_sampling_cosine(const vec3 ray, const float r) {
  const float g = 0.924f;

  float denom = (1.0f - g + 2.0f * g * r);
  if (fabsf(denom) < eps) {
    denom = copysignf(eps, denom);
  }

  const float s = (1.0f - g * g) / denom;

  return (1.0f + g * g - s * s) / (2.0f * g);
}

__device__ vec3 ocean_phase_sampling(const vec3 ray, const float2 r_dir, const float r_choice) {
  const float molecular_weight = ocean_molecular_weight(device.scene.ocean.water_type);

  float cos_angle;
  if (r_choice < molecular_weight) {
    cos_angle = ocean_molecular_phase_sampling_cosine(ray, r_dir.x);
  }
  else {
    cos_angle = ocean_particle_phase_sampling_cosine(ray, r_dir.x);
  }

  return phase_sample_basis(cos_angle, r_dir.y, ray);
}

__device__ float ocean_molecular_phase(const float cos_angle) {
  return henyey_greenstein_phase_function(cos_angle, 0.0f);
}

__device__ float ocean_particle_phase(const float cos_angle) {
  return henyey_greenstein_phase_function(cos_angle, 0.924f);
}

__device__ float ocean_phase(const float cos_angle) {
  const float molecular_weight = ocean_molecular_weight(device.scene.ocean.water_type);

  const float molecular_phase = ocean_molecular_phase(cos_angle);
  const float particle_phase  = ocean_particle_phase(cos_angle);

  return molecular_phase * molecular_weight + particle_phase * (1.0f - molecular_weight);
}

#endif /* CU_OCEAN_UTILS_H */
