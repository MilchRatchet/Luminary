#ifndef CU_OCEAN_UTILS_H
#define CU_OCEAN_UTILS_H

#include "ior_stack.cuh"
#include "material.cuh"
#include "sky_utils.cuh"
#include "utils.cuh"

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

#define OCEAN_MAX_HEIGHT (device.ocean.height + 1.33f * device.ocean.amplitude)
#define OCEAN_MIN_HEIGHT (device.ocean.height)
#define OCEAN_LIPSCHITZ (device.ocean.amplitude * 2.0f)

#define OCEAN_ITERATIONS_INTERSECTION 8
#define OCEAN_ITERATIONS_NORMAL 8
#define OCEAN_ITERATIONS_NORMAL_CAUSTICS 8

__device__ float ocean_hash(const float2 p) {
  const float x = p.x + p.y * (311.7f / 127.1f);
  return (__float_as_uint(x) * 2654435769u) * (1.0f / 0xFFFFFFFFu);
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

__device__ float ocean_octave(float2 p) {
  const float offset = ocean_noise(p);
  p.x += offset;
  p.y += offset;

  float sin_x, sin_y, cos_x, cos_y;
  sincosf(p.x, &sin_x, &cos_x);
  sincosf(p.y, &sin_y, &cos_y);

  float2 wave1;
  wave1.x = 1.0f - fabsf(sin_x);
  wave1.y = 1.0f - fabsf(sin_y);

  float2 wave2;
  wave2.x = fabsf(cos_x);
  wave2.y = fabsf(cos_y);

  wave1.x = lerp(wave1.x, wave2.x, wave1.x);
  wave1.y = lerp(wave1.y, wave2.y, wave1.y);

  float octave = 1.0f - sqrtf(wave1.x * wave1.y);
  octave *= octave;

  return octave;
}

__device__ float ocean_get_height(const vec3 p, const int steps) {
  float amplitude = 1.0f;
  float frequency = device.ocean.frequency;

  float2 q = make_float2(p.x * 0.75f, p.z);

  float h = 0.0f;

  for (int i = 0; i < steps; i++) {
    const float2 a = make_float2(q.x * frequency, q.y * frequency);
    h += ocean_octave(a) * amplitude;

    const float u = q.x;
    const float v = q.y;
    q.x           = 1.6f * u - 1.2f * v;
    q.y           = 1.2f * u + 1.6f * v;

    frequency *= 1.9f;
    amplitude *= 0.22f;
  }

  h *= device.ocean.amplitude;

  return h;
}

__device__ float ocean_get_relative_height(const vec3 p, const int steps) {
  return p.y - (device.ocean.height + ocean_get_height(p, steps));
}

__device__ bool ocean_is_underwater(const vec3 p) {
  return ocean_get_relative_height(p, OCEAN_ITERATIONS_INTERSECTION) < 0.0f;
}

__device__ vec3 ocean_get_normal(const vec3 p, const uint32_t iterations = OCEAN_ITERATIONS_NORMAL) {
  const float d = (OCEAN_LIPSCHITZ + get_length(p)) * eps;

  // Sobel filter
  const float h_0 = ocean_get_height(add_vector(p, get_vector(-d, 0.0f, d)), iterations);
  const float h_1 = ocean_get_height(add_vector(p, get_vector(0.0f, 0.0f, d)), iterations);
  const float h_2 = ocean_get_height(add_vector(p, get_vector(d, 0.0f, d)), iterations);
  const float h_3 = ocean_get_height(add_vector(p, get_vector(-d, 0.0f, 0.0f)), iterations);
  const float h_4 = ocean_get_height(add_vector(p, get_vector(d, 0.0f, 0.0f)), iterations);
  const float h_5 = ocean_get_height(add_vector(p, get_vector(-d, 0.0f, -d)), iterations);
  const float h_6 = ocean_get_height(add_vector(p, get_vector(0.0f, 0.0f, -d)), iterations);
  const float h_7 = ocean_get_height(add_vector(p, get_vector(d, 0.0f, -d)), iterations);

  vec3 normal;
  normal.x = ((h_5 + 2.0f * h_3 + h_0) - (h_7 + 2.0f * h_4 + h_2)) * (1.0f / 8.0f);
  normal.y = d;
  normal.z = ((h_5 + 2.0f * h_6 + h_7) - (h_0 + 2.0f * h_1 + h_2)) * (1.0f / 8.0f);

  return normalize_vector(normal);
}

__device__ vec3 ocean_get_normal_fast(const vec3 p, const uint32_t iterations = OCEAN_ITERATIONS_NORMAL) {
  const float d = (OCEAN_LIPSCHITZ + get_length(p)) * eps;

  const float h_0 = ocean_get_height(add_vector(p, get_vector(0.0f, 0.0f, d)), iterations);
  const float h_1 = ocean_get_height(add_vector(p, get_vector(-d, 0.0f, 0.0f)), iterations);
  const float h_2 = ocean_get_height(add_vector(p, get_vector(d, 0.0f, 0.0f)), iterations);
  const float h_3 = ocean_get_height(add_vector(p, get_vector(0.0f, 0.0f, -d)), iterations);

  vec3 normal;
  normal.x = (h_1 - h_2) * (1.0f / 4.0f);
  normal.y = d;
  normal.z = (h_3 - h_0) * (1.0f / 4.0f);

  return normalize_vector(normal);
}

__device__ float ocean_far_distance(const vec3 origin, const vec3 ray) {
  if (!sph_ray_hit_p0(ray, world_to_sky_transform(origin), world_to_sky_scale(OCEAN_MAX_HEIGHT) + SKY_WORLD_REFERENCE_HEIGHT)) {
    return FLT_MAX;
  }

  if (fabsf(ray.y) < eps) {
    return FLT_MAX;
  }

  const float d1 = OCEAN_MIN_HEIGHT - origin.y;
  const float d2 = OCEAN_MAX_HEIGHT - origin.y;

  const float inv_ray = 1.0f / ray.y;

  const float s1 = d1 * inv_ray;
  const float s2 = d2 * inv_ray;

  const float s = fmaxf(s1, s2);

  return (s >= eps) ? s : FLT_MAX;
}

__device__ float ocean_short_distance(const vec3 origin, const vec3 ray) {
  if (!sph_ray_hit_p0(ray, world_to_sky_transform(origin), world_to_sky_scale(OCEAN_MAX_HEIGHT) + SKY_WORLD_REFERENCE_HEIGHT)) {
    return FLT_MAX;
  }

  if (fabsf(ray.y) < eps) {
    return (origin.y >= OCEAN_MIN_HEIGHT && origin.y <= OCEAN_MAX_HEIGHT) ? 0.0f : FLT_MAX;
  }

  const float d1 = OCEAN_MIN_HEIGHT - origin.y;
  const float d2 = OCEAN_MAX_HEIGHT - origin.y;

  const float inv_ray = 1.0f / ray.y;

  const float s1 = d1 * inv_ray;
  const float s2 = d2 * inv_ray;

  if (s1 < 0.0f && s2 < 0.0f)
    return FLT_MAX;

  return (s1 * s2 < 0.0f) ? fmaxf(s1, s2) : fminf(s1, s2);
}

__device__ float ocean_intersection_solver(const vec3 origin, const vec3 ray, const float start, const float limit) {
  if (start >= limit)
    return FLT_MAX;

  const float target_residual = 1e-4f;

  float min = start;
  float max = limit;

  float residual_at_max = FLT_MAX;
  float residual_at_min;

  const int32_t step_count = remap(device.ocean.amplitude * device.ocean.amplitude, 0.0f, 1.0f, 4.0f, 16.0f);

  float t                       = start;
  float last_residual           = 0.0f;
  float slope_confidence_factor = fminf(8.0f / OCEAN_LIPSCHITZ, (limit - start) * (1.0f / step_count));

  for (int i = 0; i < step_count; i++) {
    const vec3 p = add_vector(origin, scale_vector(ray, t));

    const float residual_at_t = ocean_get_relative_height(p, OCEAN_ITERATIONS_INTERSECTION);

    if (last_residual * residual_at_t < 0.0f) {
      max             = t;
      residual_at_max = residual_at_t;
      break;
    }

    last_residual = residual_at_t;

    min             = t;
    residual_at_min = residual_at_t;

    t += fabsf(residual_at_t) * slope_confidence_factor;
  }

  if (residual_at_max == FLT_MAX) {
    residual_at_max = ocean_get_relative_height(add_vector(origin, scale_vector(ray, limit)), OCEAN_ITERATIONS_INTERSECTION);
  }

  for (int i = 0; i < step_count; i++) {
    const float step = residual_at_min / (residual_at_min - residual_at_max);
    const float mid  = lerp(min, max, fminf(0.95f, fmaxf(0.05f, step)));
    const vec3 p     = add_vector(origin, scale_vector(ray, mid));

    const float residual_at_mid = ocean_get_relative_height(p, OCEAN_ITERATIONS_INTERSECTION);

    if (fabsf(residual_at_mid) < target_residual) {
      return (mid >= start) ? mid : FLT_MAX;
    }

    if (residual_at_mid * residual_at_min < 0.0f) {
      max             = mid;
      residual_at_max = residual_at_mid;
    }
    else {
      min             = mid;
      residual_at_min = residual_at_mid;
    }
  }

  if (residual_at_max * residual_at_min < 0.0f) {
    return 0.5f * (min + max);
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

  if (device.ocean.amplitude == 0.0f) {
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
    case LUMINARY_JERLOV_WATER_TYPE_I:
      return get_color(0.001f, 0.002f, 0.004f);
    case LUMINARY_JERLOV_WATER_TYPE_IA:
      return get_color(0.002f, 0.004f, 0.007f);
    case LUMINARY_JERLOV_WATER_TYPE_IB:
      return get_color(0.045f, 0.054f, 0.07f);
    case LUMINARY_JERLOV_WATER_TYPE_II:
      return get_color(0.27f, 0.365f, 0.516f);
    case LUMINARY_JERLOV_WATER_TYPE_III:
      return get_color(0.737f, 0.998f, 1.413f);
    case LUMINARY_JERLOV_WATER_TYPE_1C:
      return get_color(0.274f, 0.372f, 0.526f);
    case LUMINARY_JERLOV_WATER_TYPE_3C:
      return get_color(0.904f, 1.071f, 1.532f);
    case LUMINARY_JERLOV_WATER_TYPE_5C:
      return get_color(3.589f, 1.382f, 1.857f);
    case LUMINARY_JERLOV_WATER_TYPE_7C:
      return get_color(1.772f, 2.394f, 3.376f);
    case LUMINARY_JERLOV_WATER_TYPE_9C:
      return get_color(2.347f, 3.18f, 4.496f);
  }

  return get_color(0.0f, 0.0f, 0.0f);
}

__device__ RGBF ocean_jerlov_absorption_coefficient(const JerlovWaterType type) {
  switch (type) {
    case LUMINARY_JERLOV_WATER_TYPE_I:
      return get_color(0.309f, 0.053f, 0.009f);
    case LUMINARY_JERLOV_WATER_TYPE_IA:
      return get_color(0.309f, 0.054f, 0.014f);
    case LUMINARY_JERLOV_WATER_TYPE_IB:
      return get_color(0.309f, 0.054f, 0.015f);
    case LUMINARY_JERLOV_WATER_TYPE_II:
      return get_color(0.31f, 0.054f, 0.016f);
    case LUMINARY_JERLOV_WATER_TYPE_III:
      return get_color(0.31f, 0.056f, 0.031f);
    case LUMINARY_JERLOV_WATER_TYPE_1C:
      return get_color(0.316f, 0.067f, 0.105f);
    case LUMINARY_JERLOV_WATER_TYPE_3C:
      return get_color(0.508f, 0.052f, 0.161f);
    case LUMINARY_JERLOV_WATER_TYPE_5C:
      return get_color(4.638f, 0.222f, 0.216f);
    case LUMINARY_JERLOV_WATER_TYPE_7C:
      return get_color(0.351f, 0.188f, 0.574f);
    case LUMINARY_JERLOV_WATER_TYPE_9C:
      return get_color(0.398f, 0.349f, 0.995f);
  }

  return get_color(0.0f, 0.0f, 0.0f);
}

__device__ float ocean_molecular_weight(const JerlovWaterType type) {
  switch (type) {
    case LUMINARY_JERLOV_WATER_TYPE_I:
      return 0.93f;
    case LUMINARY_JERLOV_WATER_TYPE_IA:
      return 0.44f;
    case LUMINARY_JERLOV_WATER_TYPE_IB:
      return 0.06f;
    case LUMINARY_JERLOV_WATER_TYPE_II:
      return 0.007f;
    case LUMINARY_JERLOV_WATER_TYPE_III:
      return 0.003f;
    case LUMINARY_JERLOV_WATER_TYPE_1C:
      return 0.005f;
    case LUMINARY_JERLOV_WATER_TYPE_3C:
      return 0.003f;
    case LUMINARY_JERLOV_WATER_TYPE_5C:
      return 0.001f;
    case LUMINARY_JERLOV_WATER_TYPE_7C:
      return 0.0f;
    case LUMINARY_JERLOV_WATER_TYPE_9C:
      return 0.0f;
  }

  return 0.0f;
}

// Henyey Greenstein importance sampling for g = 0
// pbrt v3 - Light Transport II: Volume Rendering - Sampling Volume Scattering
__device__ float ocean_molecular_phase_sampling_cosine(const float r) {
  return 2.0f * r - 1.0f;
}

// Henyey Greenstein importance sampling for g != 0
// pbrt v3 - Light Transport II: Volume Rendering - Sampling Volume Scattering
__device__ float ocean_particle_phase_sampling_cosine(const float r) {
  const float g = 0.924f;

  float denom = (1.0f - g + 2.0f * g * r);
  if (fabsf(denom) < eps) {
    denom = copysignf(eps, denom);
  }

  const float s = (1.0f - g * g) / denom;

  return (1.0f + g * g - s * s) / (2.0f * g);
}

__device__ vec3 ocean_phase_sampling(const vec3 ray, const float2 r_dir, const float r_choice) {
  const float molecular_weight = ocean_molecular_weight((JerlovWaterType) device.ocean.water_type);

  float cos_angle;
  if (r_choice < molecular_weight) {
    cos_angle = ocean_molecular_phase_sampling_cosine(r_dir.x);
  }
  else {
    cos_angle = ocean_particle_phase_sampling_cosine(r_dir.x);
  }

  return phase_sample_basis(cos_angle, r_dir.y, ray);
}

__device__ float ocean_phase_sampling_cos_angle(const float r_dir, const float r_choice) {
  const float molecular_weight = ocean_molecular_weight((JerlovWaterType) device.ocean.water_type);

  float cos_angle;
  if (r_choice < molecular_weight) {
    cos_angle = ocean_molecular_phase_sampling_cosine(r_dir);
  }
  else {
    cos_angle = ocean_particle_phase_sampling_cosine(r_dir);
  }

  return cos_angle;
}

__device__ float ocean_molecular_phase(const float cos_angle) {
  return henyey_greenstein_phase_function(cos_angle, 0.0f);
}

__device__ float ocean_particle_phase(const float cos_angle) {
  return henyey_greenstein_phase_function(cos_angle, 0.924f);
}

__device__ float ocean_phase(const float cos_angle) {
  const float molecular_weight = ocean_molecular_weight((JerlovWaterType) device.ocean.water_type);

  const float molecular_phase = ocean_molecular_phase(cos_angle);
  const float particle_phase  = ocean_particle_phase(cos_angle);

  return molecular_phase * molecular_weight + particle_phase * (1.0f - molecular_weight);
}

/*
 * This uses the actual Fresnel equations to compute the reflection coefficient under the following assumptions:
 *  - The media are not magnetic.
 *  - The light is not polarized.
 *  - The IORs are wavelength independent.
 */
__device__ float ocean_reflection_coefficient(
  const vec3 normal, const vec3 ray, const vec3 refraction, const float index_in, const float index_out) {
  const float NdotV = -dot_product(ray, normal);
  const float NdotT = -dot_product(refraction, normal);

  const float s_pol_term1 = index_in * NdotV;
  const float s_pol_term2 = index_out * NdotT;

  const float p_pol_term1 = index_in * NdotT;
  const float p_pol_term2 = index_out * NdotV;

  float reflection_s_pol = (s_pol_term1 - s_pol_term2) / (s_pol_term1 + s_pol_term2);
  float reflection_p_pol = (p_pol_term1 - p_pol_term2) / (p_pol_term1 + p_pol_term2);

  reflection_s_pol *= reflection_s_pol;
  reflection_p_pol *= reflection_p_pol;

  return __saturatef(0.5f * (reflection_s_pol + reflection_p_pol));
}

__device__ MaterialContextGeometry ocean_get_context(const DeviceTask task, DeviceIORStack& ior_stack) {
  vec3 normal = ocean_get_normal(task.origin);

  const bool inside_water = dot_product(task.ray, normal) > 0.0f;

  if (inside_water)
    normal = scale_vector(normal, -1.0f);

  uint32_t flags = MATERIAL_FLAG_BASE_SUBSTRATE_TRANSLUCENT;

  if (inside_water) {
    flags |= MATERIAL_FLAG_REFRACTION_IS_INSIDE;
  }

  const bool refraction_is_inside       = (flags & MATERIAL_FLAG_REFRACTION_IS_INSIDE);
  const IORStackMethod ior_stack_method = (refraction_is_inside) ? IOR_STACK_METHOD_PEEK_PREVIOUS : IOR_STACK_METHOD_PEEK_CURRENT;
  const float ray_ior                   = ior_stack_interact(ior_stack, device.ocean.refractive_index, ior_stack_method);

  MaterialContextGeometry ctx;

  ctx.instance_id = HIT_TYPE_OCEAN;
  ctx.tri_id      = 0;
  ctx.albedo      = get_RGBAF(1.0f, 1.0f, 1.0f, 1.0f);
  ctx.normal      = normal;
  ctx.position    = task.origin;
  ctx.V           = scale_vector(task.ray, -1.0f);
  ctx.state       = task.state;
  ctx.flags       = flags;
  ctx.ior_in      = (flags & MATERIAL_FLAG_REFRACTION_IS_INSIDE) ? device.ocean.refractive_index : ray_ior;
  ctx.ior_out     = (flags & MATERIAL_FLAG_REFRACTION_IS_INSIDE) ? ray_ior : device.ocean.refractive_index;

  material_set_float<MATERIAL_GEOMETRY_PARAM_ROUGHNESS>(ctx, BSDF_ROUGHNESS_CLAMP);
  material_set_color<MATERIAL_GEOMETRY_PARAM_EMISSION>(ctx, splat_color(0.0f));

  return ctx;
}

#endif /* CU_OCEAN_UTILS_H */
