#ifndef CU_BSDF_LUT_H
#define CU_BSDF_LUT_H

#include "bsdf_utils.cuh"
#include "memory.cuh"
#include "random.cuh"
#include "utils.cuh"

#define BSDF_LOAD_PRECOMPUTED_LUT 0

#if BSDF_LOAD_PRECOMPUTED_LUT

// TODO: I don't know, back in ancient times I added this section for using precomputed LUT, little did I know that I may never actually
// implement it :)

#else

#define BSDF_ENERGY_LUT_ITERATIONS (0x10000)

LUMINARY_KERNEL void bsdf_generate_ss_lut(KernelArgsBSDFGenerateSSLUT args) {
  const uint32_t id = THREAD_ID;

  if (id >= BSDF_LUT_SIZE * BSDF_LUT_SIZE)
    return;

  const uint32_t y = id / BSDF_LUT_SIZE;
  const uint32_t x = id - y * BSDF_LUT_SIZE;

  const float NdotV     = fmaxf(32.0f * eps, x * (1.0f / (BSDF_LUT_SIZE - 1)));
  const float roughness = y * (1.0f / (BSDF_LUT_SIZE - 1));

  MaterialContextGeometry ctx = material_get_default_context();
  ctx.params.flags            = MATERIAL_FLAG_BASE_SUBSTRATE_OPAQUE | MATERIAL_FLAG_METALLIC;
  ctx.normal                  = get_vector(0.0f, 0.0f, 1.0f);
  ctx.V                       = normalize_vector(get_vector(0.0f, sqrtf(1.0f - NdotV * NdotV), NdotV));

  material_set_float<MATERIAL_GEOMETRY_PARAM_ROUGHNESS>(ctx.params, roughness);

  float sum = 0.0f;

  for (uint32_t i = 0; i < BSDF_ENERGY_LUT_ITERATIONS; i++) {
    const PathID path_id = path_id_get(0, 0, i);

    const vec3 H          = bsdf_microfacet_sample(ctx.V, roughness, path_id, RANDOM_TARGET_BSDF_REFLECTION);
    const vec3 reflection = reflect_vector(ctx.V, H);

    const float NdotL = reflection.z;

    if (NdotL > 0.0f)
      sum += bsdf_microfacet_evaluate_sampled_microfacet(ctx.V, roughness, NdotL, NdotV);
  }

  sum /= BSDF_ENERGY_LUT_ITERATIONS;

  // Ceil because underestimating energy causes excessive energy.
  args.dst[id] = 1 + (uint16_t) (ceilf(__saturatef(sum) * 0xFFFE));
}

LUMINARY_KERNEL void bsdf_generate_glossy_lut(KernelArgsBSDFGenerateGlossyLUT args) {
  const uint32_t id = THREAD_ID;

  if (id >= BSDF_LUT_SIZE * BSDF_LUT_SIZE)
    return;

  const uint32_t y = id / BSDF_LUT_SIZE;
  const uint32_t x = id - y * BSDF_LUT_SIZE;

  const float NdotV     = fmaxf(32.0f * eps, x * (1.0f / (BSDF_LUT_SIZE - 1)));
  const float roughness = y * (1.0f / (BSDF_LUT_SIZE - 1));

  MaterialContextGeometry ctx = material_get_default_context();
  ctx.params.flags            = MATERIAL_FLAG_BASE_SUBSTRATE_OPAQUE | MATERIAL_FLAG_METALLIC;
  ctx.normal                  = get_vector(0.0f, 0.0f, 1.0f);
  ctx.V                       = normalize_vector(get_vector(0.0f, sqrtf(1.0f - NdotV * NdotV), NdotV));

  material_set_float<MATERIAL_GEOMETRY_PARAM_ROUGHNESS>(ctx.params, roughness);

  const RGBF f0 = get_color(0.04f, 0.04f, 0.04f);

  float sum = 0.0f;

  for (uint32_t i = 0; i < BSDF_ENERGY_LUT_ITERATIONS; i++) {
    const PathID path_id = path_id_get(0, 0, i);

    const vec3 H          = bsdf_microfacet_sample(ctx.V, roughness, path_id, RANDOM_TARGET_BSDF_REFLECTION);
    const vec3 reflection = reflect_vector(ctx.V, H);

    const float NdotL = reflection.z;

    if (NdotL > 0.0f) {
      const float HdotV  = fabsf(dot_product(H, ctx.V));
      const RGBF fresnel = bsdf_fresnel_schlick(f0, bsdf_shadowed_F90(f0), HdotV);
      sum += bsdf_microfacet_evaluate_sampled_microfacet(ctx.V, roughness, NdotL, NdotV) * color_luminance(fresnel);
    }
  }

  sum /= BSDF_ENERGY_LUT_ITERATIONS;

  const float single_scattering_term = args.src_energy_ss[id] * (1.0f / 0xFFFF);

  sum /= single_scattering_term;

  // Ceil because underestimating energy causes excessive energy.
  args.dst[id] = 1 + (uint16_t) (ceilf(__saturatef(sum) * 0xFFFE));
}

LUMINARY_KERNEL void bsdf_generate_dielectric_lut(KernelArgsBSDFGenerateDielectricLUT args) {
  const uint32_t id = THREAD_ID;

  if (id >= BSDF_LUT_SIZE * BSDF_LUT_SIZE * BSDF_LUT_SIZE)
    return;

  const uint32_t z = id / (BSDF_LUT_SIZE * BSDF_LUT_SIZE);
  const uint32_t y = (id - z * (BSDF_LUT_SIZE * BSDF_LUT_SIZE)) / BSDF_LUT_SIZE;
  const uint32_t x = id - y * BSDF_LUT_SIZE - z * BSDF_LUT_SIZE * BSDF_LUT_SIZE;

  const float NdotV     = fmaxf(32.0f * eps, x * (1.0f / (BSDF_LUT_SIZE - 1)));
  const float roughness = y * (1.0f / (BSDF_LUT_SIZE - 1));
  const float ior       = 1.0f + z * (1.0f / (BSDF_LUT_SIZE - 1)) * 2.0f;

  MaterialContextGeometry ctx = material_get_default_context();
  ctx.params.flags            = MATERIAL_FLAG_BASE_SUBSTRATE_TRANSLUCENT;
  ctx.normal                  = get_vector(0.0f, 0.0f, 1.0f);
  ctx.V                       = normalize_vector(get_vector(0.0f, sqrtf(1.0f - NdotV * NdotV), NdotV));

  material_set_float<MATERIAL_GEOMETRY_PARAM_ROUGHNESS>(ctx.params, roughness);
  material_set_float<MATERIAL_GEOMETRY_PARAM_IOR>(ctx.params, 1.0f / ior);

  float sum = 0.0f;
  bool total_reflection;

  for (uint32_t i = 0; i < BSDF_ENERGY_LUT_ITERATIONS; i++) {
    const PathID path_id = path_id_get(0, 0, i);

    vec3 H          = bsdf_microfacet_sample(ctx.V, roughness, path_id, RANDOM_TARGET_BSDF_REFLECTION);
    vec3 reflection = reflect_vector(ctx.V, H);
    vec3 refraction = refract_vector(ctx.V, H, 1.0f / ior, total_reflection);
    float fresnel   = (total_reflection) ? 1.0f : bsdf_fresnel(H, ctx.V, refraction, 1.0f / ior);
    float HdotV     = fabsf(dot_product(H, ctx.V));

    const float NdotL = reflection.z;

    if (NdotL > 0.0f) {
      sum += bsdf_microfacet_evaluate_sampled_microfacet(ctx.V, roughness, NdotL, NdotV) * fresnel;
    }

    H          = bsdf_microfacet_refraction_sample(ctx.V, roughness, path_id, RANDOM_TARGET_BSDF_REFRACTION);
    reflection = reflect_vector(ctx.V, H);
    refraction = refract_vector(ctx.V, H, 1.0f / ior, total_reflection);
    fresnel    = (total_reflection) ? 1.0f : bsdf_fresnel(H, ctx.V, refraction, 1.0f / ior);
    HdotV      = fabsf(dot_product(H, ctx.V));

    const float NdotR = -refraction.z;

    if (NdotR > 0.0f) {
      const float HdotR = fabsf(dot_product(H, refraction));
      const float NdotH = fabsf(H.z);
      const float value = bsdf_microfacet_refraction_evaluate_sampled_microfacet(roughness, HdotR, HdotV, NdotH, NdotR, NdotV, 1.0f / ior);

      sum += value * (1.0f - fresnel);
    }
  }

  sum /= BSDF_ENERGY_LUT_ITERATIONS;

  // Ceil because underestimating energy causes excessive energy.
  args.dst[id] = 1 + (uint16_t) (ceilf(__saturatef(sum) * 0xFFFE));

  material_set_float<MATERIAL_GEOMETRY_PARAM_IOR>(ctx.params, ior);

  sum = 0.0f;

  for (uint32_t i = 0; i < BSDF_ENERGY_LUT_ITERATIONS; i++) {
    const PathID path_id = path_id_get(0, 0, i);

    vec3 H          = bsdf_microfacet_sample(ctx.V, roughness, path_id, RANDOM_TARGET_BSDF_REFLECTION);
    vec3 reflection = reflect_vector(ctx.V, H);
    vec3 refraction = refract_vector(ctx.V, H, ior, total_reflection);
    float fresnel   = (total_reflection) ? 1.0f : bsdf_fresnel(H, ctx.V, refraction, ior);
    float HdotV     = fabsf(dot_product(H, ctx.V));

    const float NdotL = reflection.z;

    if (NdotL > 0.0f) {
      sum += bsdf_microfacet_evaluate_sampled_microfacet(ctx.V, roughness, NdotL, NdotV) * fresnel;
    }

    H          = bsdf_microfacet_refraction_sample(ctx.V, roughness, path_id, RANDOM_TARGET_BSDF_REFRACTION);
    reflection = reflect_vector(ctx.V, H);
    refraction = refract_vector(ctx.V, H, ior, total_reflection);
    fresnel    = (total_reflection) ? 0.0f : bsdf_fresnel(H, ctx.V, refraction, ior);
    HdotV      = fabsf(dot_product(H, ctx.V));

    const float NdotR = -refraction.z;

    if (NdotR > 0.0f) {
      const float HdotR = fabsf(dot_product(H, refraction));
      const float NdotH = fabsf(H.z);
      const float value = bsdf_microfacet_refraction_evaluate_sampled_microfacet(roughness, HdotR, HdotV, NdotH, NdotR, NdotV, ior);

      sum += value * (1.0f - fresnel);
    }
  }

  sum /= BSDF_ENERGY_LUT_ITERATIONS;

  // Ceil because underestimating energy causes excessive energy.
  args.dst_inv[id] = 1 + (uint16_t) (ceilf(__saturatef(sum) * 0xFFFE));
}

#endif

#endif /* CU_BSDF_LUT_H */
