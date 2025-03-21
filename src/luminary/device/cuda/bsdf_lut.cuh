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

  GBufferData data = gbuffer_data_default();
  data.flags       = G_BUFFER_FLAG_BASE_SUBSTRATE_OPAQUE | G_BUFFER_FLAG_METALLIC;
  data.normal      = get_vector(0.0f, 0.0f, 1.0f);
  data.roughness   = roughness;
  data.V           = normalize_vector(get_vector(0.0f, sqrtf(1.0f - NdotV * NdotV), NdotV));

  float sum = 0.0f;

  for (uint32_t i = 0; i < BSDF_ENERGY_LUT_ITERATIONS; i++) {
    const vec3 H          = bsdf_microfacet_sample(data, make_ushort2(0, 0), QUASI_RANDOM_TARGET_BSDF_MICROFACET, i, 0);
    const vec3 reflection = reflect_vector(data.V, H);

    const float NdotL = reflection.z;

    if (NdotL > 0.0f)
      sum += bsdf_microfacet_evaluate_sampled_microfacet(data, NdotL, NdotV);
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

  GBufferData data;
  data.flags     = G_BUFFER_FLAG_BASE_SUBSTRATE_OPAQUE | G_BUFFER_FLAG_METALLIC;
  data.normal    = get_vector(0.0f, 0.0f, 1.0f);
  data.roughness = roughness;
  data.V         = normalize_vector(get_vector(0.0f, sqrtf(1.0f - NdotV * NdotV), NdotV));

  const RGBF f0 = get_color(0.04f, 0.04f, 0.04f);

  float sum = 0.0f;

  for (uint32_t i = 0; i < BSDF_ENERGY_LUT_ITERATIONS; i++) {
    const vec3 H          = bsdf_microfacet_sample(data, make_ushort2(0, 0), QUASI_RANDOM_TARGET_BSDF_MICROFACET, i, 0);
    const vec3 reflection = reflect_vector(data.V, H);

    const float NdotL = reflection.z;

    if (NdotL > 0.0f) {
      const float HdotV  = fabsf(dot_product(H, data.V));
      const RGBF fresnel = bsdf_fresnel_schlick(f0, bsdf_shadowed_F90(f0), HdotV);
      sum += bsdf_microfacet_evaluate_sampled_microfacet(data, NdotL, NdotV) * luminance(fresnel);
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

  GBufferData data;
  data.flags     = G_BUFFER_FLAG_BASE_SUBSTRATE_TRANSLUCENT;
  data.normal    = get_vector(0.0f, 0.0f, 1.0f);
  data.roughness = roughness;
  data.V         = normalize_vector(get_vector(0.0f, sqrtf(1.0f - NdotV * NdotV), NdotV));
  data.ior_in    = 1.0f;
  data.ior_out   = ior;

  float sum = 0.0f;
  bool total_reflection;

  for (uint32_t i = 0; i < BSDF_ENERGY_LUT_ITERATIONS; i++) {
    vec3 H          = bsdf_microfacet_sample(data, make_ushort2(0, 0), QUASI_RANDOM_TARGET_BSDF_MICROFACET, i, 0);
    vec3 reflection = reflect_vector(data.V, H);
    vec3 refraction = refract_vector(data.V, H, data.ior_in / data.ior_out, total_reflection);
    float fresnel   = (total_reflection) ? 1.0f : bsdf_fresnel(H, data.V, refraction, data.ior_in, data.ior_out);
    float HdotV     = fabsf(dot_product(H, data.V));

    const float NdotL = reflection.z;

    if (NdotL > 0.0f) {
      sum += bsdf_microfacet_evaluate_sampled_microfacet(data, NdotL, NdotV) * fresnel;
    }

    H          = bsdf_microfacet_refraction_sample(data, make_ushort2(0, 0), QUASI_RANDOM_TARGET_BSDF_REFRACTION, i, 0);
    reflection = reflect_vector(data.V, H);
    refraction = refract_vector(data.V, H, data.ior_in / data.ior_out, total_reflection);
    fresnel    = (total_reflection) ? 1.0f : bsdf_fresnel(H, data.V, refraction, data.ior_in, data.ior_out);
    HdotV      = fabsf(dot_product(H, data.V));

    const float NdotR = -refraction.z;

    if (NdotR > 0.0f) {
      const float HdotR = fabsf(dot_product(H, refraction));
      const float NdotH = fabsf(H.z);
      sum += bsdf_microfacet_refraction_evaluate_sampled_microfacet(data, HdotR, HdotV, NdotH, NdotR, NdotV, data.ior_in / data.ior_out)
             * (1.0f - fresnel);
    }
  }

  sum /= BSDF_ENERGY_LUT_ITERATIONS;

  // Ceil because underestimating energy causes excessive energy.
  args.dst[id] = 1 + (uint16_t) (ceilf(__saturatef(sum) * 0xFFFE));

  data.ior_in  = ior;
  data.ior_out = 1.0f;

  sum = 0.0f;

  for (uint32_t i = 0; i < BSDF_ENERGY_LUT_ITERATIONS; i++) {
    vec3 H          = bsdf_microfacet_sample(data, make_ushort2(0, 0), QUASI_RANDOM_TARGET_BSDF_MICROFACET, i, 0);
    vec3 reflection = reflect_vector(data.V, H);
    vec3 refraction = refract_vector(data.V, H, data.ior_in / data.ior_out, total_reflection);
    float fresnel   = (total_reflection) ? 1.0f : bsdf_fresnel(H, data.V, refraction, data.ior_in, data.ior_out);
    float HdotV     = fabsf(dot_product(H, data.V));

    const float NdotL = reflection.z;

    if (NdotL > 0.0f) {
      sum += bsdf_microfacet_evaluate_sampled_microfacet(data, NdotL, NdotV) * fresnel;
    }

    H          = bsdf_microfacet_refraction_sample(data, make_ushort2(0, 0), QUASI_RANDOM_TARGET_BSDF_REFRACTION, i, 0);
    reflection = reflect_vector(data.V, H);
    refraction = refract_vector(data.V, H, data.ior_in / data.ior_out, total_reflection);
    fresnel    = (total_reflection) ? 0.0f : bsdf_fresnel(H, data.V, refraction, data.ior_in, data.ior_out);
    HdotV      = fabsf(dot_product(H, data.V));

    const float NdotR = -refraction.z;

    if (NdotR > 0.0f) {
      const float HdotR = fabsf(dot_product(H, refraction));
      const float NdotH = fabsf(H.z);
      sum += bsdf_microfacet_refraction_evaluate_sampled_microfacet(data, HdotR, HdotV, NdotH, NdotR, NdotV, data.ior_in / data.ior_out)
             * (1.0f - fresnel);
    }
  }

  sum /= BSDF_ENERGY_LUT_ITERATIONS;

  // Ceil because underestimating energy causes excessive energy.
  args.dst_inv[id] = 1 + (uint16_t) (ceilf(__saturatef(sum) * 0xFFFE));
}

#endif

#endif /* CU_BSDF_LUT_H */
