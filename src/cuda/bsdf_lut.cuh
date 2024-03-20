#ifndef CU_BSDF_LUT_H
#define CU_BSDF_LUT_H

#include "bench.h"
#include "bsdf_utils.cuh"
#include "random.cuh"
#include "raytrace.h"
#include "texture.h"
#include "utils.cuh"

#define BSDF_LUT_SIZE 32

#define BSDF_LOAD_PRECOMPUTED_LUT 0

#if BSDF_LOAD_PRECOMPUTED_LUT

extern "C" void bsdf_compute_energy_lut(RaytraceInstance* instance) {
}

#else

#define BSDF_ENERGY_LUT_ITERATIONS (0x10000)

__global__ void bsdf_lut_ss_generate(uint16_t* dst) {
  const uint32_t id = THREAD_ID;

  if (id >= BSDF_LUT_SIZE * BSDF_LUT_SIZE)
    return;

  const uint32_t y = id / BSDF_LUT_SIZE;
  const uint32_t x = id - y * BSDF_LUT_SIZE;

  const float NdotV     = fmaxf(32.0f * eps, x * (1.0f / (BSDF_LUT_SIZE - 1)));
  const float roughness = y * (1.0f / (BSDF_LUT_SIZE - 1));

  GBufferData data;
  data.normal    = get_vector(0.0f, 0.0f, 1.0f);
  data.metallic  = 1.0f;
  data.roughness = roughness;
  data.V         = normalize_vector(get_vector(0.0f, sqrtf(1.0f - NdotV * NdotV), NdotV));

  const vec3 ray = scale_vector(data.V, -1.0f);

  float sum = 0.0f;

  for (uint32_t i = 0; i < BSDF_ENERGY_LUT_ITERATIONS; i++) {
    const vec3 H          = bsdf_microfacet_sample(data, make_ushort2(0, 0), i, 0);
    const vec3 reflection = reflect_vector(scale_vector(data.V, -1.0f), H);

    const float NdotL = reflection.z;

    if (NdotL > 0.0f)
      sum += bsdf_microfacet_evaluate_sampled_microfacet(data, NdotL, NdotV);
  }

  sum /= BSDF_ENERGY_LUT_ITERATIONS;

  // Ceil because underestimating energy causes excessive energy.
  dst[id] = 1 + (uint16_t) (ceilf(__saturatef(sum) * 0xFFFE));
}

__global__ void bsdf_lut_specular_generate(uint16_t* dst, const uint16_t* src_energy_ss) {
  const uint32_t id = THREAD_ID;

  if (id >= BSDF_LUT_SIZE * BSDF_LUT_SIZE)
    return;

  const uint32_t y = id / BSDF_LUT_SIZE;
  const uint32_t x = id - y * BSDF_LUT_SIZE;

  const float NdotV     = fmaxf(32.0f * eps, x * (1.0f / (BSDF_LUT_SIZE - 1)));
  const float roughness = y * (1.0f / (BSDF_LUT_SIZE - 1));

  GBufferData data;
  data.normal    = get_vector(0.0f, 0.0f, 1.0f);
  data.metallic  = 1.0f;
  data.roughness = roughness;
  data.V         = normalize_vector(get_vector(0.0f, sqrtf(1.0f - NdotV * NdotV), NdotV));

  const vec3 ray = scale_vector(data.V, -1.0f);

  const RGBF f0 = get_color(0.04f, 0.04f, 0.04f);

  float sum = 0.0f;

  for (uint32_t i = 0; i < BSDF_ENERGY_LUT_ITERATIONS; i++) {
    const vec3 H          = bsdf_microfacet_sample(data, make_ushort2(0, 0), i, 0);
    const vec3 reflection = reflect_vector(scale_vector(data.V, -1.0f), H);

    const float NdotL = reflection.z;

    if (NdotL > 0.0f) {
      const float HdotV  = fabsf(dot_product(H, data.V));
      const RGBF fresnel = bsdf_fresnel_schlick(f0, bsdf_shadowed_F90(f0), HdotV);
      sum += bsdf_microfacet_evaluate_sampled_microfacet(data, NdotL, NdotV) * luminance(fresnel);
    }
  }

  sum /= BSDF_ENERGY_LUT_ITERATIONS;

  const float single_scattering_term = src_energy_ss[id] * (1.0f / 0xFFFF);

  sum /= single_scattering_term;

  // Ceil because underestimating energy causes excessive energy.
  dst[id] = 1 + (uint16_t) (ceilf(__saturatef(sum) * 0xFFFE));
}

__global__ void bsdf_lut_dielectric_generate(uint16_t* dst, uint16_t* dst_inv) {
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
  data.normal    = get_vector(0.0f, 0.0f, 1.0f);
  data.metallic  = 1.0f;
  data.roughness = roughness;
  data.V         = normalize_vector(get_vector(0.0f, sqrtf(1.0f - NdotV * NdotV), NdotV));
  data.ior_in    = 1.0f;
  data.ior_out   = ior;

  const vec3 ray = scale_vector(data.V, -1.0f);

  float sum = 0.0f;

  for (uint32_t i = 0; i < BSDF_ENERGY_LUT_ITERATIONS; i++) {
    const vec3 H          = bsdf_microfacet_sample(data, make_ushort2(0, 0), i, 0);
    const vec3 reflection = reflect_vector(ray, H);
    const vec3 refraction = refract_vector(ray, H, data.ior_in / data.ior_out);
    const float fresnel   = bsdf_fresnel(H, data.V, refraction, data.ior_in, data.ior_out);
    const float HdotV     = fabsf(dot_product(H, data.V));

    const float NdotL = reflection.z;

    if (NdotL > 0.0f) {
      sum += bsdf_microfacet_evaluate_sampled_microfacet(data, NdotL, NdotV) * fresnel;
    }

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
  dst[id] = 1 + (uint16_t) (ceilf(__saturatef(sum) * 0xFFFE));

  data.ior_in  = ior;
  data.ior_out = 1.0f;

  sum = 0.0f;

  for (uint32_t i = 0; i < BSDF_ENERGY_LUT_ITERATIONS; i++) {
    const vec3 H          = bsdf_microfacet_sample(data, make_ushort2(0, 0), i, 0);
    const vec3 reflection = reflect_vector(ray, H);
    const vec3 refraction = refract_vector(ray, H, data.ior_in / data.ior_out);
    const float fresnel   = bsdf_fresnel(H, data.V, refraction, data.ior_in, data.ior_out);
    const float HdotV     = fabsf(dot_product(H, data.V));

    const float NdotL = reflection.z;

    if (NdotL > 0.0f) {
      sum += bsdf_microfacet_evaluate_sampled_microfacet(data, NdotL, NdotV) * fresnel;
    }

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
  dst_inv[id] = 1 + (uint16_t) (ceilf(__saturatef(sum) * 0xFFFE));
}

extern "C" void bsdf_compute_energy_lut(RaytraceInstance* instance) {
  bench_tic((const char*) "BSDF Energy LUT Computation");

  TextureRGBA luts[4];
  texture_create(&luts[BSDF_LUT_SS], BSDF_LUT_SIZE, BSDF_LUT_SIZE, 1, BSDF_LUT_SIZE, (void*) 0, TexDataUINT16, 1, TexStorageGPU);
  luts[BSDF_LUT_SS].wrap_mode_S = TexModeClamp;
  luts[BSDF_LUT_SS].wrap_mode_T = TexModeClamp;

  texture_create(&luts[BSDF_LUT_SPECULAR], BSDF_LUT_SIZE, BSDF_LUT_SIZE, 1, BSDF_LUT_SIZE, (void*) 0, TexDataUINT16, 1, TexStorageGPU);
  luts[BSDF_LUT_SPECULAR].wrap_mode_S = TexModeClamp;
  luts[BSDF_LUT_SPECULAR].wrap_mode_T = TexModeClamp;

  texture_create(
    &luts[BSDF_LUT_DIELEC], BSDF_LUT_SIZE, BSDF_LUT_SIZE, BSDF_LUT_SIZE, BSDF_LUT_SIZE, (void*) 0, TexDataUINT16, 1, TexStorageGPU);
  luts[BSDF_LUT_DIELEC].wrap_mode_S = TexModeClamp;
  luts[BSDF_LUT_DIELEC].wrap_mode_T = TexModeClamp;

  texture_create(
    &luts[BSDF_LUT_DIELEC_INV], BSDF_LUT_SIZE, BSDF_LUT_SIZE, BSDF_LUT_SIZE, BSDF_LUT_SIZE, (void*) 0, TexDataUINT16, 1, TexStorageGPU);
  luts[BSDF_LUT_DIELEC_INV].wrap_mode_S = TexModeClamp;
  luts[BSDF_LUT_DIELEC_INV].wrap_mode_T = TexModeClamp;

  size_t lut_sizes[4];
  lut_sizes[BSDF_LUT_SS]       = luts[BSDF_LUT_SS].height * luts[BSDF_LUT_SS].pitch * sizeof(uint16_t);
  lut_sizes[BSDF_LUT_SPECULAR] = luts[BSDF_LUT_SPECULAR].height * luts[BSDF_LUT_SPECULAR].pitch * sizeof(uint16_t);
  lut_sizes[BSDF_LUT_DIELEC] = luts[BSDF_LUT_DIELEC].depth * luts[BSDF_LUT_DIELEC].height * luts[BSDF_LUT_DIELEC].pitch * sizeof(uint16_t);
  lut_sizes[BSDF_LUT_DIELEC_INV] =
    luts[BSDF_LUT_DIELEC_INV].depth * luts[BSDF_LUT_DIELEC_INV].height * luts[BSDF_LUT_DIELEC_INV].pitch * sizeof(uint16_t);

  device_malloc((void**) &luts[BSDF_LUT_SS].data, lut_sizes[BSDF_LUT_SS]);
  device_malloc((void**) &luts[BSDF_LUT_SPECULAR].data, lut_sizes[BSDF_LUT_SPECULAR]);
  device_malloc((void**) &luts[BSDF_LUT_DIELEC].data, lut_sizes[BSDF_LUT_DIELEC]);
  device_malloc((void**) &luts[BSDF_LUT_DIELEC_INV].data, lut_sizes[BSDF_LUT_DIELEC_INV]);

  bsdf_lut_ss_generate<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>((uint16_t*) luts[BSDF_LUT_SS].data);
  bsdf_lut_specular_generate<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(
    (uint16_t*) luts[BSDF_LUT_SPECULAR].data, (uint16_t*) luts[BSDF_LUT_SS].data);
  bsdf_lut_dielectric_generate<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(
    (uint16_t*) luts[BSDF_LUT_DIELEC].data, (uint16_t*) luts[BSDF_LUT_DIELEC_INV].data);

  texture_create_atlas(&instance->bsdf_energy_lut, luts, 4);

  device_free(luts[BSDF_LUT_SS].data, lut_sizes[BSDF_LUT_SS]);
  device_free(luts[BSDF_LUT_SPECULAR].data, lut_sizes[BSDF_LUT_SPECULAR]);
  device_free(luts[BSDF_LUT_DIELEC].data, lut_sizes[BSDF_LUT_DIELEC]);
  device_free(luts[BSDF_LUT_DIELEC_INV].data, lut_sizes[BSDF_LUT_DIELEC_INV]);

  raytrace_update_device_pointers(instance);

  bench_toc();
}

#endif

#endif /* CU_BSDF_LUT_H */
