#ifndef CU_SKY_H
#define CU_SKY_H

#include <cuda_runtime_api.h>

#include "cloud_shadow.cuh"
#include "math.cuh"
#include "raytrace.h"
#include "sky_utils.cuh"
#include "stars.h"
#include "texture.h"
#include "texture_utils.cuh"
#include "utils.cuh"

//
// In this atmosphere rendering implementation, the single scattering is computed using ray-marching. The transmittance and multiscattering
// are precomputed as in [Bru17] and [Hil20] respectively. We extended the model using the model of [Wil21] as reference. For this, we
// modified their public GUI implementation so that we could compare our model to theirs. This modification is found in
// https://github.com/MilchRatchet/pragueskymodel. Our final model found in that repository has a lot more parameters than the
// implementation that we use in Luminary. Some of these parameters were found to be superfluous and possibly harmful for an efficient
// implementation. In the following, we summarize our additions over the common real-time methods of [Bru17] and [Hil20]. We added the layer
// of water soluble aerosols which contribute to the Mie scattering. Further, we found that using a spectrum of 8 different wavelengths with
// equal gaps in the range [415,635] allows us to obtain a visual look that is similar to that of the model in [Wil21]. Using only three
// wavelengths as in [Hil20] meant that the sky would look very purple during sunsets. Other spectrums could also work well for our
// purposes but we wanted a multiple of four to optimally use the texture sampling capabilities of the hardware and we did not want to
// actually try and solve the problem of finding four optimal wavelengths, if they exist.
//

////////////////////////////////////////////////////////////////////
// Literature
////////////////////////////////////////////////////////////////////

// [Hil20]
// Sébastien Hillaire, "A Scalable and Production Ready Sky and Atmosphere Rendering Technique", Computer Graphics Forum, 2020
// https://github.com/sebh/UnrealEngineSkyAtmosphere

// [Bru17]
// Eric Bruneton, "Precomputed Atmospheric Scattering: a New Implementation", 2017
// https://ebruneton.github.io/precomputed_atmospheric_scattering/

// [Wil21]
// Alexander Wilkie, Petr Vevoda, Thomas Bashford-Rogers, Lukas Hosek, Tomas Iser, Monika Kolarova, Tobias Rittig and Jaroslav Krivanek,
// "A Fitted Radiance and Attenuation Model for Realistic Atmospheres", Association for Computing Machinery, 40 (4), pp. 1-14, 2021
// https://cgg.mff.cuni.cz/publications/skymodel-2021/

////////////////////////////////////////////////////////////////////
// Sky Utils
////////////////////////////////////////////////////////////////////

// [Wil21]
__device__ float sky_mie_density(const float height) {
  // INSO (insoluble = dust-like particles)
  const float INSO = expf(-height * (1.0f / device.scene.sky.mie_falloff));

  // WASO (water soluble = biogenic particles, organic carbon)
  float WASO = 0.0f;
  if (height < 2.0f) {
    WASO = 1.0f + 0.125f * (2.0f - height);
  }
  else if (height < 3.0f) {
    WASO = 3.0f - height;
  }
  WASO *= 60.0f / device.scene.sky.ground_visibility;

  return device.scene.sky.base_density * (INSO + WASO);
}

__device__ float sky_ozone_density(const float height) {
  if (!device.scene.sky.ozone_absorption)
    return 0.0f;

  const float min_val = (height > 25.0f) ? 0.0f : 0.1f;
  return device.scene.sky.base_density * fmaxf(min_val, 1.0f - fabsf(height - 25.0f) / device.scene.sky.ozone_layer_thickness);
}

/*
 * Computes the start and length of a ray path through atmosphere.
 * @param origin Start point of ray in sky space.
 * @param ray Direction of ray
 * @result 2 floats, first value is the start, second value is the length of the path.
 */
__device__ float2 sky_compute_path(const vec3 origin, const vec3 ray, const float min_height, const float max_height) {
  const float height = get_length(origin);

  if (height <= min_height)
    return make_float2(0.0f, -FLT_MAX);

  float distance;
  float start = 0.0f;
  if (height > max_height) {
    const float earth_dist = sph_ray_int_p0(ray, origin, min_height);
    const float atmo_dist  = sph_ray_int_p0(ray, origin, max_height);
    const float atmo_dist2 = sph_ray_int_back_p0(ray, origin, max_height);

    distance = fminf(earth_dist - atmo_dist, atmo_dist2 - atmo_dist);
    start    = atmo_dist;
  }
  else {
    const float earth_dist = sph_ray_int_p0(ray, origin, min_height);
    const float atmo_dist  = sph_ray_int_p0(ray, origin, max_height);
    distance               = fminf(earth_dist, atmo_dist);
  }

  return make_float2(start, distance);
}

////////////////////////////////////////////////////////////////////
// Sky LUT function
////////////////////////////////////////////////////////////////////

// [Bru17]
__device__ Spectrum sky_compute_transmittance_optical_depth(const float r, const float mu) {
  const int steps = 2500;

  // Distance to top of atmosphere
  const float disc = r * r * (mu * mu - 1.0f) + SKY_ATMO_RADIUS * SKY_ATMO_RADIUS;
  const float dist = fmaxf(-r * mu + sqrtf(fmaxf(0.0f, disc)), 0.0f);

  const float step_size = dist / steps;

  Spectrum depth = spectrum_set1(0.0f);

  for (int i = 0; i <= steps; i++) {
    const float reach  = i * step_size;
    const float height = sqrtf(reach * reach + 2.0f * r * mu * reach + r * r) - SKY_EARTH_RADIUS;

    const float density_rayleigh = sky_rayleigh_density(height) * device.scene.sky.rayleigh_density;
    const float density_mie      = sky_mie_density(height) * device.scene.sky.mie_density;
    const float density_ozone    = sky_ozone_density(height) * device.scene.sky.ozone_density;

    const Spectrum extinction_rayleigh = spectrum_scale(SKY_RAYLEIGH_EXTINCTION, density_rayleigh);
    const float extinction_mie         = SKY_MIE_EXTINCTION * density_mie;
    const Spectrum extinction_ozone    = spectrum_scale(SKY_OZONE_EXTINCTION, density_ozone);

    const Spectrum extinction = spectrum_add(spectrum_add(extinction_rayleigh, spectrum_set1(extinction_mie)), extinction_ozone);

    const float w = (i == 0 || i == steps) ? 0.5f : 1.0f;

    depth = spectrum_add(depth, spectrum_scale(extinction, w * step_size));
  }

  return depth;
}

// [Bru17]
LUMINARY_KERNEL void sky_compute_transmittance_lut(float4* transmittance_tex_lower, float4* transmittance_tex_higher) {
  unsigned int id = THREAD_ID;

  const int amount = SKY_TM_TEX_WIDTH * SKY_TM_TEX_HEIGHT;

  while (id < amount) {
    const int y = id / SKY_TM_TEX_WIDTH;
    const int x = id - y * SKY_TM_TEX_WIDTH;

    float fx = ((float) x + 0.5f) / SKY_TM_TEX_WIDTH;
    float fy = ((float) y + 0.5f) / SKY_TM_TEX_HEIGHT;

    fx = sky_sub_to_unit_uv(fx, SKY_TM_TEX_WIDTH);
    fy = sky_sub_to_unit_uv(fy, SKY_TM_TEX_HEIGHT);

    const float H   = sqrtf(SKY_ATMO_RADIUS * SKY_ATMO_RADIUS - SKY_EARTH_RADIUS * SKY_EARTH_RADIUS);
    const float rho = H * fy;
    const float r   = sqrtf(rho * rho + SKY_EARTH_RADIUS * SKY_EARTH_RADIUS);

    const float d_min = SKY_ATMO_RADIUS - r;
    const float d_max = rho + H;
    const float d     = d_min + fx * (d_max - d_min);

    float mu = (d == 0.0f) ? 1.0f : (H * H - rho * rho - d * d) / (2.0f * r * d);
    mu       = fminf(1.0f, fmaxf(-1.0f, mu));

    const Spectrum optical_depth = sky_compute_transmittance_optical_depth(r, mu);
    const Spectrum transmittance = spectrum_exp(spectrum_scale(optical_depth, -1.0f));

    transmittance_tex_lower[x + y * SKY_TM_TEX_WIDTH]  = spectrum_split_low(transmittance);
    transmittance_tex_higher[x + y * SKY_TM_TEX_WIDTH] = spectrum_split_high(transmittance);

    id += blockDim.x * gridDim.x;
  }
}

struct msScatteringResult {
  Spectrum L;
  Spectrum multiScatterAs1;
} typedef msScatteringResult;

// [Hil20]
__device__ msScatteringResult sky_compute_multiscattering_integration(const vec3 origin, const vec3 ray, const vec3 sun_pos) {
  msScatteringResult result;

  result.L               = spectrum_set1(0.0f);
  result.multiScatterAs1 = spectrum_set1(0.0f);

  const float2 path = sky_compute_path(origin, ray, SKY_EARTH_RADIUS, SKY_ATMO_RADIUS);

  if (path.y == -FLT_MAX) {
    return result;
  }

  const float start    = path.x;
  const float distance = path.y;

  if (distance > 0.0f) {
    const int steps = 500;
    float reach     = start;
    float step_size;

    const float light_angle = sample_sphere_solid_angle(sun_pos, SKY_SUN_RADIUS, origin);

    Spectrum transmittance = spectrum_set1(1.0f);

    const JendersieEonParams mie_params = jendersie_eon_phase_parameters(device.scene.sky.mie_diameter);

    for (int i = 0; i < steps; i++) {
      const float newReach = start + distance * (i + 0.3f) / steps;
      step_size            = newReach - reach;
      reach                = newReach;

      const vec3 pos     = add_vector(origin, scale_vector(ray, reach));
      const float height = sky_height(pos);

      const vec3 ray_scatter     = normalize_vector(sub_vector(sun_pos, pos));
      const float cos_angle      = dot_product(ray, ray_scatter);
      const float phase_rayleigh = sky_rayleigh_phase(cos_angle);
      const float phase_mie      = sky_mie_phase(cos_angle, mie_params);

      const float zenith_cos_angle = dot_product(normalize_vector(pos), ray_scatter);

      const UV transmittance_uv       = sky_transmittance_lut_uv(height, zenith_cos_angle);
      const float4 transmittance_low  = tex2D<float4>(device.ptrs.sky_tm_luts[0].tex, transmittance_uv.u, transmittance_uv.v);
      const float4 transmittance_high = tex2D<float4>(device.ptrs.sky_tm_luts[1].tex, transmittance_uv.u, transmittance_uv.v);
      const Spectrum extinction_sun   = spectrum_merge(transmittance_low, transmittance_high);

      const float density_rayleigh = sky_rayleigh_density(height) * device.scene.sky.rayleigh_density;
      const float density_mie      = sky_mie_density(height) * device.scene.sky.mie_density;
      const float density_ozone    = sky_ozone_density(height) * device.scene.sky.ozone_density;

      const Spectrum scattering_rayleigh = spectrum_scale(SKY_RAYLEIGH_SCATTERING, density_rayleigh);
      const float scattering_mie         = SKY_MIE_SCATTERING * density_mie;

      const Spectrum extinction_rayleigh = spectrum_scale(SKY_RAYLEIGH_EXTINCTION, density_rayleigh);
      const float extinction_mie         = SKY_MIE_EXTINCTION * density_mie;
      const Spectrum extinction_ozone    = spectrum_scale(SKY_OZONE_EXTINCTION, density_ozone);

      const Spectrum scattering = spectrum_add(scattering_rayleigh, spectrum_set1(scattering_mie));
      const Spectrum extinction = spectrum_add(spectrum_add(extinction_rayleigh, spectrum_set1(extinction_mie)), extinction_ozone);
      const Spectrum phaseTimesScattering =
        spectrum_add(spectrum_scale(scattering_rayleigh, phase_rayleigh), spectrum_set1(scattering_mie * phase_mie));

      const float shadow = sph_ray_hit_p0(ray_scatter, pos, SKY_EARTH_RADIUS) ? 0.0f : 1.0f;
      const Spectrum S   = spectrum_scale(spectrum_mul(extinction_sun, phaseTimesScattering), shadow * light_angle);

      Spectrum step_transmittance = extinction;
      step_transmittance          = spectrum_scale(step_transmittance, -step_size);
      step_transmittance          = spectrum_exp(step_transmittance);

      const Spectrum ssInt = spectrum_mul(spectrum_sub(S, spectrum_mul(S, step_transmittance)), spectrum_inv(extinction));
      const Spectrum msInt = spectrum_mul(spectrum_sub(scattering, spectrum_mul(scattering, step_transmittance)), spectrum_inv(extinction));

      result.L               = spectrum_add(result.L, spectrum_mul(ssInt, transmittance));
      result.multiScatterAs1 = spectrum_add(result.multiScatterAs1, spectrum_mul(msInt, transmittance));

      transmittance = spectrum_mul(transmittance, step_transmittance);
    }
  }

  return result;
}

// [Hil20]
// This kernel does not use default Luminary launch bounds, hence it may not be marked as LUMINARY_KERNEL
__global__ void sky_compute_multiscattering_lut(float4* multiscattering_tex_lower, float4* multiscattering_tex_higher) {
  const int x = blockIdx.x;
  const int y = blockIdx.y;

  float fx = ((float) x + 0.5f) / SKY_MS_TEX_SIZE;
  float fy = ((float) y + 0.5f) / SKY_MS_TEX_SIZE;

  fx = sky_sub_to_unit_uv(fx, SKY_MS_TEX_SIZE);
  fy = sky_sub_to_unit_uv(fy, SKY_MS_TEX_SIZE);

  __shared__ Spectrum luminance_shared[SKY_MS_ITER];
  __shared__ Spectrum multiscattering_shared[SKY_MS_ITER];

  const float cos_angle = fx * 2.0f - 1.0f;
  const vec3 sun_dir    = get_vector(0.0f, cos_angle, sqrtf(__saturatef(1.0f - cos_angle * cos_angle)));
  const float height    = SKY_EARTH_RADIUS + __saturatef(fy + SKY_HEIGHT_OFFSET) * (SKY_ATMO_HEIGHT - SKY_HEIGHT_OFFSET);

  const vec3 pos     = get_vector(0.0f, height, 0.0f);
  const vec3 sun_pos = scale_vector(sun_dir, SKY_SUN_DISTANCE);

  const float sqrt_sample = (float) SKY_MS_BASE;

  const float a     = threadIdx.x / SKY_MS_BASE;
  const float b     = (threadIdx.x - ((threadIdx.x / SKY_MS_BASE) * SKY_MS_BASE));
  const float randA = a / sqrt_sample;
  const float randB = b / sqrt_sample;
  const vec3 ray    = sample_ray_sphere(2.0f * randA - 1.0f, randB);

  msScatteringResult result = sky_compute_multiscattering_integration(pos, ray, sun_pos);

  luminance_shared[threadIdx.x]       = result.L;
  multiscattering_shared[threadIdx.x] = result.multiScatterAs1;

  for (int i = SKY_MS_ITER >> 1; i > 0; i = i >> 1) {
    __syncthreads();
    if (threadIdx.x < i) {
      luminance_shared[threadIdx.x]       = spectrum_add(luminance_shared[threadIdx.x], luminance_shared[threadIdx.x + i]);
      multiscattering_shared[threadIdx.x] = spectrum_add(multiscattering_shared[threadIdx.x], multiscattering_shared[threadIdx.x + i]);
    }
  }

  if (threadIdx.x > 0)
    return;

  Spectrum luminance       = spectrum_scale(luminance_shared[0], 1.0f / (sqrt_sample * sqrt_sample));
  Spectrum multiscattering = spectrum_scale(multiscattering_shared[0], 1.0f / (sqrt_sample * sqrt_sample));

  const Spectrum multiScatteringContribution = spectrum_inv(spectrum_sub(spectrum_set1(1.0f), multiscattering));

  const Spectrum L = spectrum_scale(spectrum_mul(luminance, multiScatteringContribution), device.scene.sky.multiscattering_factor);

  multiscattering_tex_lower[x + y * SKY_MS_TEX_SIZE]  = spectrum_split_low(L);
  multiscattering_tex_higher[x + y * SKY_MS_TEX_SIZE] = spectrum_split_high(L);
}

extern "C" void device_sky_generate_LUTs(RaytraceInstance* instance) {
  bench_tic((const char*) "Sky LUT Computation");

  if (instance->scene.sky.lut_initialized) {
    texture_free_atlas(instance->sky_tm_luts, 2);
    texture_free_atlas(instance->sky_ms_luts, 2);
  }

  instance->scene.sky.base_density           = instance->atmo_settings.base_density;
  instance->scene.sky.ground_visibility      = instance->atmo_settings.ground_visibility;
  instance->scene.sky.mie_density            = instance->atmo_settings.mie_density;
  instance->scene.sky.mie_falloff            = instance->atmo_settings.mie_falloff;
  instance->scene.sky.mie_diameter           = instance->atmo_settings.mie_diameter;
  instance->scene.sky.ozone_absorption       = instance->atmo_settings.ozone_absorption;
  instance->scene.sky.ozone_density          = instance->atmo_settings.ozone_density;
  instance->scene.sky.ozone_layer_thickness  = instance->atmo_settings.ozone_layer_thickness;
  instance->scene.sky.rayleigh_density       = instance->atmo_settings.rayleigh_density;
  instance->scene.sky.rayleigh_falloff       = instance->atmo_settings.rayleigh_falloff;
  instance->scene.sky.multiscattering_factor = instance->atmo_settings.multiscattering_factor;

  raytrace_update_device_scene(instance);

  TextureRGBA luts_tm_tex[2];
  texture_create(luts_tm_tex + 0, SKY_TM_TEX_WIDTH, SKY_TM_TEX_HEIGHT, 1, SKY_TM_TEX_WIDTH, (void*) 0, TexDataFP32, 4, TexStorageGPU);
  texture_create(luts_tm_tex + 1, SKY_TM_TEX_WIDTH, SKY_TM_TEX_HEIGHT, 1, SKY_TM_TEX_WIDTH, (void*) 0, TexDataFP32, 4, TexStorageGPU);
  luts_tm_tex[0].wrap_mode_S = TexModeClamp;
  luts_tm_tex[0].wrap_mode_T = TexModeClamp;
  luts_tm_tex[1].wrap_mode_S = TexModeClamp;
  luts_tm_tex[1].wrap_mode_T = TexModeClamp;

  device_malloc((void**) &luts_tm_tex[0].data, luts_tm_tex[0].height * luts_tm_tex[0].pitch * 4 * sizeof(float));
  device_malloc((void**) &luts_tm_tex[1].data, luts_tm_tex[1].height * luts_tm_tex[1].pitch * 4 * sizeof(float));

  sky_compute_transmittance_lut<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>((float4*) luts_tm_tex[0].data, (float4*) luts_tm_tex[1].data);

  gpuErrchk(cudaDeviceSynchronize());

  texture_create_atlas(&instance->sky_tm_luts, luts_tm_tex, 2);

  device_free(luts_tm_tex[0].data, luts_tm_tex[0].height * luts_tm_tex[0].pitch * 4 * sizeof(float));
  device_free(luts_tm_tex[1].data, luts_tm_tex[1].height * luts_tm_tex[1].pitch * 4 * sizeof(float));

  raytrace_update_device_pointers(instance);

  TextureRGBA luts_ms_tex[2];
  texture_create(luts_ms_tex + 0, SKY_MS_TEX_SIZE, SKY_MS_TEX_SIZE, 1, SKY_MS_TEX_SIZE, (void*) 0, TexDataFP32, 4, TexStorageGPU);
  texture_create(luts_ms_tex + 1, SKY_MS_TEX_SIZE, SKY_MS_TEX_SIZE, 1, SKY_MS_TEX_SIZE, (void*) 0, TexDataFP32, 4, TexStorageGPU);
  luts_ms_tex[0].wrap_mode_S = TexModeClamp;
  luts_ms_tex[0].wrap_mode_T = TexModeClamp;
  luts_ms_tex[1].wrap_mode_S = TexModeClamp;
  luts_ms_tex[1].wrap_mode_T = TexModeClamp;

  device_malloc((void**) &luts_ms_tex[0].data, luts_ms_tex[0].height * luts_ms_tex[0].pitch * 4 * sizeof(float));
  device_malloc((void**) &luts_ms_tex[1].data, luts_ms_tex[1].height * luts_ms_tex[1].pitch * 4 * sizeof(float));

  // We use the z component to signify its special intention
  dim3 threads_ms(SKY_MS_ITER, 1, 1);
  dim3 blocks_ms(SKY_MS_TEX_SIZE, SKY_MS_TEX_SIZE, 1);

  sky_compute_multiscattering_lut<<<blocks_ms, threads_ms>>>((float4*) luts_ms_tex[0].data, (float4*) luts_ms_tex[1].data);

  gpuErrchk(cudaDeviceSynchronize());

  texture_create_atlas(&instance->sky_ms_luts, luts_ms_tex, 2);

  device_free(luts_ms_tex[0].data, luts_ms_tex[0].height * luts_ms_tex[0].pitch * 4 * sizeof(float));
  device_free(luts_ms_tex[1].data, luts_ms_tex[1].height * luts_ms_tex[1].pitch * 4 * sizeof(float));

  raytrace_update_device_pointers(instance);

  instance->scene.sky.lut_initialized = 1;

  bench_toc();
}

////////////////////////////////////////////////////////////////////
// Atmosphere Integration
////////////////////////////////////////////////////////////////////

__device__ Spectrum sky_compute_atmosphere(
  Spectrum& transmittance_out, const vec3 origin, const vec3 ray, const float limit, const bool celestials, const bool cloud_shadows,
  const int steps, ushort2 pixel) {
  Spectrum result = spectrum_set1(0.0f);

  const float2 path = sky_compute_path(origin, ray, SKY_EARTH_RADIUS, SKY_ATMO_RADIUS);

  const float start    = path.x;
  const float distance = fminf(path.y, limit - start);

  Spectrum transmittance = spectrum_get_ident();

  if (distance > 0.0f) {
    float reach = start;
    float step_size;

    const float light_angle   = sample_sphere_solid_angle(device.sun_pos, SKY_SUN_RADIUS, origin);
    const float random_offset = quasirandom_sequence_1D(QUASI_RANDOM_TARGET_SKY_STEP_OFFSET, pixel);

    const JendersieEonParams mie_params = jendersie_eon_phase_parameters(device.scene.sky.mie_diameter);

    for (int i = 0; i < steps; i++) {
      const float new_reach = start + distance * (i + random_offset) / steps;
      step_size             = new_reach - reach;
      reach                 = new_reach;

      const vec3 pos     = add_vector(origin, scale_vector(ray, reach));
      const float height = sky_height(pos);

      const vec3 ray_scatter       = normalize_vector(sub_vector(device.sun_pos, pos));
      const float cos_angle        = dot_product(ray, ray_scatter);
      const float zenith_cos_angle = dot_product(normalize_vector(pos), ray_scatter);
      const float phase_rayleigh   = sky_rayleigh_phase(cos_angle);
      const float phase_mie        = sky_mie_phase(cos_angle, mie_params);

      float shadow;
      if (cloud_shadows) {
        shadow = sph_ray_hit_p0(ray_scatter, pos, SKY_EARTH_RADIUS) ? 0.0f : cloud_shadow(pos, ray_scatter);
      }
      else {
        shadow = sph_ray_hit_p0(ray_scatter, pos, SKY_EARTH_RADIUS) ? 0.0f : 1.0f;
      }

      const UV transmittance_uv       = sky_transmittance_lut_uv(height, zenith_cos_angle);
      const float4 transmittance_low  = tex2D<float4>(device.ptrs.sky_tm_luts[0].tex, transmittance_uv.u, transmittance_uv.v);
      const float4 transmittance_high = tex2D<float4>(device.ptrs.sky_tm_luts[1].tex, transmittance_uv.u, transmittance_uv.v);
      const Spectrum extinction_sun   = spectrum_merge(transmittance_low, transmittance_high);

      const float density_rayleigh = sky_rayleigh_density(height) * device.scene.sky.rayleigh_density;
      const float density_mie      = sky_mie_density(height) * device.scene.sky.mie_density;
      const float density_ozone    = sky_ozone_density(height) * device.scene.sky.ozone_density;

      const Spectrum scattering_rayleigh = spectrum_scale(SKY_RAYLEIGH_SCATTERING, density_rayleigh);
      const float scattering_mie         = SKY_MIE_SCATTERING * density_mie;

      const Spectrum extinction_rayleigh = spectrum_scale(SKY_RAYLEIGH_EXTINCTION, density_rayleigh);
      const float extinction_mie         = SKY_MIE_EXTINCTION * density_mie;
      const Spectrum extinction_ozone    = spectrum_scale(SKY_OZONE_EXTINCTION, density_ozone);

      const Spectrum scattering = spectrum_add(scattering_rayleigh, spectrum_set1(scattering_mie));
      const Spectrum extinction = spectrum_add(spectrum_add(extinction_rayleigh, spectrum_set1(extinction_mie)), extinction_ozone);
      const Spectrum phase_times_scattering =
        spectrum_add(spectrum_scale(scattering_rayleigh, phase_rayleigh), spectrum_set1(scattering_mie * phase_mie));

      const Spectrum ss_radiance = spectrum_scale(spectrum_mul(extinction_sun, phase_times_scattering), shadow * light_angle);

      const UV multiscattering_uv        = get_uv(zenith_cos_angle * 0.5f + 0.5f, height / SKY_ATMO_HEIGHT);
      const float4 multiscattering_low   = tex2D<float4>(device.ptrs.sky_ms_luts[0].tex, multiscattering_uv.u, multiscattering_uv.v);
      const float4 multiscattering_high  = tex2D<float4>(device.ptrs.sky_ms_luts[1].tex, multiscattering_uv.u, multiscattering_uv.v);
      const Spectrum multiscattering_tex = spectrum_merge(multiscattering_low, multiscattering_high);
      const Spectrum ms_radiance         = spectrum_mul(multiscattering_tex, scattering);

      const Spectrum S = spectrum_add(ss_radiance, ms_radiance);

      Spectrum step_transmittance = extinction;
      step_transmittance          = spectrum_scale(step_transmittance, -step_size);
      step_transmittance          = spectrum_exp(step_transmittance);

      const Spectrum Sint = spectrum_mul(spectrum_sub(S, spectrum_mul(S, step_transmittance)), spectrum_inv(extinction));

      result        = spectrum_add(result, spectrum_mul(Sint, transmittance));
      transmittance = spectrum_mul(transmittance, step_transmittance);
    }

    const Spectrum sun_radiance = spectrum_scale(SKY_SUN_RADIANCE, device.scene.sky.sun_strength);
    result                      = spectrum_mul(result, sun_radiance);
  }

  if (celestials) {
    const float sun_hit   = sphere_ray_intersection(ray, origin, device.sun_pos, SKY_SUN_RADIUS);
    const float earth_hit = sph_ray_int_p0(ray, origin, SKY_EARTH_RADIUS);
    const float moon_hit  = sphere_ray_intersection(ray, origin, device.moon_pos, SKY_MOON_RADIUS);

    if (earth_hit > sun_hit && moon_hit > sun_hit) {
      const Spectrum S = spectrum_mul(transmittance, spectrum_scale(SKY_SUN_RADIANCE, device.scene.sky.sun_strength));

      result = spectrum_add(result, S);
    }
    else if (earth_hit > moon_hit) {
      const vec3 moon_pos   = add_vector(origin, scale_vector(ray, moon_hit));
      const vec3 bounce_ray = normalize_vector(sub_vector(device.sun_pos, moon_pos));

      if (!sphere_ray_hit(bounce_ray, moon_pos, get_vector(0.0f, 0.0f, 0.0f), SKY_EARTH_RADIUS)) {
        vec3 normal = normalize_vector(sub_vector(moon_pos, device.moon_pos));

        const float tex_u = 0.5f + device.scene.sky.moon_tex_offset + atan2f(normal.z, normal.x) * (1.0f / (2.0f * PI));
        const float tex_v = 0.5f + asinf(normal.y) * (1.0f / PI);

        const UV uv = get_uv(tex_u, tex_v);

        const Mat3x3 tangent_space = create_basis(normal);

        const float4 normal_vals = texture_load(*device.ptrs.sky_moon_normal_tex, uv);

        vec3 map_normal = get_vector(normal_vals.x, normal_vals.y, normal_vals.z);
        map_normal      = scale_vector(map_normal, 2.0f);
        map_normal      = sub_vector(map_normal, get_vector(1.0f, 1.0f, 1.0f));

        normal = normalize_vector(transform_vec3(tangent_space, map_normal));

        const float NdotL = dot_product(normal, bounce_ray);

        if (NdotL > 0.0f) {
          const float albedo = texture_load(*device.ptrs.sky_moon_albedo_tex, uv).x;

          const float light_angle = sample_sphere_solid_angle(device.sun_pos, SKY_SUN_RADIUS, moon_pos);
          const float weight      = albedo * device.scene.sky.sun_strength * NdotL * light_angle / (2.0f * PI);

          result =
            spectrum_add(result, spectrum_mul(transmittance, spectrum_mul(SKY_MOON_SOLAR_FLUX, spectrum_scale(SKY_SUN_RADIANCE, weight))));
        }
      }
    }

    if (sun_hit == FLT_MAX && earth_hit == FLT_MAX && moon_hit == FLT_MAX) {
      const float ray_altitude = asinf(ray.y);
      const float ray_azimuth  = atan2f(-ray.z, -ray.x) + PI;

      const int x = (int) (ray_azimuth * 10.0f);
      const int y = (int) ((ray_altitude + PI * 0.5f) * 10.0f);

      const int grid = x + y * STARS_GRID_LD;

      const int a = device.scene.sky.stars_offsets[grid];
      const int b = device.scene.sky.stars_offsets[grid + 1];

      for (int i = a; i < b; i++) {
        const Star star     = device.scene.sky.stars[i];
        const vec3 star_pos = angles_to_direction(star.altitude, star.azimuth);

        if (sphere_ray_hit(ray, get_vector(0.0f, 0.0f, 0.0f), star_pos, star.radius)) {
          result = spectrum_add(result, spectrum_scale(transmittance, star.intensity * device.scene.sky.stars_intensity));
        }
      }
    }
  }

  transmittance_out = spectrum_mul(transmittance_out, transmittance);

  return result;
}

////////////////////////////////////////////////////////////////////
// Wrapper
////////////////////////////////////////////////////////////////////

__device__ RGBF sky_get_color(const vec3 origin, const vec3 ray, const float limit, const bool celestials, const int steps, ushort2 pixel) {
  Spectrum unused = spectrum_set1(0.0f);

  const Spectrum radiance = sky_compute_atmosphere(unused, origin, ray, limit, celestials, false, steps, pixel);

  return sky_compute_color_from_spectrum(radiance);
}

__device__ RGBF sky_trace_inscattering(const vec3 origin, const vec3 ray, const float limit, RGBF& record, ushort2 pixel) {
  Spectrum transmittance = spectrum_set1(1.0f);

  const float base_range = (IS_PRIMARY_RAY) ? 40.0f : 80.0f;

  const int steps = fminf(fmaxf(0.5f, limit / base_range), 2.0f) * (device.scene.sky.steps / 6)
                    + quasirandom_sequence_1D(QUASI_RANDOM_TARGET_SKY_INSCATTERING_STEP, pixel) - 0.5f;

  const Spectrum radiance = sky_compute_atmosphere(transmittance, origin, ray, limit, false, true, steps, pixel);

  const RGBF inscattering = mul_color(sky_compute_color_from_spectrum(radiance), record);

  record = mul_color(record, sky_compute_color_from_spectrum(transmittance));

  return inscattering;
}

////////////////////////////////////////////////////////////////////
// Kernel
////////////////////////////////////////////////////////////////////

LUMINARY_KERNEL void process_sky_tasks() {
  const int task_count  = device.ptrs.task_counts[THREAD_ID * TASK_ADDRESS_COUNT_STRIDE + TASK_ADDRESS_OFFSET_SKY];
  const int task_offset = device.ptrs.task_offsets[THREAD_ID * TASK_ADDRESS_OFFSET_STRIDE + TASK_ADDRESS_OFFSET_SKY];

  for (int i = 0; i < task_count; i++) {
    const ShadingTask task = load_shading_task(device.ptrs.trace_tasks + get_task_address(task_offset + i));
    const int pixel        = task.index.y * device.width + task.index.x;

    const RGBF record = load_RGBF(device.ptrs.records + pixel);

    RGBF sky;
    if (device.scene.sky.hdri_active) {
      sky = sky_hdri_sample(task.ray, 0.0f);

      if (state_peek(pixel, STATE_FLAG_CAMERA_DIRECTION)) {
        const vec3 sky_origin = world_to_sky_transform(device.scene.sky.hdri_origin);

        // HDRI does not include the sun, compute sun visibility
        const bool ray_hits_sun   = sphere_ray_hit(task.ray, sky_origin, device.sun_pos, SKY_SUN_RADIUS);
        const bool ray_hits_earth = sph_ray_hit_p0(task.ray, sky_origin, SKY_EARTH_RADIUS);

        if (ray_hits_sun && !ray_hits_earth) {
          const RGBF sun_color = sky_get_sun_color(sky_origin, task.ray);

          sky = add_color(sky, sun_color);
        }
      }
    }
    else {
      const vec3 sky_origin  = world_to_sky_transform(task.position);
      const bool include_sun = state_peek(pixel, STATE_FLAG_CAMERA_DIRECTION);

      sky = sky_get_color(sky_origin, task.ray, FLT_MAX, include_sun, device.scene.sky.steps, task.index);
    }

    sky = mul_color(sky, record);

    write_beauty_buffer(sky, pixel);
    write_albedo_buffer(sky, pixel);
    write_normal_buffer(get_vector(0.0f, 0.0f, 0.0f), pixel);
  }
}

LUMINARY_KERNEL void process_debug_sky_tasks() {
  const int task_count  = device.ptrs.task_counts[THREAD_ID * TASK_ADDRESS_COUNT_STRIDE + TASK_ADDRESS_OFFSET_SKY];
  const int task_offset = device.ptrs.task_offsets[THREAD_ID * TASK_ADDRESS_OFFSET_STRIDE + TASK_ADDRESS_OFFSET_SKY];

  for (int i = 0; i < task_count; i++) {
    const ShadingTask task = load_shading_task(device.ptrs.trace_tasks + get_task_address(task_offset + i));
    const int pixel        = task.index.y * device.width + task.index.x;

    if (device.shading_mode == SHADING_ALBEDO) {
      RGBF sky;
      if (device.scene.sky.hdri_active) {
        sky = sky_hdri_sample(task.ray, device.scene.sky.hdri_mip_bias);
      }
      else {
        sky = sky_get_color(world_to_sky_transform(task.position), task.ray, FLT_MAX, true, device.scene.sky.steps, task.index);
      }
      write_beauty_buffer(sky, pixel, true);
    }
    else if (device.shading_mode == SHADING_DEPTH) {
      const float value = __saturatef((1.0f / device.scene.camera.far_clip_distance) * 2.0f);
      write_beauty_buffer(get_color(value, value, value), pixel, true);
    }
    else if (device.shading_mode == SHADING_IDENTIFICATION) {
      write_beauty_buffer(get_color(0.0f, 0.63f, 1.0f), pixel, true);
    }
  }
}

#endif /* CU_SKY_H */
