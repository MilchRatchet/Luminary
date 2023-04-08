#ifndef CU_SKY_H
#define CU_SKY_H

#include <cuda_runtime_api.h>

#include "cloud_shadow.cuh"
#include "math.cuh"
#include "raytrace.h"
#include "sky_utils.cuh"
#include "stars.h"
#include "texture.h"
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
// Spectrum Math Functions
////////////////////////////////////////////////////////////////////

struct Spectrum {
  float v[8];
} typedef Spectrum;

// This is the spectrum that transforms to the identity color (1,1,1)
__device__ Spectrum spectrum_get_ident() {
  Spectrum result;

  result.v[0] = 8.4205e-03f;
  result.v[1] = 2.6449e-01f;
  result.v[2] = 4.0273e-01f;
  result.v[3] = 1.6624e-01f;
  result.v[4] = 2.4324e-01f;
  result.v[5] = 3.5849e-01f;
  result.v[6] = 3.6342e-01f;
  result.v[7] = 2.4177e-01f;

  return result;
}

__device__ Spectrum spectrum_set1(const float v) {
  Spectrum result;

#pragma unroll
  for (int i = 0; i < 8; i++) {
    result.v[i] = v;
  }

  return result;
}

__device__ Spectrum spectrum_set(
  const float v0, const float v1, const float v2, const float v3, const float v4, const float v5, const float v6, const float v7) {
  Spectrum result;

  result.v[0] = v0;
  result.v[1] = v1;
  result.v[2] = v2;
  result.v[3] = v3;
  result.v[4] = v4;
  result.v[5] = v5;
  result.v[6] = v6;
  result.v[7] = v7;

  return result;
}

__device__ Spectrum spectrum_add(const Spectrum a, const Spectrum b) {
  Spectrum result;

#pragma unroll
  for (int i = 0; i < 8; i++) {
    result.v[i] = a.v[i] + b.v[i];
  }

  return result;
}

__device__ Spectrum spectrum_sub(const Spectrum a, const Spectrum b) {
  Spectrum result;

#pragma unroll
  for (int i = 0; i < 8; i++) {
    result.v[i] = a.v[i] - b.v[i];
  }

  return result;
}

__device__ Spectrum spectrum_scale(const Spectrum a, const float b) {
  Spectrum result;

#pragma unroll
  for (int i = 0; i < 8; i++) {
    result.v[i] = a.v[i] * b;
  }

  return result;
}

__device__ Spectrum spectrum_mul(const Spectrum a, const Spectrum b) {
  Spectrum result;

#pragma unroll
  for (int i = 0; i < 8; i++) {
    result.v[i] = a.v[i] * b.v[i];
  }

  return result;
}

__device__ Spectrum spectrum_inv(const Spectrum a) {
  Spectrum result;

#pragma unroll
  for (int i = 0; i < 8; i++) {
    result.v[i] = 1.0f / a.v[i];
  }

  return result;
}

__device__ Spectrum spectrum_exp(const Spectrum a) {
  Spectrum result;

#pragma unroll
  for (int i = 0; i < 8; i++) {
    result.v[i] = expf(a.v[i]);
  }

  return result;
}

__device__ Spectrum spectrum_merge(const float4 low, const float4 high) {
  Spectrum result;

  result.v[0] = low.x;
  result.v[1] = low.y;
  result.v[2] = low.z;
  result.v[3] = low.w;
  result.v[4] = high.x;
  result.v[5] = high.y;
  result.v[6] = high.z;
  result.v[7] = high.w;

  return result;
}

__device__ float4 spectrum_split_low(const Spectrum a) {
  float4 result;

  result.x = a.v[0];
  result.y = a.v[1];
  result.z = a.v[2];
  result.w = a.v[3];

  return result;
}

__device__ float4 spectrum_split_high(const Spectrum a) {
  float4 result;

  result.x = a.v[4];
  result.y = a.v[5];
  result.z = a.v[6];
  result.w = a.v[7];

  return result;
}

////////////////////////////////////////////////////////////////////
// Defines
////////////////////////////////////////////////////////////////////

#define SKY_WAVELENGTHS \
  spectrum_set(4.150000e+02f, 4.464286e+02f, 4.778571e+02f, 5.092857e+02f, 5.407143e+02f, 5.721428e+02f, 6.035714e+02f, 6.350000e+02f)

#define SKY_SUN_RADIANCE \
  spectrum_set(2.463170e+04f, 2.888721e+04f, 2.795153e+04f, 2.629836e+04f, 2.667237e+04f, 2.638737e+04f, 2.490630e+04f, 2.338930e+04f)

#define SKY_RAYLEIGH_SCATTERING \
  spectrum_set(3.945800e-02f, 2.939289e-02f, 2.235060e-02f, 1.730112e-02f, 1.360286e-02f, 1.084340e-02f, 8.750306e-03f, 7.139216e-03f)
#define SKY_MIE_SCATTERING (3.996f * 0.001f)

#define SKY_RAYLEIGH_EXTINCTION SKY_RAYLEIGH_SCATTERING
#define SKY_MIE_EXTINCTION (4.440f * 0.001f)
#define SKY_OZONE_EXTINCTION \
  spectrum_set(1.484836e-05f, 8.501668e-05f, 2.646158e-04f, 7.953520e-04f, 1.661103e-03f, 2.510733e-03f, 2.697211e-03f, 1.727741e-03f)

#define SKY_HEIGHT_OFFSET 0.0005f

#define SKY_MS_TEX_SIZE 32
#define SKY_TM_TEX_WIDTH 256
#define SKY_TM_TEX_HEIGHT 64

// Value must be now larger than (1 << 5) because max Kernel Block Dimension in x is 1024 (for z it is only 64)
#define SKY_MS_BASE (1 << 4)
#define SKY_MS_ITER (SKY_MS_BASE * SKY_MS_BASE)

////////////////////////////////////////////////////////////////////
// Sky Utils
////////////////////////////////////////////////////////////////////

// [Hil20]
__device__ float sky_unit_to_sub_uv(const float u, const float resolution) {
  return (u + 0.5f / resolution) * (resolution / (resolution + 1.0f));
}

// [Hil20]
__device__ float sky_sub_to_unit_uv(const float u, const float resolution) {
  return (u - 0.5f / resolution) * (resolution / (resolution - 1.0f));
}

__device__ float sky_rayleigh_phase(const float cos_angle) {
  return 3.0f * (1.0f + cos_angle * cos_angle) / (16.0f * 3.1415926535f);
}

__device__ float sky_rayleigh_density(const float height) {
  return 2.5f * device.scene.sky.base_density * expf(-height * (1.0f / device.scene.sky.rayleigh_falloff));
}

__device__ float sky_mie_phase(const float cos_angle) {
  const float g = device.scene.sky.mie_g;
  return (3.0f * (1.0f - g * g) * (1.0f + cos_angle * cos_angle))
         / (4.0f * PI * 2.0f * (2.0f + g * g) * pow(1.0f + g * g - 2.0f * g * cos_angle, 3.0f / 2.0f));
}

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

// [Wil21]
__device__ RGBF sky_compute_color_from_spectrum(const Spectrum radiance) {
  // Radiance to XYZ
  //  {{0.0076500,0.2345212,0.3027571,0.0357158,0.0698887,0.4671353,0.9607998,0.9384000},
  //   {0.0002170,0.0085286,0.0548572,0.2024885,0.7218860,0.9971143,0.8316430,0.4412000},
  //   {0.0362100,1.1380633,1.7012999,0.4867545,0.0752499,0.0074643,0.0014714,0.0002400}}
  // XYZ TO sRGB
  //  {{3.2406, -1.5372, -0.4986},
  //   {-0.9689, 1.8758, 0.0415},
  //   {0.0557, -0.2040, 1.0570}}
  //
  // I could premultiply the matrices, there is no particular reason besides lazyness as to why that did not happen yet.

  const float x = 0.0076500f * radiance.v[0] + 0.2345212f * radiance.v[1] + 0.3027571f * radiance.v[2] + 0.0357158f * radiance.v[3]
                  + 0.0698887f * radiance.v[4] + 0.4671353f * radiance.v[5] + 0.9607998f * radiance.v[6] + 0.9384000f * radiance.v[7];
  const float y = 0.0002170f * radiance.v[0] + 0.0085286f * radiance.v[1] + 0.0548572f * radiance.v[2] + 0.2024885f * radiance.v[3]
                  + 0.7218860f * radiance.v[4] + 0.9971143f * radiance.v[5] + 0.8316430f * radiance.v[6] + 0.4412000f * radiance.v[7];
  const float z = 0.0362100f * radiance.v[0] + 1.1380633f * radiance.v[1] + 1.7012999f * radiance.v[2] + 0.4867545f * radiance.v[3]
                  + 0.0752499f * radiance.v[4] + 0.0074643f * radiance.v[5] + 0.0014714f * radiance.v[6] + 0.0002400f * radiance.v[7];

  RGBF result;

  result.r = 3.2406f * x - 1.5372f * y - 0.4986f * z;
  result.g = -0.9689f * x + 1.8758f * y + 0.0415f * z;
  result.b = 0.0557f * x - 0.2040f * y + 1.0570f * z;

  return result;
}

////////////////////////////////////////////////////////////////////
// Sky LUT function
////////////////////////////////////////////////////////////////////

// [Hil20]
__device__ UV sky_transmittance_lut_uv(float height, float zenith_cos_angle) {
  height += SKY_EARTH_RADIUS;

  const float H   = sqrtf(fmaxf(0.0f, SKY_ATMO_RADIUS * SKY_ATMO_RADIUS - SKY_EARTH_RADIUS * SKY_EARTH_RADIUS));
  const float rho = sqrtf(fmaxf(0.0f, height * height - SKY_EARTH_RADIUS * SKY_EARTH_RADIUS));

  const float discriminant = height * height * (zenith_cos_angle * zenith_cos_angle - 1.0f) + SKY_ATMO_RADIUS * SKY_ATMO_RADIUS;
  const float d            = fmaxf(0.0f, (-height * zenith_cos_angle + sqrtf(discriminant)));

  const float d_min = SKY_ATMO_RADIUS - height;
  const float d_max = rho + H;
  const float u     = (d - d_min) / (d_max - d_min);
  const float v     = rho / H;

  return get_UV(u, v);
}

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
__global__ void sky_compute_transmittance_lut(float4* transmittance_tex_lower, float4* transmittance_tex_higher) {
  unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;

  const int amount = SKY_TM_TEX_WIDTH * SKY_TM_TEX_HEIGHT;

  while (id < amount) {
    const int x = id % SKY_TM_TEX_WIDTH;
    const int y = id / SKY_TM_TEX_WIDTH;

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

    for (int i = 0; i < steps; i++) {
      const float newReach = start + distance * (i + 0.3f) / steps;
      step_size            = newReach - reach;
      reach                = newReach;

      const vec3 pos     = add_vector(origin, scale_vector(ray, reach));
      const float height = sky_height(pos);

      const vec3 ray_scatter     = normalize_vector(sub_vector(sun_pos, pos));
      const float cos_angle      = dot_product(ray, ray_scatter);
      const float phase_rayleigh = sky_rayleigh_phase(cos_angle);
      const float phase_mie      = sky_mie_phase(cos_angle);

      const float zenith_cos_angle = dot_product(normalize_vector(pos), ray_scatter);

      const UV transmittance_uv       = sky_transmittance_lut_uv(height, zenith_cos_angle);
      const float4 transmittance_low  = tex2D<float4>(device.ptrs.sky_tm_luts[0], transmittance_uv.u, transmittance_uv.v);
      const float4 transmittance_high = tex2D<float4>(device.ptrs.sky_tm_luts[1], transmittance_uv.u, transmittance_uv.v);
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
  const float theta = 2.0f * PI * randA;
  const float phi   = 2.0f * PI * randB - PI;
  const vec3 ray    = angles_to_direction(phi, theta);

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
  bench_tic();

  if (instance->scene.sky.lut_initialized) {
    texture_free_atlas(instance->sky_tm_luts, 2);
    texture_free_atlas(instance->sky_ms_luts, 2);
  }

  instance->scene.sky.base_density           = instance->atmo_settings.base_density;
  instance->scene.sky.ground_visibility      = instance->atmo_settings.ground_visibility;
  instance->scene.sky.mie_density            = instance->atmo_settings.mie_density;
  instance->scene.sky.mie_falloff            = instance->atmo_settings.mie_falloff;
  instance->scene.sky.mie_g                  = instance->atmo_settings.mie_g;
  instance->scene.sky.ozone_absorption       = instance->atmo_settings.ozone_absorption;
  instance->scene.sky.ozone_density          = instance->atmo_settings.ozone_density;
  instance->scene.sky.ozone_layer_thickness  = instance->atmo_settings.ozone_layer_thickness;
  instance->scene.sky.rayleigh_density       = instance->atmo_settings.rayleigh_density;
  instance->scene.sky.rayleigh_falloff       = instance->atmo_settings.rayleigh_falloff;
  instance->scene.sky.multiscattering_factor = instance->atmo_settings.multiscattering_factor;

  raytrace_update_device_scene(instance);

  TextureRGBA luts_tm_tex[2];
  texture_create(luts_tm_tex + 0, SKY_TM_TEX_WIDTH, SKY_TM_TEX_HEIGHT, 1, SKY_TM_TEX_WIDTH, (void*) 0, TexDataFP32, TexStorageGPU);
  texture_create(luts_tm_tex + 1, SKY_TM_TEX_WIDTH, SKY_TM_TEX_HEIGHT, 1, SKY_TM_TEX_WIDTH, (void*) 0, TexDataFP32, TexStorageGPU);
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
  texture_create(luts_ms_tex + 0, SKY_MS_TEX_SIZE, SKY_MS_TEX_SIZE, 1, SKY_MS_TEX_SIZE, (void*) 0, TexDataFP32, TexStorageGPU);
  texture_create(luts_ms_tex + 1, SKY_MS_TEX_SIZE, SKY_MS_TEX_SIZE, 1, SKY_MS_TEX_SIZE, (void*) 0, TexDataFP32, TexStorageGPU);
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

  bench_toc((char*) "Sky LUT Computation");
}

////////////////////////////////////////////////////////////////////
// Atmosphere Integration
////////////////////////////////////////////////////////////////////

__device__ Spectrum sky_compute_atmosphere(
  Spectrum& transmittance_out, const vec3 origin, const vec3 ray, const float limit, const bool celestials, const bool cloud_shadows,
  const int steps) {
  Spectrum result = spectrum_set1(0.0f);

  const float2 path = sky_compute_path(origin, ray, SKY_EARTH_RADIUS, SKY_ATMO_RADIUS);

  const float start    = path.x;
  const float distance = fminf(path.y, limit - start);

  Spectrum transmittance = spectrum_get_ident();

  uint32_t random_key = device.ptrs.randoms[threadIdx.x + blockIdx.x * blockDim.x];

  if (distance > 0.0f) {
    float reach = start;
    float step_size;

    const float light_angle = sample_sphere_solid_angle(device.sun_pos, SKY_SUN_RADIUS, origin);

    for (int i = 0; i < steps; i++) {
      const float new_reach = start + distance * (i + white_noise_offset(random_key++)) / steps;
      step_size             = new_reach - reach;
      reach                 = new_reach;

      const vec3 pos     = add_vector(origin, scale_vector(ray, reach));
      const float height = sky_height(pos);

      const vec3 ray_scatter       = normalize_vector(sub_vector(device.sun_pos, pos));
      const float cos_angle        = dot_product(ray, ray_scatter);
      const float zenith_cos_angle = dot_product(normalize_vector(pos), ray_scatter);
      const float phase_rayleigh   = sky_rayleigh_phase(cos_angle);
      const float phase_mie        = sky_mie_phase(cos_angle);

      float shadow;
      if (cloud_shadows) {
        shadow = sph_ray_hit_p0(ray_scatter, pos, SKY_EARTH_RADIUS) ? 0.0f : cloud_shadow(pos, ray_scatter);
      }
      else {
        shadow = sph_ray_hit_p0(ray_scatter, pos, SKY_EARTH_RADIUS) ? 0.0f : 1.0f;
      }

      const UV transmittance_uv       = sky_transmittance_lut_uv(height, zenith_cos_angle);
      const float4 transmittance_low  = tex2D<float4>(device.ptrs.sky_tm_luts[0], transmittance_uv.u, transmittance_uv.v);
      const float4 transmittance_high = tex2D<float4>(device.ptrs.sky_tm_luts[1], transmittance_uv.u, transmittance_uv.v);
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

      const UV multiscattering_uv        = get_UV(zenith_cos_angle * 0.5f + 0.5f, height / SKY_ATMO_HEIGHT);
      const float4 multiscattering_low   = tex2D<float4>(device.ptrs.sky_ms_luts[0], multiscattering_uv.u, multiscattering_uv.v);
      const float4 multiscattering_high  = tex2D<float4>(device.ptrs.sky_ms_luts[1], multiscattering_uv.u, multiscattering_uv.v);
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
      const vec3 normal     = normalize_vector(sub_vector(moon_pos, device.moon_pos));
      const vec3 bounce_ray = normalize_vector(sub_vector(device.sun_pos, moon_pos));
      const float NdotL     = dot_product(normal, bounce_ray);

      if (!sphere_ray_hit(bounce_ray, moon_pos, get_vector(0.0f, 0.0f, 0.0f), SKY_EARTH_RADIUS) && NdotL > 0.0f) {
        const float light_angle = sample_sphere_solid_angle(device.sun_pos, SKY_SUN_RADIUS, moon_pos);
        const float weight      = device.scene.sky.sun_strength * device.scene.sky.moon_albedo * NdotL * light_angle / (2.0f * PI);

        result = spectrum_add(result, spectrum_mul(transmittance, spectrum_scale(SKY_SUN_RADIANCE, weight)));
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

  device.ptrs.randoms[threadIdx.x + blockIdx.x * blockDim.x] = random_key;

  transmittance_out = spectrum_mul(transmittance_out, transmittance);

  return result;
}

////////////////////////////////////////////////////////////////////
// Wrapper
////////////////////////////////////////////////////////////////////

__device__ RGBF sky_get_color(const vec3 origin, const vec3 ray, const float limit, const bool celestials, const int steps) {
  Spectrum unused = spectrum_set1(0.0f);

  const Spectrum radiance = sky_compute_atmosphere(unused, origin, ray, limit, celestials, false, steps);

  return sky_compute_color_from_spectrum(radiance);
}

// This is a quick way of obtaining the color of the sun disk times transmittance
// Note that it is not checked whether the sun is actually hit by ray, it is simply assumed
// Inscattering is not included
__device__ RGBF sky_get_sun_color(const vec3 origin, const vec3 ray) {
  const float height           = sky_height(origin);
  const float zenith_cos_angle = dot_product(normalize_vector(origin), ray);

  const UV transmittance_uv       = sky_transmittance_lut_uv(height, zenith_cos_angle);
  const float4 transmittance_low  = tex2D<float4>(device.ptrs.sky_tm_luts[0], transmittance_uv.u, transmittance_uv.v);
  const float4 transmittance_high = tex2D<float4>(device.ptrs.sky_tm_luts[1], transmittance_uv.u, transmittance_uv.v);
  const Spectrum extinction_sun   = spectrum_mul(spectrum_get_ident(), spectrum_merge(transmittance_low, transmittance_high));

  const Spectrum sun_radiance = spectrum_scale(SKY_SUN_RADIANCE, device.scene.sky.sun_strength);
  const Spectrum radiance     = spectrum_mul(extinction_sun, sun_radiance);

  return sky_compute_color_from_spectrum(radiance);
}

__device__ RGBF sky_trace_inscattering(const vec3 origin, const vec3 ray, const float limit, RGBF& record) {
  Spectrum transmittance = spectrum_set1(1.0f);

  const float base_range = (is_first_ray()) ? 40.0f : 80.0f;

  const int steps = fminf(fmaxf(0.5f, limit / base_range), 2.0f) * (device.scene.sky.steps / 6) + white_noise() - 0.5f;

  const Spectrum radiance = sky_compute_atmosphere(transmittance, origin, ray, limit, false, true, steps);

  const RGBF inscattering = mul_color(sky_compute_color_from_spectrum(radiance), record);

  record = mul_color(record, sky_compute_color_from_spectrum(transmittance));

  return inscattering;
}

////////////////////////////////////////////////////////////////////
// Kernel
////////////////////////////////////////////////////////////////////

__global__ __launch_bounds__(THREADS_PER_BLOCK, 7) void process_sky_tasks() {
  const int id = threadIdx.x + blockIdx.x * blockDim.x;

  const int task_count  = device.ptrs.task_counts[id * 6 + 2];
  const int task_offset = device.ptrs.task_offsets[id * 5 + 2];

  for (int i = 0; i < task_count; i++) {
    const SkyTask task = load_sky_task(device.trace_tasks + get_task_address(task_offset + i));
    const int pixel    = task.index.y * device.width + task.index.x;

    const RGBAhalf record = load_RGBAhalf(device.records + pixel);
    const uint32_t light  = device.ptrs.light_sample_history[pixel];

    RGBAhalf sky;

    if (device.scene.sky.use_hdri) {
      const float mip_bias = (device.iteration_type == TYPE_CAMERA) ? 0.0f : 1.0f;

      sky = mul_RGBAhalf(RGBF_to_RGBAhalf(sky_hdri_sample(task.ray, mip_bias)), record);
    }
    else {
      const vec3 origin = world_to_sky_transform(task.origin);

      const bool sample_sun = proper_light_sample(light, LIGHT_ID_SUN);

      RGBF sky_color;
      if (device.iteration_type == TYPE_LIGHT && sample_sun) {
        sky_color = sky_get_sun_color(origin, task.ray);
      }
      else if (device.iteration_type != TYPE_LIGHT) {
        sky_color = sky_get_color(origin, task.ray, FLT_MAX, sample_sun, device.scene.sky.steps);
      }
      else {
        continue;
      }

      sky = mul_RGBAhalf(RGBF_to_RGBAhalf(sky_color), record);
    }

    store_RGBAhalf(device.ptrs.frame_buffer + pixel, add_RGBAhalf(load_RGBAhalf(device.ptrs.frame_buffer + pixel), sky));
    write_albedo_buffer(RGBAhalf_to_RGBF(sky), pixel);
    write_normal_buffer(get_vector(0.0f, 0.0f, 0.0f), pixel);
  }
}

__global__ __launch_bounds__(THREADS_PER_BLOCK, 7) void process_debug_sky_tasks() {
  const int id = threadIdx.x + blockIdx.x * blockDim.x;

  const int task_count  = device.ptrs.task_counts[id * 6 + 2];
  const int task_offset = device.ptrs.task_offsets[id * 5 + 2];

  for (int i = 0; i < task_count; i++) {
    const SkyTask task = load_sky_task(device.trace_tasks + get_task_address(task_offset + i));
    const int pixel    = task.index.y * device.width + task.index.x;

    if (device.shading_mode == SHADING_ALBEDO) {
      store_RGBAhalf(
        device.ptrs.frame_buffer + pixel,
        RGBF_to_RGBAhalf(sky_get_color(world_to_sky_transform(task.origin), task.ray, FLT_MAX, true, device.scene.sky.steps)));
    }
    else if (device.shading_mode == SHADING_DEPTH) {
      const float value = __saturatef((1.0f / device.scene.camera.far_clip_distance) * 2.0f);
      store_RGBAhalf(device.ptrs.frame_buffer + pixel, get_RGBAhalf(value, value, value, value));
    }
  }
}

#endif /* CU_SKY_H */
