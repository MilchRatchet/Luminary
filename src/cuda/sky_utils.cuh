#ifndef SKY_UTILS_CUH
#define SKY_UTILS_CUH

#include "sky_defines.h"

__device__ float sky_height(const vec3 point) {
  return get_length(point) - SKY_EARTH_RADIUS;
}

__device__ float world_to_sky_scale(float input) {
  return input * 0.001f;
}

__device__ float sky_to_world_scale(float input) {
  return input * 1000.0f;
}

__device__ vec3 world_to_sky_transform(vec3 input) {
  vec3 result;

  result.x = world_to_sky_scale(input.x);
  result.y = world_to_sky_scale(input.y) + SKY_EARTH_RADIUS;
  result.z = world_to_sky_scale(input.z);

  result = add_vector(result, device.scene.sky.geometry_offset);

  return result;
}

__device__ vec3 sky_to_world_transform(vec3 input) {
  vec3 result;

  input = sub_vector(input, device.scene.sky.geometry_offset);

  result.x = sky_to_world_scale(input.x);
  result.y = sky_to_world_scale(input.y - SKY_EARTH_RADIUS);
  result.z = sky_to_world_scale(input.z);

  return result;
}

__device__ bool sky_ray_hits_sun(const vec3 origin_sky, const vec3 ray) {
  return sphere_ray_intersection(ray, origin_sky, device.sun_pos, SKY_SUN_RADIUS) != FLT_MAX;
}

__device__ RGBF sky_hdri_sample(const vec3 ray, const float mip_bias) {
  const float theta = atan2f(ray.z, ray.x);
  const float phi   = asinf(ray.y);

  const float u = (theta + PI) / (2.0f * PI);
  const float v = 1.0f - ((phi + 0.5f * PI) / PI);

  const float4 hdri = tex2DLod<float4>(device.ptrs.sky_hdri_luts[0].tex, u, v, mip_bias + device.scene.sky.hdri_mip_bias);

  return get_color(hdri.x, hdri.y, hdri.z);
}

__device__ float sky_hdri_sample_alpha(const vec3 ray) {
  const float theta = atan2f(ray.z, ray.x);
  const float phi   = asinf(ray.y);

  const float u = (theta + PI) / (2.0f * PI);
  const float v = 1.0f - ((phi + 0.5f * PI) / PI);

  return tex2D<float>(device.ptrs.sky_hdri_luts[1].tex, u, v);
}

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

__device__ float sky_mie_phase(const float cos_angle, const JendersieEonParams params) {
  return jendersie_eon_phase_function(cos_angle, params);
}

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

// Roughly based on the solar flux reported in "The spectral irradiance of the moon" by H. Kieffer and T. Stone,
// however, I only looked at the table and didn't read the paper.
#define SKY_MOON_SOLAR_FLUX spectrum_set(1.7f, 1.8f, 2.0f, 1.9f, 1.87f, 1.7f, 1.65f, 1.55f)

#define SKY_HEIGHT_OFFSET 0.0005f

#define SKY_MS_TEX_SIZE 32
#define SKY_TM_TEX_WIDTH 256
#define SKY_TM_TEX_HEIGHT 64

// Value must be now larger than (1 << 5) because max Kernel Block Dimension in x is 1024 (for z it is only 64)
#define SKY_MS_BASE (1 << 4)
#define SKY_MS_ITER (SKY_MS_BASE * SKY_MS_BASE)

#define SKY_WORLD_REFERENCE_HEIGHT (get_length(world_to_sky_transform(get_vector(0.0f, 0.0f, 0.0f))))

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

  return get_uv(u, v);
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

  // Negative numbers can show up, hence we need to clamp to 0 here.
  RGBF result;
  result.r = fmaxf(3.2406f * x - 1.5372f * y - 0.4986f * z, 0.0f);
  result.g = fmaxf(-0.9689f * x + 1.8758f * y + 0.0415f * z, 0.0f);
  result.b = fmaxf(0.0557f * x - 0.2040f * y + 1.0570f * z, 0.0f);

  return result;
}

// This is a quick way of obtaining the color of the sun disk times transmittance
// Note that it is not checked whether the sun is actually hit by ray, it is simply assumed
// Inscattering is not included
__device__ RGBF sky_get_sun_color(const vec3 origin, const vec3 ray, const bool include_cloud_hdri = true) {
  const float height           = sky_height(origin);
  const float zenith_cos_angle = dot_product(normalize_vector(origin), ray);

  const UV transmittance_uv       = sky_transmittance_lut_uv(height, zenith_cos_angle);
  const float4 transmittance_low  = tex2D<float4>(device.ptrs.sky_tm_luts[0].tex, transmittance_uv.u, transmittance_uv.v);
  const float4 transmittance_high = tex2D<float4>(device.ptrs.sky_tm_luts[1].tex, transmittance_uv.u, transmittance_uv.v);
  const Spectrum extinction_sun   = spectrum_mul(spectrum_get_ident(), spectrum_merge(transmittance_low, transmittance_high));

  const Spectrum sun_radiance = spectrum_scale(SKY_SUN_RADIANCE, device.scene.sky.sun_strength);
  const Spectrum radiance     = spectrum_mul(extinction_sun, sun_radiance);

  RGBF sun_color = sky_compute_color_from_spectrum(radiance);

  if (include_cloud_hdri && device.scene.sky.hdri_active) {
    const float cloud_alpha = sky_hdri_sample_alpha(ray);
    sun_color               = scale_color(sun_color, cloud_alpha);
  }

  return sun_color;
}

#endif /* SKY_UTILS_CUH */
