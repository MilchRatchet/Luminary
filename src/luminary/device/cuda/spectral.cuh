#ifndef CU_LUMINARY_SPECTRAL_H
#define CU_LUMINARY_SPECTRAL_H

#include "math.cuh"
#include "utils.cuh"

__device__ float spectral_sample_wavelength(const float random, float& pdf) {
  uint32_t range = SPECTRAL_MAX_WAVELENGTH - SPECTRAL_MIN_WAVELENGTH + 1;
  uint32_t first = 0;

  while (range > 0) {
    const uint32_t step   = range >> 1;
    const uint32_t index  = first + step;
    const float cdf_value = __ldg(device.ptrs.spectral_cdf + index);

    if (cdf_value <= random) {
      first = index + 1;
      range -= step + 1;
    }
    else {
      range = step;
    }
  }

  const uint32_t index_lower  = first - 1;
  const uint32_t index_higher = first;

  const float cdf_lower  = __ldg(device.ptrs.spectral_cdf + index_lower);
  const float cdf_higher = __ldg(device.ptrs.spectral_cdf + index_higher);

  pdf = cdf_higher - cdf_lower;

  const float wavelength_lower  = SPECTRAL_MIN_WAVELENGTH + index_lower;
  const float wavelength_higher = SPECTRAL_MIN_WAVELENGTH + index_higher;

  return remap(random, cdf_lower, cdf_higher, wavelength_lower, wavelength_higher);
}

__device__ RGBF spectral_xyz_to_rgb(const float3 XYZ) {
  RGBF result;

  result.r = 3.2406f * XYZ.x - 1.5372f * XYZ.y - 0.4986f * XYZ.z;
  result.g = -0.9689f * XYZ.x + 1.8758f * XYZ.y + 0.0415f * XYZ.z;
  result.b = 0.0557f * XYZ.x - 0.2040f * XYZ.y + 1.0570f * XYZ.z;

  return result;
}

__device__ float3 spectral_wavelength_to_xyz(const float wavelength) {
  const float sample = remap01(wavelength, SPECTRAL_MIN_WAVELENGTH, SPECTRAL_MAX_WAVELENGTH);

  const float2 XY = tex2D<float2>(device.spectral_xy_lut_tex.handle, sample, 0.0f);
  const float Z   = tex2D<float>(device.spectral_z_lut_tex.handle, sample, 0.0f);

  return make_float3(XY.x, XY.y, Z);
}

__device__ RGBF spectral_wavelength_to_rgb(const float wavelength) {
  const float3 XYZ = spectral_wavelength_to_xyz(wavelength);

  return spectral_xyz_to_rgb(XYZ);
}

#endif /* CU_LUMINARY_SPECTRAL_H */
