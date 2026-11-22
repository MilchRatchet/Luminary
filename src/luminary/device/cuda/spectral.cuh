#ifndef CU_LUMINARY_SPECTRAL_H
#define CU_LUMINARY_SPECTRAL_H

#include "math.cuh"
#include "utils.cuh"

LUMINARY_FUNCTION float spectral_sample_wavelength(const float random, float& pdf) {
  constexpr float wavelength_range = SPECTRAL_MAX_WAVELENGTH - SPECTRAL_MIN_WAVELENGTH;
  pdf                              = 1.0f / wavelength_range;

  return SPECTRAL_MIN_WAVELENGTH + random * wavelength_range;
}

LUMINARY_FUNCTION RGBF spectral_xyz_to_rgb(const float3 XYZ) {
  RGBF result;

  result.r = 3.2406f * XYZ.x - 1.5372f * XYZ.y - 0.4986f * XYZ.z;
  result.g = -0.9689f * XYZ.x + 1.8758f * XYZ.y + 0.0415f * XYZ.z;
  result.b = 0.0557f * XYZ.x - 0.2040f * XYZ.y + 1.0570f * XYZ.z;

  return result;
}

LUMINARY_FUNCTION float3 spectral_wavelength_to_xyz(const float wavelength) {
  const float sample = (wavelength - SPECTRAL_MIN_WAVELENGTH + 0.5f) / (SPECTRAL_MAX_WAVELENGTH - SPECTRAL_MIN_WAVELENGTH + 1.0f);

  const float2 XY = tex2D<float2>(device.spectral_xy_lut_tex.handle, sample, 0.0f);
  const float Z   = tex2D<float>(device.spectral_z_lut_tex.handle, sample, 0.0f);

  return make_float3(XY.x, XY.y, Z);
}

LUMINARY_FUNCTION RGBF spectral_wavelength_to_rgb(const float wavelength) {
  const float3 XYZ = spectral_wavelength_to_xyz(wavelength);

  return spectral_xyz_to_rgb(XYZ);
}

#endif /* CU_LUMINARY_SPECTRAL_H */
