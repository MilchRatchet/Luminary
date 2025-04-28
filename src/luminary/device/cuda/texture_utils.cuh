#ifndef CU_TEXTURE_UTILS_H
#define CU_TEXTURE_UTILS_H

#include "utils.cuh"

__device__ float4 texture_load(const DeviceTextureObject tex, const UV uv, const bool flip_v = true, const bool apply_gamma = true) {
  const float u = uv.u;
  const float v = flip_v ? 1.0f - uv.v : uv.v;

  float4 result = tex2D<float4>(tex.handle, u, v);

  if (apply_gamma) {
    result.x = powf(result.x, tex.gamma);
    result.y = powf(result.y, tex.gamma);
    result.z = powf(result.z, tex.gamma);
    //  Gamma is never applied to the alpha of a texture according to PNG standard.
  }

  return result;
}

#endif /* CU_TEXTURE_UTILS_H */
