#ifndef CU_TEXTURE_UTILS_H
#define CU_TEXTURE_UTILS_H

#include "utils.cuh"

__device__ float4 texture_load(const DeviceTextureObject tex, const UV uv) {
  float4 v = tex2D<float4>(tex.handle, uv.u, 1.0f - uv.v);

  v.x = powf(v.x, tex.gamma);
  v.y = powf(v.y, tex.gamma);
  v.z = powf(v.z, tex.gamma);
  // Gamma is never applied to the alpha of a texture according to PNG standard.

  return v;
}

#endif /* CU_TEXTURE_UTILS_H */
