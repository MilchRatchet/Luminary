#ifndef CU_TEXTURE_UTILS_H
#define CU_TEXTURE_UTILS_H

#include "utils.cuh"

struct TextureLoadArgs {
  bool flip_v;
  bool apply_gamma;
  float mip_level;
  float4 default_result;
} typedef TextureLoadArgs;

LUMINARY_FUNCTION TextureLoadArgs texture_get_default_args() {
  TextureLoadArgs args;

  args.flip_v         = true;
  args.apply_gamma    = true;
  args.mip_level      = 0.0f;
  args.default_result = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

  return args;
}

LUMINARY_FUNCTION bool texture_is_valid(const DeviceTextureObject tex) {
  return tex.handle != TEXTURE_OBJECT_INVALID;
}

LUMINARY_FUNCTION float4 texture_load(DeviceTextureObject tex, UV uv, const TextureLoadArgs args = texture_get_default_args()) {
  if (texture_is_valid(tex) == false)
    return args.default_result;

  const float u = uv.u;
  const float v = args.flip_v ? 1.0f - uv.v : uv.v;

  float4 result = tex2DLod<float4>(tex.handle, u, v, args.mip_level);

  if (args.apply_gamma) {
    result.x = powf(result.x, tex.gamma);
    result.y = powf(result.y, tex.gamma);
    result.z = powf(result.z, tex.gamma);
    //  Gamma is never applied to the alpha of a texture according to PNG standard.
  }

  return result;
}

#endif /* CU_TEXTURE_UTILS_H */
