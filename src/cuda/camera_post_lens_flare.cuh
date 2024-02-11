#ifndef CU_CAMERA_POST_LENS_FLARE_H
#define CU_CAMERA_POST_LENS_FLARE_H

#include "utils.cuh"
#include "utils.h"

/*
 * I based this implementation on a nice blog post by Lena Piquet:
 * https://www.froyok.fr/blog/2021-09-ue4-custom-lens-flare/
 */

LUM_DEVICE_FUNC UV _lens_flare_fisheye(UV uv, const float compression, const float zoom) {
  uv.u = 2.0f * uv.u - 1.0f;
  uv.v = 2.0f * uv.v - 1.0f;

  const float scale           = compression * atanf(1.0f / compression);
  const float radius_distance = sqrtf(uv.u * uv.u + uv.v * uv.v) * scale;
  const float radius_dir      = compression * tanf(radius_distance / compression) * zoom;
  const float phi             = atan2f(uv.v, uv.u);

  UV result;

  result.u = __saturatef(0.5f * (radius_dir * cosf(phi) + 1.0f));
  result.v = __saturatef(0.5f * (radius_dir * sinf(phi) + 1.0f));

  return result;
}

#endif /* CU_CAMERA_POST_LENS_FLARE_H */
