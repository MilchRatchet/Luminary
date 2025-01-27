#ifndef CU_CAMERA_POST_LENS_FLARE_H
#define CU_CAMERA_POST_LENS_FLARE_H

#include "utils.h"

/*
 * I based this implementation on a nice blog post by Lena Piquet:
 * https://www.froyok.fr/blog/2021-09-ue4-custom-lens-flare/
 */

LUMINARY_KERNEL void camera_post_lens_flare_ghosts(const KernelArgsCameraPostLensFlareGhosts args) {
  uint32_t id = THREAD_ID;

  const uint32_t ghost_count = 8;
  const float ghost_scales[] = {-1.0f, -0.5f, -0.25f, -2.0f, -3.0f, -4.0f, 2.0f, 0.25f};
  const RGBF ghost_colors[]  = {get_color(1.0f, 0.5f, 1.0f),   get_color(0.1f, 0.5f, 1.0f),  get_color(1.0f, 1.0f, 0.5f),
                                get_color(1.0f, 0.75f, 0.1f),  get_color(1.0f, 0.1f, 1.0f),  get_color(1.0f, 0.5f, 0.1f),
                                get_color(0.25f, 0.1f, 0.75f), get_color(0.75f, 0.1f, 0.25f)};

  const float scale_x = 1.0f / (args.tw - 1);
  const float scale_y = 1.0f / (args.th - 1);

  while (id < args.tw * args.th) {
    const uint32_t y = id / args.tw;
    const uint32_t x = id - y * args.tw;

    const float sx = scale_x * x;
    const float sy = scale_y * y;

    RGBF pixel = get_color(0.0f, 0.0f, 0.0f);

    for (uint32_t i = 0; i < ghost_count; i++) {
      const float sxi = ((sx - 0.5f) * ghost_scales[i]) + 0.5f;
      const float syi = ((sy - 0.5f) * ghost_scales[i]) + 0.5f;

      RGBF ghost = sample_pixel_border(args.src, sxi, syi, args.sw, args.sh);

      ghost = scale_color(ghost, 0.0005f);
      ghost = mul_color(ghost, ghost_colors[i]);

      pixel = add_color(pixel, ghost);
    }

    store_RGBF(args.dst + x + y * args.tw, pixel);

    id += blockDim.x * gridDim.x;
  }
}

__device__ UV _lens_flare_fisheye(UV uv, const float compression, const float zoom) {
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

LUMINARY_KERNEL void camera_post_lens_flare_halo(const KernelArgsCameraPostLensFlareHalo args) {
  uint32_t id = THREAD_ID;

  const float scale_x = 1.0f / (args.tw - 1);
  const float scale_y = 1.0f / (args.th - 1);

  while (id < args.tw * args.th) {
    const uint32_t y = id / args.tw;
    const uint32_t x = id - y * args.tw;

    const float width        = 0.45f;
    const float chroma_shift = 0.005f;

    UV src_uv;
    src_uv.u = scale_x * x;
    src_uv.v = scale_y * y;

    const UV fish_uv          = _lens_flare_fisheye(src_uv, 1.0f, 1.0f);
    const float2 neg_offset   = make_float2(0.5f - src_uv.u, 0.5f - src_uv.v);
    const float offset_length = sqrtf(neg_offset.x * neg_offset.x + neg_offset.y * neg_offset.y);
    const float factor        = width / fmaxf(eps, offset_length);
    const float2 v_halo       = make_float2(factor * neg_offset.x, factor * neg_offset.y);

    UV src_uv_r;
    src_uv_r.u = (fish_uv.u - 0.5f) * (1.0f + chroma_shift) + 0.5f + v_halo.x;
    src_uv_r.v = (fish_uv.v - 0.5f) * (1.0f + chroma_shift) + 0.5f + v_halo.y;

    UV src_uv_g;
    src_uv_g.u = fish_uv.u + v_halo.x;
    src_uv_g.v = fish_uv.v + v_halo.y;

    UV src_uv_b;
    src_uv_b.u = (fish_uv.u - 0.5f) * (1.0f - chroma_shift) + 0.5f + v_halo.x;
    src_uv_b.v = (fish_uv.v - 0.5f) * (1.0f - chroma_shift) + 0.5f + v_halo.y;

    const RGBF pr = sample_pixel_border(args.src, src_uv_r.u, src_uv_r.v, args.sw, args.sh);
    const RGBF pg = sample_pixel_border(args.src, src_uv_g.u, src_uv_g.v, args.sw, args.sh);
    const RGBF pb = sample_pixel_border(args.src, src_uv_b.u, src_uv_b.v, args.sw, args.sh);

    const RGBF pixel = get_color(pr.r, pg.g, pb.b);

    store_RGBF(args.dst + x + y * args.tw, pixel);

    id += blockDim.x * gridDim.x;
  }
}

#endif /* CU_CAMERA_POST_LENS_FLARE_H */
