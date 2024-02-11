#include "camera_post_common.cuh"
#include "camera_post_lens_flare.cuh"
#include "math.cuh"
#include "memory.cuh"
#include "utils.cuh"

__global__ void camera_lens_flare_halo(const RGBF* src, const int sw, const int sh, RGBF* target, const int tw, const int th) {
  unsigned int id = THREAD_ID;

  const float scale_x = 1.0f / (tw - 1);
  const float scale_y = 1.0f / (th - 1);

  while (id < tw * th) {
    const int x = id % tw;
    const int y = id / tw;

    const float width        = 0.45f;
    const float chroma_shift = 0.005f;

    UV src_uv;
    src_uv.u = scale_x * x + 0.5f * scale_x;
    src_uv.v = scale_y * y + 0.5f * scale_y;

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

    const RGBF pr = sample_pixel_border(src, src_uv_r.u, src_uv_r.v, sw, sh);
    const RGBF pg = sample_pixel_border(src, src_uv_g.u, src_uv_g.v, sw, sh);
    const RGBF pb = sample_pixel_border(src, src_uv_b.u, src_uv_b.v, sw, sh);

    RGBF pixel = get_color(pr.r, pg.g, pb.b);

    store_RGBF(target + x + y * tw, pixel);

    id += blockDim.x * gridDim.x;
  }
}
