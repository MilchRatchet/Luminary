#include "camera_post_common.cuh"
#include "math.cuh"
#include "memory.cuh"
#include "utils.cuh"

__global__ void camera_lens_flare_ghosts(const RGBF* source, const int sw, const int sh, RGBF* target, const int tw, const int th) {
  unsigned int id = THREAD_ID;

  const int ghost_count      = 8;
  const float ghost_scales[] = {-1.0f, -0.5f, -0.25f, -2.0f, -3.0f, -4.0f, 2.0f, 0.25f};
  const RGBF ghost_colors[]  = {get_color(1.0f, 0.5f, 1.0f),   get_color(0.1f, 0.5f, 1.0f),  get_color(1.0f, 1.0f, 0.5f),
                                get_color(1.0f, 0.75f, 0.1f),  get_color(1.0f, 0.1f, 1.0f),  get_color(1.0f, 0.5f, 0.1f),
                                get_color(0.25f, 0.1f, 0.75f), get_color(0.75f, 0.1f, 0.25f)};

  const float scale_x = 1.0f / (tw - 1);
  const float scale_y = 1.0f / (th - 1);

  while (id < tw * th) {
    const int x = id % tw;
    const int y = id / tw;

    const float sx = scale_x * x + 0.5f * scale_x;
    const float sy = scale_y * y + 0.5f * scale_y;

    RGBF pixel = get_color(0.0f, 0.0f, 0.0f);

    for (int i = 0; i < ghost_count; i++) {
      const float sxi = ((sx - 0.5f) * ghost_scales[i]) + 0.5f;
      const float syi = ((sy - 0.5f) * ghost_scales[i]) + 0.5f;

      RGBF ghost = sample_pixel_border(source, sxi, syi, sw, sh);

      ghost = scale_color(ghost, 0.0005f);
      ghost = mul_color(ghost, ghost_colors[i]);

      pixel = add_color(pixel, ghost);
    }

    store_RGBF(target + x + y * tw, pixel);

    id += blockDim.x * gridDim.x;
  }
}
