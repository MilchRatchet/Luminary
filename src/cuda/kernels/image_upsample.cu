#include "camera_post_common.cuh"
#include "math.cuh"
#include "memory.cuh"
#include "utils.cuh"

__global__ void image_upsample(
  const RGBF* source, const int sw, const int sh, RGBF* target, const RGBF* base, const int tw, const int th, const float sa,
  const float sb) {
  unsigned int id = THREAD_ID;

  const float scale_x = 1.0f / (tw - 1);
  const float scale_y = 1.0f / (th - 1);
  const float step_x  = 1.0f / (sw - 1);
  const float step_y  = 1.0f / (sh - 1);

  while (id < tw * th) {
    const int x = id % tw;
    const int y = id / tw;

    const float sx = scale_x * x + 0.5f * scale_x;
    const float sy = scale_y * y + 0.5f * scale_y;

    RGBF pixel = sample_pixel_clamp(source, sx - step_x, sy - step_y, sw, sh);

    pixel = add_color(pixel, scale_color(sample_pixel_clamp(source, sx, sy - step_y, sw, sh), 2.0f));
    pixel = add_color(pixel, sample_pixel_clamp(source, sx + step_x, sy - step_y, sw, sh));
    pixel = add_color(pixel, scale_color(sample_pixel_clamp(source, sx - step_x, sy, sw, sh), 2.0f));
    pixel = add_color(pixel, scale_color(sample_pixel_clamp(source, sx, sy, sw, sh), 4.0f));
    pixel = add_color(pixel, scale_color(sample_pixel_clamp(source, sx + step_x, sy, sw, sh), 2.0f));
    pixel = add_color(pixel, sample_pixel_clamp(source, sx - step_x, sy + step_y, sw, sh));
    pixel = add_color(pixel, scale_color(sample_pixel_clamp(source, sx, sy + step_y, sw, sh), 2.0f));
    pixel = add_color(pixel, sample_pixel_clamp(source, sx + step_x, sy + step_y, sw, sh));

    pixel = scale_color(pixel, 0.0625f * sa);

    RGBF base_pixel = load_RGBF(base + x + y * tw);
    base_pixel      = scale_color(base_pixel, sb);
    pixel           = add_color(pixel, base_pixel);

    store_RGBF(target + x + y * tw, pixel);

    id += blockDim.x * gridDim.x;
  }
}
