#include "camera_post_common.cuh"
#include "math.cuh"
#include "memory.cuh"
#include "utils.cuh"

__global__ void image_downsample(const RGBF* source, const int sw, const int sh, RGBF* target, const int tw, const int th) {
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

    RGBF a1 = sample_pixel_clamp(source, sx - 0.5f * step_x, sy - 0.5f * step_y, sw, sh);
    RGBF a2 = sample_pixel_clamp(source, sx + 0.5f * step_x, sy - 0.5f * step_y, sw, sh);
    RGBF a3 = sample_pixel_clamp(source, sx - 0.5f * step_x, sy + 0.5f * step_y, sw, sh);
    RGBF a4 = sample_pixel_clamp(source, sx + 0.5f * step_x, sy + 0.5f * step_y, sw, sh);

    RGBF pixel = add_color(add_color(a1, a2), add_color(a3, a4));

    pixel = add_color(pixel, sample_pixel_clamp(source, sx, sy, sw, sh));
    pixel = add_color(pixel, scale_color(sample_pixel_clamp(source, sx, sy - step_y, sw, sh), 0.5f));
    pixel = add_color(pixel, scale_color(sample_pixel_clamp(source, sx - step_x, sy, sw, sh), 0.5f));
    pixel = add_color(pixel, scale_color(sample_pixel_clamp(source, sx + step_x, sy, sw, sh), 0.5f));
    pixel = add_color(pixel, scale_color(sample_pixel_clamp(source, sx, sy + step_y, sw, sh), 0.5f));
    pixel = add_color(pixel, scale_color(sample_pixel_clamp(source, sx - step_x, sy - step_y, sw, sh), 0.25f));
    pixel = add_color(pixel, scale_color(sample_pixel_clamp(source, sx + step_x, sy - step_y, sw, sh), 0.25f));
    pixel = add_color(pixel, scale_color(sample_pixel_clamp(source, sx - step_x, sy + step_y, sw, sh), 0.25f));
    pixel = add_color(pixel, scale_color(sample_pixel_clamp(source, sx + step_x, sy + step_y, sw, sh), 0.25f));

    pixel = scale_color(pixel, 0.125f);

    store_RGBF(target + x + y * tw, pixel);

    id += blockDim.x * gridDim.x;
  }
}
