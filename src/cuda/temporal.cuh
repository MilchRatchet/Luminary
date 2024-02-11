#ifndef CU_TEMPORAL_H
#define CU_TEMPORAL_H

#include "camera_post_common.cuh"
#include "utils.cuh"

/*
 * TAA implementation based on blog by Alex Tardiff (http://alextardif.com/TAA.html)
 * CatmullRom filter implementation based on https://gist.github.com/TheRealMJP/c83b8c0f46b63f3a88a5986f4fa982b1
 */

LUM_DEVICE_FUNC RGBF sample_pixel_catmull_rom(const RGBF* image, float x, float y, const int width, const int height) {
  float px = floorf(x - 0.5f) + 0.5f;
  float py = floorf(y - 0.5f) + 0.5f;

  float fx = x - px;
  float fy = y - py;

  float wx0 = fx * (-0.5f + fx * (1.0f - 0.5f * fx));
  float wx1 = 1.0f + fx * fx * (-2.5f + 1.5f * fx);
  float wx2 = fx * (0.5f + fx * (2.0f - 1.5f * fx));
  float wx3 = fx * fx * (-0.5f + 0.5f * fx);
  float wy0 = fy * (-0.5f + fy * (1.0f - 0.5f * fy));
  float wy1 = 1.0f + fy * fy * (-2.5f + 1.5f * fy);
  float wy2 = fy * (0.5f + fy * (2.0f - 1.5f * fy));
  float wy3 = fy * fy * (-0.5f + 0.5f * fy);

  float wx12 = wx1 + wx2;
  float wy12 = wy1 + wy2;

  float ox12 = wx2 / wx12;
  float oy12 = wy2 / wy12;

  float x0  = px - 1.0f;
  float y0  = py - 1.0f;
  float x3  = px + 2.0f;
  float y3  = py + 2.0f;
  float x12 = px + ox12;
  float y12 = py + oy12;

  x0 /= (width - 1);
  y0 /= (height - 1);
  x3 /= (width - 1);
  y3 /= (height - 1);
  x12 /= (width - 1);
  y12 /= (height - 1);

  RGBF result = get_color(0.0f, 0.0f, 0.0f);

  result = add_color(result, scale_color(sample_pixel_clamp(image, x0, y0, width, height), wx0 * wy0));
  result = add_color(result, scale_color(sample_pixel_clamp(image, x12, y0, width, height), wx12 * wy0));
  result = add_color(result, scale_color(sample_pixel_clamp(image, x3, y0, width, height), wx3 * wy0));

  result = add_color(result, scale_color(sample_pixel_clamp(image, x0, y12, width, height), wx0 * wy12));
  result = add_color(result, scale_color(sample_pixel_clamp(image, x12, y12, width, height), wx12 * wy12));
  result = add_color(result, scale_color(sample_pixel_clamp(image, x3, y12, width, height), wx3 * wy12));

  result = add_color(result, scale_color(sample_pixel_clamp(image, x0, y3, width, height), wx0 * wy3));
  result = add_color(result, scale_color(sample_pixel_clamp(image, x12, y3, width, height), wx12 * wy3));
  result = add_color(result, scale_color(sample_pixel_clamp(image, x3, y3, width, height), wx3 * wy3));

  return result;
}

#endif /* CU_TEMPORAL_H */
