#ifndef CU_TONEMAP_H
#define CU_TONEMAP_H

#include <math.cuh>

__device__ RGBF tonemap_aces(RGBF pixel) {
  RGBF color;
  color.r = 0.59719f * pixel.r + 0.35458f * pixel.g + 0.04823f * pixel.b;
  color.g = 0.07600f * pixel.r + 0.90834f * pixel.g + 0.01566f * pixel.b;
  color.b = 0.02840f * pixel.r + 0.13383f * pixel.g + 0.83777f * pixel.b;

  RGBF a = add_color(color, get_color(0.0245786f, 0.0245786f, 0.0245786f));
  a      = mul_color(color, a);
  a      = add_color(a, get_color(-0.000090537f, -0.000090537f, -0.000090537f));
  RGBF b = mul_color(color, get_color(0.983729f, 0.983729f, 0.983729f));
  b      = add_color(b, get_color(0.432951f, 0.432951f, 0.432951f));
  b      = mul_color(color, b);
  b      = add_color(b, get_color(0.238081f, 0.238081f, 0.238081f));
  b      = get_color(1.0f / b.r, 1.0f / b.g, 1.0f / b.b);
  color  = mul_color(a, b);

  pixel.r = 1.60475f * color.r - 0.53108f * color.g - 0.07367f * color.b;
  pixel.g = -0.10208f * color.r + 1.10813f * color.g - 0.00605f * color.b;
  pixel.b = -0.00327f * color.r - 0.07276f * color.g + 1.07602f * color.b;

  return pixel;
}

__device__ RGBF uncharted2_partial(RGBF pixel) {
  const float a = 0.15f;
  const float b = 0.50f;
  const float c = 0.10f;
  const float d = 0.20f;
  const float e = 0.02f;
  const float f = 0.30f;

  RGBF result;
  result.r = ((pixel.r * (a * pixel.r + c * b) + d * e) / (pixel.r * (a * pixel.r + b) + d * f)) - e / f;
  result.g = ((pixel.g * (a * pixel.g + c * b) + d * e) / (pixel.g * (a * pixel.g + b) + d * f)) - e / f;
  result.b = ((pixel.b * (a * pixel.b + c * b) + d * e) / (pixel.b * (a * pixel.b + b) + d * f)) - e / f;

  return result;
}

__device__ RGBF tonemap_uncharted2(RGBF pixel) {
  const float exposure_bias = 2.0f;

  pixel = mul_color(pixel, get_color(exposure_bias, exposure_bias, exposure_bias));
  pixel = uncharted2_partial(pixel);

  RGBF scale = uncharted2_partial(get_color(11.2f, 11.2f, 11.2f));
  scale      = get_color(1.0f / scale.r, 1.0f / scale.g, 1.0f / scale.b);

  return mul_color(pixel, scale);
}

__device__ RGBF tonemap_reinhard(RGBF pixel) {
  const float factor = 1.0f / (1.0f + luminance(pixel));
  pixel.r *= factor;
  pixel.g *= factor;
  pixel.b *= factor;

  return pixel;
}

#endif /* CU_TONEMAP_H */
