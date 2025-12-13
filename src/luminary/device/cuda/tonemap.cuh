#ifndef CU_TONEMAP_H
#define CU_TONEMAP_H

#include "math.cuh"
#include "random.cuh"
#include "utils.cuh"

LUMINARY_FUNCTION RGBF tonemap_aces(RGBF pixel) {
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

LUMINARY_FUNCTION RGBF uncharted2_partial(RGBF pixel) {
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

LUMINARY_FUNCTION RGBF tonemap_uncharted2(RGBF pixel) {
  const float exposure_bias = 2.0f;

  pixel = mul_color(pixel, get_color(exposure_bias, exposure_bias, exposure_bias));
  pixel = uncharted2_partial(pixel);

  RGBF scale = uncharted2_partial(get_color(11.2f, 11.2f, 11.2f));
  scale      = get_color(1.0f / scale.r, 1.0f / scale.g, 1.0f / scale.b);

  return mul_color(pixel, scale);
}

LUMINARY_FUNCTION RGBF tonemap_reinhard(RGBF pixel) {
  const float factor = 1.0f / (1.0f + color_luminance(pixel));
  pixel.r *= factor;
  pixel.g *= factor;
  pixel.b *= factor;

  return pixel;
}

//
// AgX approximation based on https://iolite-engine.com/blog_posts/minimal_agx_implementation
//

LUMINARY_FUNCTION float agx_constrast_approx_polynomial(float v) {
  const float v2 = v * v;
  const float v4 = v2 * v2;

  return 15.5f * v4 * v2 - 40.14f * v4 * v + 31.96f * v4 - 6.868f * v2 * v + 0.4298f * v2 + 0.1191f * v - 0.00232f;
}

LUMINARY_FUNCTION RGBF agx_contrast_approx(RGBF pixel) {
  const float r = agx_constrast_approx_polynomial(pixel.r);
  const float g = agx_constrast_approx_polynomial(pixel.g);
  const float b = agx_constrast_approx_polynomial(pixel.b);

  return get_color(r, g, b);
}

LUMINARY_FUNCTION RGBF agx_conversion(RGBF pixel) {
  RGBF agx;

  agx = get_color(0.0f, 0.0f, 0.0f);
  agx = add_color(agx, scale_color(get_color(0.842479062253094f, 0.0423282422610123f, 0.0423756549057051f), pixel.r));
  agx = add_color(agx, scale_color(get_color(0.0784335999999992f, 0.878468636469772f, 0.0784336f), pixel.g));
  agx = add_color(agx, scale_color(get_color(0.0792237451477643f, 0.0791661274605434f, 0.879142973793104f), pixel.b));

  const float min_val = -12.47393f;
  const float max_val = 4.026069f;

  // Clamp Value to inbetween the allowed values first
  agx = max_color(agx, get_color(0.00017578139f, 0.00017578139f, 0.00017578139f));

  agx.r = fminf(fmaxf(log2f(agx.r), min_val), max_val);
  agx.g = fminf(fmaxf(log2f(agx.g), min_val), max_val);
  agx.b = fminf(fmaxf(log2f(agx.b), min_val), max_val);

  agx.r = (agx.r - min_val) / (max_val - min_val);
  agx.g = (agx.g - min_val) / (max_val - min_val);
  agx.b = (agx.b - min_val) / (max_val - min_val);

  return agx_contrast_approx(agx);
}

LUMINARY_FUNCTION RGBF agx_inv_conversion(RGBF pixel) {
  RGBF agx;

  agx = get_color(0.0f, 0.0f, 0.0f);
  agx = add_color(agx, scale_color(get_color(1.19687900512017f, -0.0528968517574562f, -0.0529716355144438f), pixel.r));
  agx = add_color(agx, scale_color(get_color(-0.0980208811401368f, 1.15190312990417f, -0.0980434501171241f), pixel.g));
  agx = add_color(agx, scale_color(get_color(-0.0990297440797205f, -0.0989611768448433f, 1.15107367264116f), pixel.b));

  // Color could be negative now
  agx = max_color(agx, get_color(0.0f, 0.0f, 0.0f));

  // This should be sRGB to linear conversion
  agx.r = SRGB_to_linearRGB(agx.r);
  agx.g = SRGB_to_linearRGB(agx.g);
  agx.b = SRGB_to_linearRGB(agx.b);

  return agx;
}

LUMINARY_FUNCTION RGBF agx_look(RGBF pixel, const RGBF slope, const RGBF power, const float saturation) {
  const float lum = color_luminance(pixel);

  pixel = mul_color(pixel, slope);

  pixel.r = powf(pixel.r, power.r);
  pixel.g = powf(pixel.g, power.g);
  pixel.b = powf(pixel.b, power.b);

  pixel.r = lerp(lum, pixel.r, saturation);
  pixel.g = lerp(lum, pixel.g, saturation);
  pixel.b = lerp(lum, pixel.b, saturation);

  return pixel;
}

LUMINARY_FUNCTION RGBF tonemap_agx(RGBF pixel) {
  pixel = agx_conversion(pixel);
  pixel = agx_inv_conversion(pixel);

  return pixel;
}

LUMINARY_FUNCTION RGBF tonemap_agx_punchy(RGBF pixel) {
  pixel = agx_conversion(pixel);
  pixel = agx_look(pixel, get_color(1.0f, 1.0f, 1.0f), get_color(1.35f, 1.35f, 1.35f), 1.4f);
  pixel = agx_inv_conversion(pixel);

  return pixel;
}

LUMINARY_FUNCTION RGBF tonemap_agx_custom(RGBF pixel, const AGXCustomParams agx_params) {
  pixel = agx_conversion(pixel);

  RGBF slope = get_color(agx_params.slope, agx_params.slope, agx_params.slope);
  RGBF power = get_color(agx_params.power, agx_params.power, agx_params.power);

  pixel = agx_look(pixel, slope, power, agx_params.saturation);
  pixel = agx_inv_conversion(pixel);

  return pixel;
}

LUMINARY_FUNCTION RGBF
  tonemap_apply(RGBF pixel, const uint32_t x, const uint32_t y, const RGBF color_correction, const AGXCustomParams agx_params) {
  if (device.settings.shading_mode != LUMINARY_SHADING_MODE_DEFAULT)
    return pixel;

  if (device.settings.adaptive_sampling_output_mode != LUMINARY_ADAPTIVE_SAMPLING_OUTPUT_MODE_BEAUTY)
    return pixel;

  if (device.camera.purkinje) {
    pixel = purkinje_shift(pixel);
  }

  if (device.camera.use_color_correction) {
    RGBF hsv = rgb_to_hsv(pixel);

    hsv = add_color(hsv, color_correction);

    if (hsv.r < 0.0f)
      hsv.r += 1.0f;
    if (hsv.r > 1.0f)
      hsv.r -= 1.0f;
    hsv.g = __saturatef(hsv.g);
    if (hsv.b < 0.0f)
      hsv.b = 0.0f;

    pixel = hsv_to_rgb(hsv);
  }

  pixel.r *= device.camera.exposure;
  pixel.g *= device.camera.exposure;
  pixel.b *= device.camera.exposure;

  const float grain = device.camera.film_grain * (random_grain_mask(x, y) - 0.5f);

  pixel.r = fmaxf(0.0f, pixel.r + grain);
  pixel.g = fmaxf(0.0f, pixel.g + grain);
  pixel.b = fmaxf(0.0f, pixel.b + grain);

  switch (device.camera.tonemap) {
    case LUMINARY_TONEMAP_NONE:
      break;
    case LUMINARY_TONEMAP_ACES:
      pixel = tonemap_aces(pixel);
      break;
    case LUMINARY_TONEMAP_REINHARD:
      pixel = tonemap_reinhard(pixel);
      break;
    case LUMINARY_TONEMAP_UNCHARTED2:
      pixel = tonemap_uncharted2(pixel);
      break;
    case LUMINARY_TONEMAP_AGX:
      pixel = tonemap_agx(pixel);
      break;
    case LUMINARY_TONEMAP_AGX_PUNCHY:
      pixel = tonemap_agx_punchy(pixel);
      break;
    case LUMINARY_TONEMAP_AGX_CUSTOM:
      pixel = tonemap_agx_custom(pixel, agx_params);
      break;
  }

  return pixel;
}

#endif /* CU_TONEMAP_H */
