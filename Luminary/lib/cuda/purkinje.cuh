#ifndef CU_PURKINJE_H
#define CU_PURKINJE_H

/*
 * This is based on the paper "Perceptually Based Tone Mapping for Low-Light Conditions" by Adam G. Kirk and James F. O'Brien (2011)
 * and the SIGGRAPH 2021 presentation by Jasmin Patry about Ghost of Tsushima.
 *
 * The sRGB to LMSR transformation is taken from the GIMP plugin "Low-Light Tone Mapper" by Yeon Jin Lee (2012).
 *
 * The weighting of the rod response based gain custom made as the blending in the paper did not yield good results for me
 * and the weighting is not mentioned in the SIGGRAPH presentation besides in the (probably incorrect) shader code that is shown.
 *
 * The LMS to sRGB transformation could be optimized but it does not really matter.
 */

#include "image.h"
#include "math.cuh"

__device__ RGBF purkinje_shift(RGBF pixel) {
  // sRGB => LMSR
  float long_cone   = 0.096869562190332f * pixel.r + 0.318940374720484f * pixel.g - 0.188428411786113f * pixel.b;
  float medium_cone = 0.020208210904239f * pixel.r + 0.291385283197581f * pixel.g - 0.090918262127325f * pixel.b;
  float short_cone  = 0.002760510899553f * pixel.r - 0.008341563564118f * pixel.g + 0.067213551661950f * pixel.b;
  float rod         = -0.007607045462440f * pixel.r + 0.122492925567539f * pixel.g + 0.022445835141881f * pixel.b;

  const float kappa1 = device_scene.camera.purkinje_kappa1;
  const float kappa2 = device_scene.camera.purkinje_kappa2;
  const float lm     = 1.0f / 0.63721f;
  const float mm     = 1.0f / 0.39242f;
  const float sm     = 1.0f / 1.6064f;

  RGBF signal;

  signal.r = 1.0f / sqrtf(1.0f + (1.0f / 3.0f) * lm * (long_cone + kappa1 * rod));
  signal.g = 1.0f / sqrtf(1.0f + (1.0f / 3.0f) * mm * (medium_cone + kappa1 * rod));
  signal.b = 1.0f / sqrtf(1.0f + (1.0f / 3.0f) * sm * (short_cone + kappa2 * rod));

  const float K  = 45.0f;
  const float S  = 10.0f;
  const float k3 = 0.6f;
  const float rw = 0.139f;
  const float p  = 0.6189f;

  RGBF opponent;

  opponent.r = ((-k3 - rw) * signal.r + (1.0f + k3 * rw) * signal.g) * kappa1 * lm;
  opponent.g = (p * k3 * signal.r + (1.0f - p) * k3 * signal.g + signal.b) * kappa1 * mm;
  opponent.b = (p * S * signal.r + (1.0f - p) * S * signal.g) * kappa2 * sm;

  opponent = scale_color(opponent, (K / S) * rod);

  RGBF LMS;

  LMS.r = long_cone + 0.5f * (opponent.b - opponent.r);
  LMS.g = medium_cone + 0.5f * (opponent.b + opponent.r);
  LMS.b = short_cone + opponent.g + opponent.b;

  RGBF XYZ;

  // LMS => XYZ
  XYZ.r = 1.9102f * LMS.r - 1.1121f * LMS.g + 0.2019f * LMS.b;
  XYZ.g = 0.3710f * LMS.r + 0.6291f * LMS.g + 0.0000f * LMS.b;
  XYZ.b = 0.0000f * LMS.r + 0.0000f * LMS.g + 1.0000f * LMS.b;

  RGBF sRGB;

  // XYZ => sRGB
  sRGB.r = 3.2405f * XYZ.r - 1.5371f * XYZ.g - 0.4985f * XYZ.b;
  sRGB.g = -0.9693f * XYZ.r + 1.876f * XYZ.g + 0.0416f * XYZ.b;
  sRGB.b = 0.0556f * XYZ.r - 0.2040f * XYZ.g + 1.0572f * XYZ.b;

  float blend = __saturatef(1.0f - 100.0f * luminance(pixel));

  blend *= blend;

  return add_color(scale_color(pixel, 1.0f - blend), scale_color(sRGB, blend));
}

#endif /* CU_PURKINJE_H */
