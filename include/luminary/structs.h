/*
  Copyright (c) 2021-2024, MilchRatchet

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
*/

#ifndef LUMINARY_API_STRUCTS_H
#define LUMINARY_API_STRUCTS_H

#include <luminary/api_utils.h>

////////////////////////////////////////////////////////////////////
// General
////////////////////////////////////////////////////////////////////

LUMINARY_API enum LuminaryShadingMode {
  LUMINARY_SHADING_DEFAULT        = 0,
  LUMINARY_SHADING_ALBEDO         = 1,
  LUMINARY_SHADING_DEPTH          = 2,
  LUMINARY_SHADING_NORMAL         = 3,
  LUMINARY_SHADING_HEAT           = 4,
  LUMINARY_SHADING_IDENTIFICATION = 5,
  LUMINARY_SHADING_LIGHTS         = 6
} typedef LuminaryShadingMode;

////////////////////////////////////////////////////////////////////
// Camera
////////////////////////////////////////////////////////////////////

LUMINARY_API enum LuminaryFilter {
  LUMINARY_FILTER_NONE       = 0,
  LUMINARY_FILTER_GRAY       = 1,
  LUMINARY_FILTER_SEPIA      = 2,
  LUMINARY_FILTER_GAMEBOY    = 3,
  LUMINARY_FILTER_2BITGRAY   = 4,
  LUMINARY_FILTER_CRT        = 5,
  LUMINARY_FILTER_BLACKWHITE = 6
} typedef LuminaryFilter;

LUMINARY_API enum LuminaryToneMap {
  LUMINARY_TONEMAP_NONE       = 0,
  LUMINARY_TONEMAP_ACES       = 1,
  LUMINARY_TONEMAP_REINHARD   = 2,
  LUMINARY_TONEMAP_UNCHARTED2 = 3,
  LUMINARY_TONEMAP_AGX        = 4,
  LUMINARY_TONEMAP_AGX_PUNCHY = 5,
  LUMINARY_TONEMAP_AGX_CUSTOM = 6
} typedef LuminaryToneMap;

LUMINARY_API enum LuminaryApertureShape { LUMINARY_APERTURE_ROUND = 0, LUMINARY_APERTURE_BLADED = 1 } typedef LuminaryApertureShape;

LUMINARY_API struct LuminaryCamera {
  LuminaryVec3 pos;
  LuminaryVec3 rotation;
  float fov;
  float focal_length;
  float aperture_size;
  LuminaryApertureShape aperture_shape;
  int aperture_blade_count;
  float exposure;
  float max_exposure;
  float min_exposure;
  int auto_exposure;
  float far_clip_distance;
  LuminaryToneMap tonemap;
  float agx_custom_slope;
  float agx_custom_power;
  float agx_custom_saturation;
  LuminaryFilter filter;
  int bloom;
  float bloom_blend;
  int lens_flare;
  float lens_flare_threshold;
  int dithering;
  int purkinje;
  float purkinje_kappa1;
  float purkinje_kappa2;
  float wasd_speed;
  float mouse_speed;
  int smooth_movement;
  float smoothing_factor;
  float temporal_blend_factor;
  float russian_roulette_threshold;
  int use_color_correction;
  LuminaryRGBF color_correction;
  int do_firefly_clamping;
  float film_grain;
} typedef LuminaryCamera;

////////////////////////////////////////////////////////////////////
// Ocean
////////////////////////////////////////////////////////////////////

LUMINARY_API enum LuminaryJerlovWaterType {
  LUMINARY_JERLOV_WATER_TYPE_I   = 0,
  LUMINARY_JERLOV_WATER_TYPE_IA  = 1,
  LUMINARY_JERLOV_WATER_TYPE_IB  = 2,
  LUMINARY_JERLOV_WATER_TYPE_II  = 3,
  LUMINARY_JERLOV_WATER_TYPE_III = 4,
  LUMINARY_JERLOV_WATER_TYPE_1C  = 5,
  LUMINARY_JERLOV_WATER_TYPE_3C  = 6,
  LUMINARY_JERLOV_WATER_TYPE_5C  = 7,
  LUMINARY_JERLOV_WATER_TYPE_7C  = 8,
  LUMINARY_JERLOV_WATER_TYPE_9C  = 9
} typedef LuminaryJerlovWaterType;

LUMINARY_API struct LuminaryOcean {
  int active;
  float height;
  float amplitude;
  float frequency;
  float choppyness;
  float refractive_index;
  LuminaryJerlovWaterType water_type;
  int caustics_active;
  int caustics_ris_sample_count;
  float caustics_domain_scale;
  int multiscattering;
  int triangle_light_contribution;
} typedef LuminaryOcean;

////////////////////////////////////////////////////////////////////
// Material
////////////////////////////////////////////////////////////////////

LUMINARY_API struct LuminaryMaterialFlags {
  bool emission_active : 1;
  bool ior_shadowing : 1;
  bool thin_walled : 1;
  bool colored_transparency : 1;
} typedef LuminaryMaterialFlags;

LUMINARY_API struct LuminaryMaterial {
  LuminaryRGBAF albedo;
  LuminaryRGBF emission;
  float metallic;
  float roughness;
  float refraction_index;
  uint16_t albedo_tex;
  uint16_t luminance_tex;
  uint16_t material_tex;
  uint16_t normal_tex;
  LuminaryMaterialFlags flags;
} typedef LuminaryMaterial;

#endif /* LUMINARY_STRUCTS_H */
