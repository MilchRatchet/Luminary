#include "utils.h"

const char* const luminary_strings_shading_mode[LUMINARY_SHADING_MODE_COUNT] = {
  [LUMINARY_SHADING_MODE_DEFAULT]        = "None",
  [LUMINARY_SHADING_MODE_ALBEDO]         = "Albedo",
  [LUMINARY_SHADING_MODE_DEPTH]          = "Depth",
  [LUMINARY_SHADING_MODE_NORMAL]         = "Normal",
  [LUMINARY_SHADING_MODE_IDENTIFICATION] = "Identification",
  [LUMINARY_SHADING_MODE_LIGHTS]         = "Lights"};

const char* const luminary_strings_lens_model[LUMINARY_LENS_MODEL_COUNT] = {[LUMINARY_LENS_MODEL_THIN] = "Thin Lens"};

const char* const luminary_strings_filter[LUMINARY_FILTER_COUNT] = {
  [LUMINARY_FILTER_NONE]       = "None",
  [LUMINARY_FILTER_GRAY]       = "Gray",
  [LUMINARY_FILTER_SEPIA]      = "Sepia",
  [LUMINARY_FILTER_GAMEBOY]    = "Gameboy",
  [LUMINARY_FILTER_2BITGRAY]   = "2 Bit Gray",
  [LUMINARY_FILTER_CRT]        = "CRT",
  [LUMINARY_FILTER_BLACKWHITE] = "Black & White"};

const char* const luminary_strings_tonemap[LUMINARY_TONEMAP_COUNT] = {
  [LUMINARY_TONEMAP_NONE]       = "None",
  [LUMINARY_TONEMAP_ACES]       = "ACES",
  [LUMINARY_TONEMAP_REINHARD]   = "Reinhard",
  [LUMINARY_TONEMAP_UNCHARTED2] = "Uncharted 2",
  [LUMINARY_TONEMAP_AGX]        = "Agx",
  [LUMINARY_TONEMAP_AGX_PUNCHY] = "Agx Punchy",
  [LUMINARY_TONEMAP_AGX_CUSTOM] = "Agx Custom"};

const char* const luminary_strings_aperture[LUMINARY_APERTURE_COUNT] =
  {[LUMINARY_APERTURE_ROUND] = "Round", [LUMINARY_APERTURE_BLADED] = "Bladed"};

const char* const luminary_strings_jerlov_water_type[LUMINARY_JERLOV_WATER_TYPE_COUNT] = {
  [LUMINARY_JERLOV_WATER_TYPE_I] = "I",   [LUMINARY_JERLOV_WATER_TYPE_IA] = "IA",   [LUMINARY_JERLOV_WATER_TYPE_IB] = "IB",
  [LUMINARY_JERLOV_WATER_TYPE_II] = "II", [LUMINARY_JERLOV_WATER_TYPE_III] = "III", [LUMINARY_JERLOV_WATER_TYPE_1C] = "1C",
  [LUMINARY_JERLOV_WATER_TYPE_3C] = "3C", [LUMINARY_JERLOV_WATER_TYPE_5C] = "5C",   [LUMINARY_JERLOV_WATER_TYPE_7C] = "7C",
  [LUMINARY_JERLOV_WATER_TYPE_9C] = "9C"};

const char* const luminary_strings_sky_mode[LUMINARY_SKY_MODE_COUNT] =
  {[LUMINARY_SKY_MODE_DEFAULT] = "Default", [LUMINARY_SKY_MODE_HDRI] = "HDRI", [LUMINARY_SKY_MODE_CONSTANT_COLOR] = "Constant Color"};

const char* const luminary_strings_material_base_substrate[LUMINARY_MATERIAL_BASE_SUBSTRATE_COUNT] =
  {[LUMINARY_MATERIAL_BASE_SUBSTRATE_OPAQUE] = "Opaque", [LUMINARY_MATERIAL_BASE_SUBSTRATE_TRANSLUCENT] = "Translucent"};
