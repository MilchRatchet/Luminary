#ifndef LUMINARY_LUM_BUILTINS_H
#define LUMINARY_LUM_BUILTINS_H

#include "utils.h"

#define LUM_VERSION_CURRENT (1)

////////////////////////////////////////////////////////////////////
// LumBuiltin Type Meta Data
////////////////////////////////////////////////////////////////////

enum LumBuiltinType {
  // Version 1
  LUM_BUILTIN_TYPE_VOID,
  LUM_BUILTIN_TYPE_RGBF,
  LUM_BUILTIN_TYPE_VEC3,
  LUM_BUILTIN_TYPE_UINT,
  LUM_BUILTIN_TYPE_BOOL,
  LUM_BUILTIN_TYPE_FLOAT,
  LUM_BUILTIN_TYPE_ENUM,
  LUM_BUILTIN_TYPE_SETTINGS,
  LUM_BUILTIN_TYPE_CAMERA,
  LUM_BUILTIN_TYPE_OCEAN,
  LUM_BUILTIN_TYPE_SKY,
  LUM_BUILTIN_TYPE_CLOUD,
  LUM_BUILTIN_TYPE_FOG,
  LUM_BUILTIN_TYPE_PARTICLES,
  LUM_BUILTIN_TYPE_MATERIAL,
  LUM_BUILTIN_TYPE_INSTANCE,
  LUM_BUILTIN_TYPE_STRING,
  LUM_BUILTIN_TYPE_LUMINARY,
  LUM_BUILTIN_TYPE_WAVEFRONTOBJFILE,
  LUM_BUILTIN_TYPE_ADAPTIVESAMPLING,
  LUM_BUILTIN_TYPE_COUNT_VERSION_1,

  LUM_BUILTIN_TYPE_COUNT = LUM_BUILTIN_TYPE_COUNT_VERSION_1
} typedef LumBuiltinType;

extern const char* lum_builtin_types_strings[LUM_BUILTIN_TYPE_COUNT];
extern const size_t lum_builtin_types_sizes[LUM_BUILTIN_TYPE_COUNT];
extern const char* lum_builtin_types_mnemonic[LUM_BUILTIN_TYPE_COUNT];
extern const bool lum_builtin_types_addressable[LUM_BUILTIN_TYPE_COUNT];

////////////////////////////////////////////////////////////////////
// LumBuiltin Enums
////////////////////////////////////////////////////////////////////

#define LUM_BUILTIN_ENUM_COUNT                                                                            \
  (LUMINARY_SHADING_MODE_COUNT + LUMINARY_FILTER_COUNT + LUMINARY_TONEMAP_COUNT + LUMINARY_APERTURE_COUNT \
   + LUMINARY_JERLOV_WATER_TYPE_COUNT + LUMINARY_SKY_MODE_COUNT + LUMINARY_MATERIAL_BASE_SUBSTRATE_COUNT)

struct LumBuiltinEnumValuePair {
  const char* string;
  uint32_t value;
  uint32_t min_version;
  uint32_t max_version;
} typedef LumBuiltinEnumValuePair;

extern const LumBuiltinEnumValuePair lum_builtin_enums[LUM_BUILTIN_ENUM_COUNT];

enum LumBuiltinEnum {
  // LuminaryShadingMode
  LUM_BUILTIN_ENUM_SHADING_MODE_DEFAULT        = LUMINARY_SHADING_MODE_DEFAULT,
  LUM_BUILTIN_ENUM_SHADING_MODE_ALBEDO         = LUMINARY_SHADING_MODE_ALBEDO,
  LUM_BUILTIN_ENUM_SHADING_MODE_DEPTH          = LUMINARY_SHADING_MODE_DEPTH,
  LUM_BUILTIN_ENUM_SHADING_MODE_NORMAL         = LUMINARY_SHADING_MODE_NORMAL,
  LUM_BUILTIN_ENUM_SHADING_MODE_IDENTIFICATION = LUMINARY_SHADING_MODE_IDENTIFICATION,
  LUM_BUILTIN_ENUM_SHADING_MODE_LIGHTS         = LUMINARY_SHADING_MODE_LIGHTS,
  // LuminaryFilter
  LUM_BUILTIN_ENUM_FILTER_NONE       = LUMINARY_FILTER_NONE,
  LUM_BUILTIN_ENUM_FILTER_GRAY       = LUMINARY_FILTER_GRAY,
  LUM_BUILTIN_ENUM_FILTER_SEPIA      = LUMINARY_FILTER_SEPIA,
  LUM_BUILTIN_ENUM_FILTER_GAMEBOY    = LUMINARY_FILTER_GAMEBOY,
  LUM_BUILTIN_ENUM_FILTER_2BITGRAY   = LUMINARY_FILTER_2BITGRAY,
  LUM_BUILTIN_ENUM_FILTER_CRT        = LUMINARY_FILTER_CRT,
  LUM_BUILTIN_ENUM_FILTER_BLACKWHITE = LUMINARY_FILTER_BLACKWHITE,
  // LuminaryTonemap
  LUM_BUILTIN_ENUM_TONEMAP_NONE       = LUMINARY_TONEMAP_NONE,
  LUM_BUILTIN_ENUM_TONEMAP_ACES       = LUMINARY_TONEMAP_ACES,
  LUM_BUILTIN_ENUM_TONEMAP_REINHARD   = LUMINARY_TONEMAP_REINHARD,
  LUM_BUILTIN_ENUM_TONEMAP_UNCHARTED2 = LUMINARY_TONEMAP_UNCHARTED2,
  LUM_BUILTIN_ENUM_TONEMAP_AGX        = LUMINARY_TONEMAP_AGX,
  LUM_BUILTIN_ENUM_TONEMAP_AGX_PUNCHY = LUMINARY_TONEMAP_AGX_PUNCHY,
  LUM_BUILTIN_ENUM_TONEMAP_AGX_CUSTOM = LUMINARY_TONEMAP_AGX_CUSTOM,
  // LuminaryAperture
  LUM_BUILTIN_ENUM_APERTURE_ROUND  = LUMINARY_APERTURE_ROUND,
  LUM_BUILTIN_ENUM_APERTURE_BLADED = LUMINARY_APERTURE_BLADED,
  // LuminaryJerlovWaterType
  LUM_BUILTIN_ENUM_JERLOV_WATER_TYPE_I   = LUMINARY_JERLOV_WATER_TYPE_I,
  LUM_BUILTIN_ENUM_JERLOV_WATER_TYPE_IA  = LUMINARY_JERLOV_WATER_TYPE_IA,
  LUM_BUILTIN_ENUM_JERLOV_WATER_TYPE_IB  = LUMINARY_JERLOV_WATER_TYPE_IB,
  LUM_BUILTIN_ENUM_JERLOV_WATER_TYPE_II  = LUMINARY_JERLOV_WATER_TYPE_II,
  LUM_BUILTIN_ENUM_JERLOV_WATER_TYPE_III = LUMINARY_JERLOV_WATER_TYPE_III,
  LUM_BUILTIN_ENUM_JERLOV_WATER_TYPE_1C  = LUMINARY_JERLOV_WATER_TYPE_1C,
  LUM_BUILTIN_ENUM_JERLOV_WATER_TYPE_3C  = LUMINARY_JERLOV_WATER_TYPE_3C,
  LUM_BUILTIN_ENUM_JERLOV_WATER_TYPE_5C  = LUMINARY_JERLOV_WATER_TYPE_5C,
  LUM_BUILTIN_ENUM_JERLOV_WATER_TYPE_7C  = LUMINARY_JERLOV_WATER_TYPE_7C,
  LUM_BUILTIN_ENUM_JERLOV_WATER_TYPE_9C  = LUMINARY_JERLOV_WATER_TYPE_9C,
  // LuminarySkyMode
  LUM_BUILTIN_ENUM_SKY_MODE_DEFAULT        = LUMINARY_SKY_MODE_DEFAULT,
  LUM_BUILTIN_ENUM_SKY_MODE_HDRI           = LUMINARY_SKY_MODE_HDRI,
  LUM_BUILTIN_ENUM_SKY_MODE_CONSTANT_COLOR = LUMINARY_SKY_MODE_CONSTANT_COLOR,
  // LuminaryMaterialBaseSubstrate
  LUM_BUILTIN_ENUM_MATERIAL_BASE_SUBSTRATE_OPAQUE      = LUMINARY_MATERIAL_BASE_SUBSTRATE_OPAQUE,
  LUM_BUILTIN_ENUM_MATERIAL_BASE_SUBSTRATE_TRANSLUCENT = LUMINARY_MATERIAL_BASE_SUBSTRATE_TRANSLUCENT,
} typedef LumBuiltinEnum;

////////////////////////////////////////////////////////////////////
// LumBuiltin Members
////////////////////////////////////////////////////////////////////

struct LumBuiltinTypeMember {
  LumBuiltinType type;
  size_t offset;
  const char* name;
  uint32_t min_version;
  uint32_t max_version;
} typedef LumBuiltinTypeMember;

extern const uint32_t lum_builtin_types_member_counts[LUM_BUILTIN_TYPE_COUNT];
extern const LumBuiltinTypeMember* lum_builtin_types_member[LUM_BUILTIN_TYPE_COUNT];

////////////////////////////////////////////////////////////////////
// LumBuiltin Struct Definitions
////////////////////////////////////////////////////////////////////

struct LumBuiltinAdaptiveSampling {
  bool enable;
  uint32_t max_sampling_rate;
  uint32_t avg_sampling_rate;
  uint32_t update_interval;
  bool exposure_aware;
  LuminaryAdaptiveSamplingOutputMode output_mode;
} typedef LumBuiltinAdaptiveSampling;

struct LumBuiltinSettings {
  uint32_t width;
  uint32_t height;
  uint32_t max_ray_depth;
  uint32_t bridge_max_num_vertices;
  uint32_t undersampling;
  uint32_t supersampling;
  LumBuiltinAdaptiveSampling adaptive_sampling_settings;
  LuminaryShadingMode shading_mode;
  float region_x;
  float region_y;
  float region_width;
  float region_height;
} typedef LumBuiltinSettings;

struct LumBuiltinCamera {
  LuminaryVec3 pos;
} typedef LumBuiltinCamera;

struct LumBuiltinLuminary {
  uint32_t compatibility_version;
} typedef LumBuiltinLuminary;

#endif /* LUMINARY_LUM_BUILTINS_H */
