#include "lum_builtins.h"

#include <stddef.h>

#include "lum_tokenizer.h"

const char* lum_builtin_types_strings[LUM_BUILTIN_TYPE_COUNT] = {
  [LUM_BUILTIN_TYPE_VOID]             = "void",
  [LUM_BUILTIN_TYPE_RGBF]             = "RGBF",
  [LUM_BUILTIN_TYPE_VEC3]             = "vec3",
  [LUM_BUILTIN_TYPE_UINT]             = "uint",
  [LUM_BUILTIN_TYPE_BOOL]             = "bool",
  [LUM_BUILTIN_TYPE_FLOAT]            = "float",
  [LUM_BUILTIN_TYPE_ENUM]             = "Enum",
  [LUM_BUILTIN_TYPE_SETTINGS]         = "Settings",
  [LUM_BUILTIN_TYPE_CAMERA]           = "Camera",
  [LUM_BUILTIN_TYPE_OCEAN]            = "Ocean",
  [LUM_BUILTIN_TYPE_SKY]              = "Sky",
  [LUM_BUILTIN_TYPE_CLOUD]            = "Cloud",
  [LUM_BUILTIN_TYPE_FOG]              = "Fog",
  [LUM_BUILTIN_TYPE_PARTICLES]        = "Particles",
  [LUM_BUILTIN_TYPE_MATERIAL]         = "Material",
  [LUM_BUILTIN_TYPE_INSTANCE]         = "Instance",
  [LUM_BUILTIN_TYPE_STRING]           = "String",
  [LUM_BUILTIN_TYPE_LUMINARY]         = "Luminary",
  [LUM_BUILTIN_TYPE_WAVEFRONTOBJFILE] = "WavefrontObjFile",
  [LUM_BUILTIN_TYPE_ADAPTIVESAMPLING] = "AdaptiveSamplingSettings"};

const size_t lum_builtin_types_sizes[LUM_BUILTIN_TYPE_COUNT] = {
  [LUM_BUILTIN_TYPE_VOID]             = 0,
  [LUM_BUILTIN_TYPE_RGBF]             = sizeof(LuminaryRGBF),
  [LUM_BUILTIN_TYPE_VEC3]             = sizeof(LuminaryVec3),
  [LUM_BUILTIN_TYPE_UINT]             = sizeof(uint32_t),
  [LUM_BUILTIN_TYPE_BOOL]             = sizeof(bool),
  [LUM_BUILTIN_TYPE_FLOAT]            = sizeof(float),
  [LUM_BUILTIN_TYPE_ENUM]             = sizeof(uint32_t),
  [LUM_BUILTIN_TYPE_SETTINGS]         = sizeof(LuminaryRendererSettings),
  [LUM_BUILTIN_TYPE_CAMERA]           = sizeof(LuminaryCamera),
  [LUM_BUILTIN_TYPE_OCEAN]            = sizeof(LuminaryOcean),
  [LUM_BUILTIN_TYPE_SKY]              = sizeof(LuminarySky),
  [LUM_BUILTIN_TYPE_CLOUD]            = sizeof(LuminaryCloud),
  [LUM_BUILTIN_TYPE_FOG]              = sizeof(LuminaryFog),
  [LUM_BUILTIN_TYPE_PARTICLES]        = sizeof(LuminaryParticles),
  [LUM_BUILTIN_TYPE_MATERIAL]         = sizeof(LuminaryMaterial),
  [LUM_BUILTIN_TYPE_INSTANCE]         = sizeof(LuminaryInstance),
  [LUM_BUILTIN_TYPE_STRING]           = sizeof(uint32_t),
  [LUM_BUILTIN_TYPE_LUMINARY]         = 4,
  [LUM_BUILTIN_TYPE_WAVEFRONTOBJFILE] = 0,
  [LUM_BUILTIN_TYPE_ADAPTIVESAMPLING] = sizeof(LuminaryAdaptiveSamplingSettings)};

const char* lum_builtin_types_mnemonic[LUM_BUILTIN_TYPE_COUNT] = {
  [LUM_BUILTIN_TYPE_VOID]             = "",
  [LUM_BUILTIN_TYPE_RGBF]             = "f32x3",
  [LUM_BUILTIN_TYPE_VEC3]             = "f32x3",
  [LUM_BUILTIN_TYPE_UINT]             = "u32",
  [LUM_BUILTIN_TYPE_BOOL]             = "bool",
  [LUM_BUILTIN_TYPE_FLOAT]            = "f32",
  [LUM_BUILTIN_TYPE_ENUM]             = "u32",
  [LUM_BUILTIN_TYPE_SETTINGS]         = "set",
  [LUM_BUILTIN_TYPE_CAMERA]           = "cam",
  [LUM_BUILTIN_TYPE_OCEAN]            = "oce",
  [LUM_BUILTIN_TYPE_SKY]              = "sky",
  [LUM_BUILTIN_TYPE_CLOUD]            = "clo",
  [LUM_BUILTIN_TYPE_FOG]              = "fog",
  [LUM_BUILTIN_TYPE_PARTICLES]        = "par",
  [LUM_BUILTIN_TYPE_MATERIAL]         = "mat",
  [LUM_BUILTIN_TYPE_INSTANCE]         = "ins",
  [LUM_BUILTIN_TYPE_STRING]           = "str",
  [LUM_BUILTIN_TYPE_LUMINARY]         = "lum",
  [LUM_BUILTIN_TYPE_WAVEFRONTOBJFILE] = "obj",
  [LUM_BUILTIN_TYPE_ADAPTIVESAMPLING] = "asam"};

const bool lum_builtin_types_addressable[LUM_BUILTIN_TYPE_COUNT] = {
  [LUM_BUILTIN_TYPE_VOID]             = false,
  [LUM_BUILTIN_TYPE_RGBF]             = false,
  [LUM_BUILTIN_TYPE_VEC3]             = false,
  [LUM_BUILTIN_TYPE_UINT]             = false,
  [LUM_BUILTIN_TYPE_BOOL]             = false,
  [LUM_BUILTIN_TYPE_FLOAT]            = false,
  [LUM_BUILTIN_TYPE_ENUM]             = false,
  [LUM_BUILTIN_TYPE_SETTINGS]         = false,
  [LUM_BUILTIN_TYPE_CAMERA]           = false,
  [LUM_BUILTIN_TYPE_OCEAN]            = false,
  [LUM_BUILTIN_TYPE_SKY]              = false,
  [LUM_BUILTIN_TYPE_CLOUD]            = false,
  [LUM_BUILTIN_TYPE_FOG]              = false,
  [LUM_BUILTIN_TYPE_PARTICLES]        = false,
  [LUM_BUILTIN_TYPE_MATERIAL]         = true,
  [LUM_BUILTIN_TYPE_INSTANCE]         = true,
  [LUM_BUILTIN_TYPE_STRING]           = false,
  [LUM_BUILTIN_TYPE_LUMINARY]         = false,
  [LUM_BUILTIN_TYPE_WAVEFRONTOBJFILE] = true,
  [LUM_BUILTIN_TYPE_ADAPTIVESAMPLING] = false};

#define __BUILTIN_ENUM_PAIR(__internal_macro_enum, __macro_min_ver, __macro_max_ver) \
  {.string      = "LUMINARY_" #__internal_macro_enum,                                \
   .value       = LUM_BUILTIN_ENUM_##__internal_macro_enum,                          \
   .min_version = (__macro_min_ver),                                                 \
   .max_version = (__macro_max_ver)}

const LumBuiltinEnumValuePair lum_builtin_enums[] = {
  // LuminaryShadingMode
  __BUILTIN_ENUM_PAIR(SHADING_MODE_DEFAULT, 1, LUM_VERSION_CURRENT), __BUILTIN_ENUM_PAIR(SHADING_MODE_ALBEDO, 1, LUM_VERSION_CURRENT),
  __BUILTIN_ENUM_PAIR(SHADING_MODE_DEPTH, 1, LUM_VERSION_CURRENT), __BUILTIN_ENUM_PAIR(SHADING_MODE_NORMAL, 1, LUM_VERSION_CURRENT),
  __BUILTIN_ENUM_PAIR(SHADING_MODE_IDENTIFICATION, 1, LUM_VERSION_CURRENT),
  __BUILTIN_ENUM_PAIR(SHADING_MODE_LIGHTS, 1, LUM_VERSION_CURRENT),
  // LuminaryFilter
  __BUILTIN_ENUM_PAIR(FILTER_NONE, 1, LUM_VERSION_CURRENT), __BUILTIN_ENUM_PAIR(FILTER_GRAY, 1, LUM_VERSION_CURRENT),
  __BUILTIN_ENUM_PAIR(FILTER_SEPIA, 1, LUM_VERSION_CURRENT), __BUILTIN_ENUM_PAIR(FILTER_GAMEBOY, 1, LUM_VERSION_CURRENT),
  __BUILTIN_ENUM_PAIR(FILTER_2BITGRAY, 1, LUM_VERSION_CURRENT), __BUILTIN_ENUM_PAIR(FILTER_CRT, 1, LUM_VERSION_CURRENT),
  __BUILTIN_ENUM_PAIR(FILTER_BLACKWHITE, 1, LUM_VERSION_CURRENT),
  // LuminaryTonemap
  __BUILTIN_ENUM_PAIR(TONEMAP_NONE, 1, LUM_VERSION_CURRENT), __BUILTIN_ENUM_PAIR(TONEMAP_ACES, 1, LUM_VERSION_CURRENT),
  __BUILTIN_ENUM_PAIR(TONEMAP_REINHARD, 1, LUM_VERSION_CURRENT), __BUILTIN_ENUM_PAIR(TONEMAP_UNCHARTED2, 1, LUM_VERSION_CURRENT),
  __BUILTIN_ENUM_PAIR(TONEMAP_AGX, 1, LUM_VERSION_CURRENT), __BUILTIN_ENUM_PAIR(TONEMAP_AGX_PUNCHY, 1, LUM_VERSION_CURRENT),
  __BUILTIN_ENUM_PAIR(TONEMAP_AGX_CUSTOM, 1, LUM_VERSION_CURRENT),
  // LuminaryAperture
  __BUILTIN_ENUM_PAIR(APERTURE_ROUND, 1, LUM_VERSION_CURRENT), __BUILTIN_ENUM_PAIR(APERTURE_BLADED, 1, LUM_VERSION_CURRENT),
  // LuminaryJerlovWaterType
  __BUILTIN_ENUM_PAIR(JERLOV_WATER_TYPE_I, 1, LUM_VERSION_CURRENT), __BUILTIN_ENUM_PAIR(JERLOV_WATER_TYPE_IA, 1, LUM_VERSION_CURRENT),
  __BUILTIN_ENUM_PAIR(JERLOV_WATER_TYPE_IB, 1, LUM_VERSION_CURRENT), __BUILTIN_ENUM_PAIR(JERLOV_WATER_TYPE_II, 1, LUM_VERSION_CURRENT),
  __BUILTIN_ENUM_PAIR(JERLOV_WATER_TYPE_III, 1, LUM_VERSION_CURRENT), __BUILTIN_ENUM_PAIR(JERLOV_WATER_TYPE_1C, 1, LUM_VERSION_CURRENT),
  __BUILTIN_ENUM_PAIR(JERLOV_WATER_TYPE_3C, 1, LUM_VERSION_CURRENT), __BUILTIN_ENUM_PAIR(JERLOV_WATER_TYPE_5C, 1, LUM_VERSION_CURRENT),
  __BUILTIN_ENUM_PAIR(JERLOV_WATER_TYPE_7C, 1, LUM_VERSION_CURRENT), __BUILTIN_ENUM_PAIR(JERLOV_WATER_TYPE_9C, 1, LUM_VERSION_CURRENT),
  // LuminarySkyMode
  __BUILTIN_ENUM_PAIR(SKY_MODE_DEFAULT, 1, LUM_VERSION_CURRENT), __BUILTIN_ENUM_PAIR(SKY_MODE_HDRI, 1, LUM_VERSION_CURRENT),
  __BUILTIN_ENUM_PAIR(SKY_MODE_CONSTANT_COLOR, 1, LUM_VERSION_CURRENT),
  // LuminaryMaterialBaseSubstrate
  __BUILTIN_ENUM_PAIR(MATERIAL_BASE_SUBSTRATE_OPAQUE, 1, LUM_VERSION_CURRENT),
  __BUILTIN_ENUM_PAIR(MATERIAL_BASE_SUBSTRATE_TRANSLUCENT, 1, LUM_VERSION_CURRENT)};
LUM_STATIC_SIZE_ASSERT(lum_builtin_enums, sizeof(LumBuiltinEnumValuePair) * LUM_BUILTIN_ENUM_COUNT);

static const LumBuiltinTypeMember _lum_builtin_member_rgbf[] = {
  {.type = LUM_BUILTIN_TYPE_FLOAT, .offset = offsetof(LuminaryRGBF, r), .name = "r", .min_version = 1, .max_version = LUM_VERSION_CURRENT},
  {.type = LUM_BUILTIN_TYPE_FLOAT, .offset = offsetof(LuminaryRGBF, g), .name = "g", .min_version = 1, .max_version = LUM_VERSION_CURRENT},
  {.type = LUM_BUILTIN_TYPE_FLOAT, .offset = offsetof(LuminaryRGBF, b), .name = "b", .min_version = 1, .max_version = LUM_VERSION_CURRENT}};

static const LumBuiltinTypeMember _lum_builtin_member_vec3[] = {
  {.type = LUM_BUILTIN_TYPE_FLOAT, .offset = offsetof(LuminaryVec3, x), .name = "x", .min_version = 1, .max_version = LUM_VERSION_CURRENT},
  {.type = LUM_BUILTIN_TYPE_FLOAT, .offset = offsetof(LuminaryVec3, y), .name = "y", .min_version = 1, .max_version = LUM_VERSION_CURRENT},
  {.type = LUM_BUILTIN_TYPE_FLOAT, .offset = offsetof(LuminaryVec3, z), .name = "z", .min_version = 1, .max_version = LUM_VERSION_CURRENT}};

#define _LUM_BUILTIN_C_TYPE_TO_BUILTIN_TYPE(__macro_base_struct, __macro_member_name) \
  _Generic(                                                                           \
    ((__macro_base_struct*) 0)->__macro_member_name,                                  \
    LuminaryRGBF: LUM_BUILTIN_TYPE_RGBF,                                              \
    LuminaryVec3: LUM_BUILTIN_TYPE_VEC3,                                              \
    uint32_t: LUM_BUILTIN_TYPE_UINT,                                                  \
    bool: LUM_BUILTIN_TYPE_BOOL,                                                      \
    float: LUM_BUILTIN_TYPE_FLOAT,                                                    \
    int32_t: LUM_BUILTIN_TYPE_ENUM,                                                   \
    LumBuiltinAdaptiveSampling: LUM_BUILTIN_TYPE_ADAPTIVESAMPLING)

#define _LUM_BUILTIN_MEMBER(__macro_base_struct, __macro_member_name, __macro_min_ver, __macro_max_ver) \
  {.type        = _LUM_BUILTIN_C_TYPE_TO_BUILTIN_TYPE(__macro_base_struct, __macro_member_name),        \
   .offset      = offsetof(__macro_base_struct, __macro_member_name),                                   \
   .name        = #__macro_member_name,                                                                 \
   .min_version = (__macro_min_ver),                                                                    \
   .max_version = (__macro_max_ver)}

static const LumBuiltinTypeMember _lum_builtin_member_settings[] = {
  _LUM_BUILTIN_MEMBER(LumBuiltinSettings, width, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinSettings, height, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinSettings, max_ray_depth, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinSettings, bridge_max_num_vertices, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinSettings, undersampling, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinSettings, supersampling, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinSettings, adaptive_sampling_settings, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinSettings, shading_mode, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinSettings, region_x, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinSettings, region_y, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinSettings, region_width, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinSettings, region_height, 1, LUM_VERSION_CURRENT),
};

static const LumBuiltinTypeMember _lum_builtin_member_camera[] = {
  _LUM_BUILTIN_MEMBER(LumBuiltinCamera, pos, 1, LUM_VERSION_CURRENT),
};

static const LumBuiltinTypeMember _lum_builtin_member_luminary[] = {
  _LUM_BUILTIN_MEMBER(LumBuiltinLuminary, compatibility_version, 1, LUM_VERSION_CURRENT),
};

static const LumBuiltinTypeMember _lum_builtin_member_adaptive_sampling[] = {
  _LUM_BUILTIN_MEMBER(LumBuiltinAdaptiveSampling, enable, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinAdaptiveSampling, max_sampling_rate, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinAdaptiveSampling, avg_sampling_rate, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinAdaptiveSampling, update_interval, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinAdaptiveSampling, exposure_aware, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinAdaptiveSampling, output_mode, 1, LUM_VERSION_CURRENT),
};

const uint32_t lum_builtin_types_member_counts[LUM_BUILTIN_TYPE_COUNT] = {
  [LUM_BUILTIN_TYPE_VOID]             = 0,
  [LUM_BUILTIN_TYPE_RGBF]             = sizeof(_lum_builtin_member_rgbf) / sizeof(LumBuiltinTypeMember),
  [LUM_BUILTIN_TYPE_VEC3]             = sizeof(_lum_builtin_member_vec3) / sizeof(LumBuiltinTypeMember),
  [LUM_BUILTIN_TYPE_UINT]             = 0,
  [LUM_BUILTIN_TYPE_BOOL]             = 0,
  [LUM_BUILTIN_TYPE_FLOAT]            = 0,
  [LUM_BUILTIN_TYPE_ENUM]             = 0,
  [LUM_BUILTIN_TYPE_SETTINGS]         = sizeof(_lum_builtin_member_settings) / sizeof(LumBuiltinTypeMember),
  [LUM_BUILTIN_TYPE_CAMERA]           = sizeof(_lum_builtin_member_camera) / sizeof(LumBuiltinTypeMember),
  [LUM_BUILTIN_TYPE_OCEAN]            = 0,
  [LUM_BUILTIN_TYPE_SKY]              = 0,
  [LUM_BUILTIN_TYPE_CLOUD]            = 0,
  [LUM_BUILTIN_TYPE_FOG]              = 0,
  [LUM_BUILTIN_TYPE_PARTICLES]        = 0,
  [LUM_BUILTIN_TYPE_MATERIAL]         = 0,
  [LUM_BUILTIN_TYPE_INSTANCE]         = 0,
  [LUM_BUILTIN_TYPE_STRING]           = 0,
  [LUM_BUILTIN_TYPE_LUMINARY]         = sizeof(_lum_builtin_member_luminary) / sizeof(LumBuiltinTypeMember),
  [LUM_BUILTIN_TYPE_WAVEFRONTOBJFILE] = 0,
  [LUM_BUILTIN_TYPE_ADAPTIVESAMPLING] = sizeof(_lum_builtin_member_adaptive_sampling) / sizeof(LumBuiltinTypeMember)};

const LumBuiltinTypeMember* lum_builtin_types_member[LUM_BUILTIN_TYPE_COUNT] = {
  [LUM_BUILTIN_TYPE_VOID]             = 0,
  [LUM_BUILTIN_TYPE_RGBF]             = _lum_builtin_member_rgbf,
  [LUM_BUILTIN_TYPE_VEC3]             = _lum_builtin_member_vec3,
  [LUM_BUILTIN_TYPE_UINT]             = 0,
  [LUM_BUILTIN_TYPE_BOOL]             = 0,
  [LUM_BUILTIN_TYPE_FLOAT]            = 0,
  [LUM_BUILTIN_TYPE_ENUM]             = 0,
  [LUM_BUILTIN_TYPE_SETTINGS]         = _lum_builtin_member_settings,
  [LUM_BUILTIN_TYPE_CAMERA]           = _lum_builtin_member_camera,
  [LUM_BUILTIN_TYPE_OCEAN]            = 0,
  [LUM_BUILTIN_TYPE_SKY]              = 0,
  [LUM_BUILTIN_TYPE_CLOUD]            = 0,
  [LUM_BUILTIN_TYPE_FOG]              = 0,
  [LUM_BUILTIN_TYPE_PARTICLES]        = 0,
  [LUM_BUILTIN_TYPE_MATERIAL]         = 0,
  [LUM_BUILTIN_TYPE_INSTANCE]         = 0,
  [LUM_BUILTIN_TYPE_STRING]           = 0,
  [LUM_BUILTIN_TYPE_LUMINARY]         = _lum_builtin_member_luminary,
  [LUM_BUILTIN_TYPE_WAVEFRONTOBJFILE] = 0,
  [LUM_BUILTIN_TYPE_ADAPTIVESAMPLING] = _lum_builtin_member_adaptive_sampling};
