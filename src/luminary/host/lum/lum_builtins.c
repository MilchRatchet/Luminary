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
  [LUM_BUILTIN_TYPE_FILE]             = "File",
  [LUM_BUILTIN_TYPE_VERSIONCONTROL]   = "VersionControl",
  [LUM_BUILTIN_TYPE_WAVEFRONTOBJFILE] = "WavefrontObjFile"};

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
  [LUM_BUILTIN_TYPE_STRING]           = sizeof(char*),
  [LUM_BUILTIN_TYPE_LUMINARY]         = 4,
  [LUM_BUILTIN_TYPE_FILE]             = 0,
  [LUM_BUILTIN_TYPE_VERSIONCONTROL]   = 0,
  [LUM_BUILTIN_TYPE_WAVEFRONTOBJFILE] = 0};

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
  [LUM_BUILTIN_TYPE_FILE]             = "file",
  [LUM_BUILTIN_TYPE_VERSIONCONTROL]   = "ver",
  [LUM_BUILTIN_TYPE_WAVEFRONTOBJFILE] = "obj"};

const bool lum_builtin_types_addressable[LUM_BUILTIN_TYPE_COUNT] = {
  [LUM_BUILTIN_TYPE_VOID] = false,    [LUM_BUILTIN_TYPE_RGBF] = false,          [LUM_BUILTIN_TYPE_VEC3] = false,
  [LUM_BUILTIN_TYPE_UINT] = false,    [LUM_BUILTIN_TYPE_BOOL] = false,          [LUM_BUILTIN_TYPE_FLOAT] = false,
  [LUM_BUILTIN_TYPE_ENUM] = false,    [LUM_BUILTIN_TYPE_SETTINGS] = false,      [LUM_BUILTIN_TYPE_CAMERA] = false,
  [LUM_BUILTIN_TYPE_OCEAN] = false,   [LUM_BUILTIN_TYPE_SKY] = false,           [LUM_BUILTIN_TYPE_CLOUD] = false,
  [LUM_BUILTIN_TYPE_FOG] = false,     [LUM_BUILTIN_TYPE_PARTICLES] = false,     [LUM_BUILTIN_TYPE_MATERIAL] = true,
  [LUM_BUILTIN_TYPE_INSTANCE] = true, [LUM_BUILTIN_TYPE_STRING] = false,        [LUM_BUILTIN_TYPE_LUMINARY] = false,
  [LUM_BUILTIN_TYPE_FILE] = true,     [LUM_BUILTIN_TYPE_VERSIONCONTROL] = true, [LUM_BUILTIN_TYPE_WAVEFRONTOBJFILE] = true};

#define __BUILTIN_ENUM_PAIR(__internal_macro_enum) {.string = #__internal_macro_enum, .value = __internal_macro_enum}

const LumBuiltinEnumValuePair lum_builtin_enums[] = {
  // LuminaryShadingMode
  __BUILTIN_ENUM_PAIR(LUMINARY_SHADING_MODE_DEFAULT), __BUILTIN_ENUM_PAIR(LUMINARY_SHADING_MODE_ALBEDO),
  __BUILTIN_ENUM_PAIR(LUMINARY_SHADING_MODE_DEPTH), __BUILTIN_ENUM_PAIR(LUMINARY_SHADING_MODE_NORMAL),
  __BUILTIN_ENUM_PAIR(LUMINARY_SHADING_MODE_IDENTIFICATION), __BUILTIN_ENUM_PAIR(LUMINARY_SHADING_MODE_LIGHTS),
  // LuminaryFilter
  __BUILTIN_ENUM_PAIR(LUMINARY_FILTER_NONE), __BUILTIN_ENUM_PAIR(LUMINARY_FILTER_GRAY), __BUILTIN_ENUM_PAIR(LUMINARY_FILTER_SEPIA),
  __BUILTIN_ENUM_PAIR(LUMINARY_FILTER_GAMEBOY), __BUILTIN_ENUM_PAIR(LUMINARY_FILTER_2BITGRAY), __BUILTIN_ENUM_PAIR(LUMINARY_FILTER_CRT),
  __BUILTIN_ENUM_PAIR(LUMINARY_FILTER_BLACKWHITE),
  // LuminaryTonemap
  __BUILTIN_ENUM_PAIR(LUMINARY_TONEMAP_NONE), __BUILTIN_ENUM_PAIR(LUMINARY_TONEMAP_ACES), __BUILTIN_ENUM_PAIR(LUMINARY_TONEMAP_REINHARD),
  __BUILTIN_ENUM_PAIR(LUMINARY_TONEMAP_UNCHARTED2), __BUILTIN_ENUM_PAIR(LUMINARY_TONEMAP_AGX),
  __BUILTIN_ENUM_PAIR(LUMINARY_TONEMAP_AGX_PUNCHY), __BUILTIN_ENUM_PAIR(LUMINARY_TONEMAP_AGX_CUSTOM),
  // LuminaryAperture
  __BUILTIN_ENUM_PAIR(LUMINARY_APERTURE_ROUND), __BUILTIN_ENUM_PAIR(LUMINARY_APERTURE_BLADED),
  // LuminaryJerlovWaterType
  __BUILTIN_ENUM_PAIR(LUMINARY_JERLOV_WATER_TYPE_I), __BUILTIN_ENUM_PAIR(LUMINARY_JERLOV_WATER_TYPE_IA),
  __BUILTIN_ENUM_PAIR(LUMINARY_JERLOV_WATER_TYPE_IB), __BUILTIN_ENUM_PAIR(LUMINARY_JERLOV_WATER_TYPE_II),
  __BUILTIN_ENUM_PAIR(LUMINARY_JERLOV_WATER_TYPE_III), __BUILTIN_ENUM_PAIR(LUMINARY_JERLOV_WATER_TYPE_1C),
  __BUILTIN_ENUM_PAIR(LUMINARY_JERLOV_WATER_TYPE_3C), __BUILTIN_ENUM_PAIR(LUMINARY_JERLOV_WATER_TYPE_5C),
  __BUILTIN_ENUM_PAIR(LUMINARY_JERLOV_WATER_TYPE_7C), __BUILTIN_ENUM_PAIR(LUMINARY_JERLOV_WATER_TYPE_9C),
  // LuminarySkyMode
  __BUILTIN_ENUM_PAIR(LUMINARY_SKY_MODE_DEFAULT), __BUILTIN_ENUM_PAIR(LUMINARY_SKY_MODE_HDRI),
  __BUILTIN_ENUM_PAIR(LUMINARY_SKY_MODE_CONSTANT_COLOR),
  // LuminaryMaterialBaseSubstrate
  __BUILTIN_ENUM_PAIR(LUMINARY_MATERIAL_BASE_SUBSTRATE_OPAQUE), __BUILTIN_ENUM_PAIR(LUMINARY_MATERIAL_BASE_SUBSTRATE_TRANSLUCENT)};
LUM_STATIC_SIZE_ASSERT(lum_builtin_enums, sizeof(LumBuiltinEnumValuePair) * LUM_BUILTIN_ENUM_COUNT);

static const LumBuiltinTypeMember _lum_builtin_member_vec3[] = {
  {.type = LUM_BUILTIN_TYPE_FLOAT, .offset = offsetof(LuminaryVec3, x), .name = "x"},
  {.type = LUM_BUILTIN_TYPE_FLOAT, .offset = offsetof(LuminaryVec3, y), .name = "y"},
  {.type = LUM_BUILTIN_TYPE_FLOAT, .offset = offsetof(LuminaryVec3, z), .name = "z"}};

static const LumBuiltinTypeMember _lum_builtin_member_luminary[] = {
  {.type = LUM_BUILTIN_TYPE_UINT, .offset = 0, .name = "compatibility_version"}};

const uint32_t lum_builtin_types_member_counts[LUM_BUILTIN_TYPE_COUNT] = {
  [LUM_BUILTIN_TYPE_VOID]             = 0,
  [LUM_BUILTIN_TYPE_RGBF]             = 0,
  [LUM_BUILTIN_TYPE_VEC3]             = sizeof(_lum_builtin_member_vec3) / sizeof(LumBuiltinTypeMember),
  [LUM_BUILTIN_TYPE_UINT]             = 0,
  [LUM_BUILTIN_TYPE_BOOL]             = 0,
  [LUM_BUILTIN_TYPE_FLOAT]            = 0,
  [LUM_BUILTIN_TYPE_ENUM]             = 0,
  [LUM_BUILTIN_TYPE_SETTINGS]         = 0,
  [LUM_BUILTIN_TYPE_CAMERA]           = 0,
  [LUM_BUILTIN_TYPE_OCEAN]            = 0,
  [LUM_BUILTIN_TYPE_SKY]              = 0,
  [LUM_BUILTIN_TYPE_CLOUD]            = 0,
  [LUM_BUILTIN_TYPE_FOG]              = 0,
  [LUM_BUILTIN_TYPE_PARTICLES]        = 0,
  [LUM_BUILTIN_TYPE_MATERIAL]         = 0,
  [LUM_BUILTIN_TYPE_INSTANCE]         = 0,
  [LUM_BUILTIN_TYPE_STRING]           = 0,
  [LUM_BUILTIN_TYPE_LUMINARY]         = sizeof(_lum_builtin_member_luminary) / sizeof(LumBuiltinTypeMember),
  [LUM_BUILTIN_TYPE_FILE]             = 0,
  [LUM_BUILTIN_TYPE_VERSIONCONTROL]   = 0,
  [LUM_BUILTIN_TYPE_WAVEFRONTOBJFILE] = 0};

const LumBuiltinTypeMember* lum_builtin_types_member[LUM_BUILTIN_TYPE_COUNT] = {
  [LUM_BUILTIN_TYPE_VOID]             = 0,
  [LUM_BUILTIN_TYPE_RGBF]             = 0,
  [LUM_BUILTIN_TYPE_VEC3]             = _lum_builtin_member_vec3,
  [LUM_BUILTIN_TYPE_UINT]             = 0,
  [LUM_BUILTIN_TYPE_BOOL]             = 0,
  [LUM_BUILTIN_TYPE_FLOAT]            = 0,
  [LUM_BUILTIN_TYPE_ENUM]             = 0,
  [LUM_BUILTIN_TYPE_SETTINGS]         = 0,
  [LUM_BUILTIN_TYPE_CAMERA]           = 0,
  [LUM_BUILTIN_TYPE_OCEAN]            = 0,
  [LUM_BUILTIN_TYPE_SKY]              = 0,
  [LUM_BUILTIN_TYPE_CLOUD]            = 0,
  [LUM_BUILTIN_TYPE_FOG]              = 0,
  [LUM_BUILTIN_TYPE_PARTICLES]        = 0,
  [LUM_BUILTIN_TYPE_MATERIAL]         = 0,
  [LUM_BUILTIN_TYPE_INSTANCE]         = 0,
  [LUM_BUILTIN_TYPE_STRING]           = 0,
  [LUM_BUILTIN_TYPE_LUMINARY]         = _lum_builtin_member_luminary,
  [LUM_BUILTIN_TYPE_FILE]             = 0,
  [LUM_BUILTIN_TYPE_VERSIONCONTROL]   = 0,
  [LUM_BUILTIN_TYPE_WAVEFRONTOBJFILE] = 0};
