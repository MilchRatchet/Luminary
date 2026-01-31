#include "lum_builtins.h"

#include <stddef.h>
#include <string.h>

#include "internal_error.h"
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
  [LUM_BUILTIN_TYPE_ADAPTIVESAMPLING] = "AdaptiveSamplingSettings",
  [LUM_BUILTIN_TYPE_CLOUDLAYER]       = "CloudLayer",
};

const size_t lum_builtin_types_sizes[LUM_BUILTIN_TYPE_COUNT] = {
  [LUM_BUILTIN_TYPE_VOID]             = 0,
  [LUM_BUILTIN_TYPE_RGBF]             = sizeof(LuminaryRGBF),
  [LUM_BUILTIN_TYPE_VEC3]             = sizeof(LuminaryVec3),
  [LUM_BUILTIN_TYPE_UINT]             = sizeof(uint32_t),
  [LUM_BUILTIN_TYPE_BOOL]             = sizeof(bool),
  [LUM_BUILTIN_TYPE_FLOAT]            = sizeof(float),
  [LUM_BUILTIN_TYPE_ENUM]             = sizeof(uint32_t),
  [LUM_BUILTIN_TYPE_SETTINGS]         = sizeof(LumBuiltinSettings),
  [LUM_BUILTIN_TYPE_CAMERA]           = sizeof(LumBuiltinCamera),
  [LUM_BUILTIN_TYPE_OCEAN]            = sizeof(LumBuiltinOcean),
  [LUM_BUILTIN_TYPE_SKY]              = sizeof(LumBuiltinSky),
  [LUM_BUILTIN_TYPE_CLOUD]            = sizeof(LumBuiltinCloud),
  [LUM_BUILTIN_TYPE_FOG]              = sizeof(LumBuiltinFog),
  [LUM_BUILTIN_TYPE_PARTICLES]        = sizeof(LumBuiltinParticles),
  [LUM_BUILTIN_TYPE_MATERIAL]         = sizeof(LumBuiltinMaterial),
  [LUM_BUILTIN_TYPE_INSTANCE]         = sizeof(LumBuiltinMaterial),
  [LUM_BUILTIN_TYPE_STRING]           = sizeof(uint32_t),
  [LUM_BUILTIN_TYPE_LUMINARY]         = sizeof(LumBuiltinLuminary),
  [LUM_BUILTIN_TYPE_WAVEFRONTOBJFILE] = 0,
  [LUM_BUILTIN_TYPE_ADAPTIVESAMPLING] = sizeof(LumBuiltinAdaptiveSampling),
  [LUM_BUILTIN_TYPE_CLOUDLAYER]       = sizeof(LumBuiltinCloudLayer)};

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
  [LUM_BUILTIN_TYPE_ADAPTIVESAMPLING] = "asam",
  [LUM_BUILTIN_TYPE_CLOUDLAYER]       = "clol"};

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
  [LUM_BUILTIN_TYPE_ADAPTIVESAMPLING] = false,
  [LUM_BUILTIN_TYPE_CLOUDLAYER]       = false,
};

#define __BUILTIN_ENUM_PAIR(__internal_macro_enum, __macro_min_ver, __macro_max_ver) \
  {.string      = "LUMINARY_" #__internal_macro_enum,                                \
   .value       = LUM_BUILTIN_ENUM_##__internal_macro_enum,                          \
   .min_version = (__macro_min_ver),                                                 \
   .max_version = (__macro_max_ver)}

const LumBuiltinEnumValuePair lum_builtin_enums[] = {
  // LuminaryShadingMode
  __BUILTIN_ENUM_PAIR(SHADING_MODE_DEFAULT, 1, LUM_VERSION_CURRENT),
  __BUILTIN_ENUM_PAIR(SHADING_MODE_ALBEDO, 1, LUM_VERSION_CURRENT),
  __BUILTIN_ENUM_PAIR(SHADING_MODE_DEPTH, 1, LUM_VERSION_CURRENT),
  __BUILTIN_ENUM_PAIR(SHADING_MODE_NORMAL, 1, LUM_VERSION_CURRENT),
  __BUILTIN_ENUM_PAIR(SHADING_MODE_IDENTIFICATION, 1, LUM_VERSION_CURRENT),
  __BUILTIN_ENUM_PAIR(SHADING_MODE_LIGHTS, 1, LUM_VERSION_CURRENT),
  // LuminaryFilter
  __BUILTIN_ENUM_PAIR(FILTER_NONE, 1, LUM_VERSION_CURRENT),
  __BUILTIN_ENUM_PAIR(FILTER_GRAY, 1, LUM_VERSION_CURRENT),
  __BUILTIN_ENUM_PAIR(FILTER_SEPIA, 1, LUM_VERSION_CURRENT),
  __BUILTIN_ENUM_PAIR(FILTER_GAMEBOY, 1, LUM_VERSION_CURRENT),
  __BUILTIN_ENUM_PAIR(FILTER_2BITGRAY, 1, LUM_VERSION_CURRENT),
  __BUILTIN_ENUM_PAIR(FILTER_CRT, 1, LUM_VERSION_CURRENT),
  __BUILTIN_ENUM_PAIR(FILTER_BLACKWHITE, 1, LUM_VERSION_CURRENT),
  // LuminaryTonemap
  __BUILTIN_ENUM_PAIR(TONEMAP_NONE, 1, LUM_VERSION_CURRENT),
  __BUILTIN_ENUM_PAIR(TONEMAP_ACES, 1, LUM_VERSION_CURRENT),
  __BUILTIN_ENUM_PAIR(TONEMAP_REINHARD, 1, LUM_VERSION_CURRENT),
  __BUILTIN_ENUM_PAIR(TONEMAP_UNCHARTED2, 1, LUM_VERSION_CURRENT),
  __BUILTIN_ENUM_PAIR(TONEMAP_AGX, 1, LUM_VERSION_CURRENT),
  __BUILTIN_ENUM_PAIR(TONEMAP_AGX_PUNCHY, 1, LUM_VERSION_CURRENT),
  __BUILTIN_ENUM_PAIR(TONEMAP_AGX_CUSTOM, 1, LUM_VERSION_CURRENT),
  // LuminaryAperture
  __BUILTIN_ENUM_PAIR(APERTURE_ROUND, 1, LUM_VERSION_CURRENT),
  __BUILTIN_ENUM_PAIR(APERTURE_BLADED, 1, LUM_VERSION_CURRENT),
  // LuminaryJerlovWaterType
  __BUILTIN_ENUM_PAIR(JERLOV_WATER_TYPE_I, 1, LUM_VERSION_CURRENT),
  __BUILTIN_ENUM_PAIR(JERLOV_WATER_TYPE_IA, 1, LUM_VERSION_CURRENT),
  __BUILTIN_ENUM_PAIR(JERLOV_WATER_TYPE_IB, 1, LUM_VERSION_CURRENT),
  __BUILTIN_ENUM_PAIR(JERLOV_WATER_TYPE_II, 1, LUM_VERSION_CURRENT),
  __BUILTIN_ENUM_PAIR(JERLOV_WATER_TYPE_III, 1, LUM_VERSION_CURRENT),
  __BUILTIN_ENUM_PAIR(JERLOV_WATER_TYPE_1C, 1, LUM_VERSION_CURRENT),
  __BUILTIN_ENUM_PAIR(JERLOV_WATER_TYPE_3C, 1, LUM_VERSION_CURRENT),
  __BUILTIN_ENUM_PAIR(JERLOV_WATER_TYPE_5C, 1, LUM_VERSION_CURRENT),
  __BUILTIN_ENUM_PAIR(JERLOV_WATER_TYPE_7C, 1, LUM_VERSION_CURRENT),
  __BUILTIN_ENUM_PAIR(JERLOV_WATER_TYPE_9C, 1, LUM_VERSION_CURRENT),
  // LuminarySkyMode
  __BUILTIN_ENUM_PAIR(SKY_MODE_DEFAULT, 1, LUM_VERSION_CURRENT),
  __BUILTIN_ENUM_PAIR(SKY_MODE_HDRI, 1, LUM_VERSION_CURRENT),
  __BUILTIN_ENUM_PAIR(SKY_MODE_CONSTANT_COLOR, 1, LUM_VERSION_CURRENT),
  // LuminaryMaterialBaseSubstrate
  __BUILTIN_ENUM_PAIR(MATERIAL_BASE_SUBSTRATE_OPAQUE, 1, LUM_VERSION_CURRENT),
  __BUILTIN_ENUM_PAIR(MATERIAL_BASE_SUBSTRATE_TRANSLUCENT, 1, LUM_VERSION_CURRENT),
};
LUM_STATIC_SIZE_ASSERT(lum_builtin_enums, sizeof(LumBuiltinEnumValuePair) * LUM_BUILTIN_ENUM_COUNT);

#define _LUM_BUILTIN_C_TYPE_TO_BUILTIN_TYPE(__macro_base_struct, __macro_member_name) \
  _Generic(                                                                           \
    ((__macro_base_struct*) 0)->__macro_member_name,                                  \
    LuminaryRGBF: LUM_BUILTIN_TYPE_RGBF,                                              \
    LuminaryVec3: LUM_BUILTIN_TYPE_VEC3,                                              \
    uint32_t: LUM_BUILTIN_TYPE_UINT,                                                  \
    bool: LUM_BUILTIN_TYPE_BOOL,                                                      \
    float: LUM_BUILTIN_TYPE_FLOAT,                                                    \
    int32_t: LUM_BUILTIN_TYPE_ENUM,                                                   \
    LumBuiltinAdaptiveSampling: LUM_BUILTIN_TYPE_ADAPTIVESAMPLING,                    \
    LumBuiltinCloudLayer: LUM_BUILTIN_TYPE_CLOUDLAYER)

#define _LUM_BUILTIN_MEMBER(__macro_base_struct, __macro_member_name, __macro_min_ver, __macro_max_ver) \
  {.type        = _LUM_BUILTIN_C_TYPE_TO_BUILTIN_TYPE(__macro_base_struct, __macro_member_name),        \
   .offset      = offsetof(__macro_base_struct, __macro_member_name),                                   \
   .name        = #__macro_member_name,                                                                 \
   .min_version = (__macro_min_ver),                                                                    \
   .max_version = (__macro_max_ver)}

static const LumBuiltinTypeMember _lum_builtin_member_rgbf[] = {
  _LUM_BUILTIN_MEMBER(LuminaryRGBF, r, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LuminaryRGBF, g, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LuminaryRGBF, b, 1, LUM_VERSION_CURRENT),
};

static const LumBuiltinTypeMember _lum_builtin_member_vec3[] = {
  _LUM_BUILTIN_MEMBER(LuminaryVec3, x, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LuminaryVec3, y, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LuminaryVec3, z, 1, LUM_VERSION_CURRENT),
};

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
  _LUM_BUILTIN_MEMBER(LumBuiltinCamera, rotation, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinCamera, aperture_shape, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinCamera, aperture_blade_count, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinCamera, exposure, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinCamera, tonemap, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinCamera, agx_custom_slope, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinCamera, agx_custom_power, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinCamera, agx_custom_saturation, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinCamera, filter, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinCamera, use_local_error_minimization, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinCamera, bloom_blend, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinCamera, dithering, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinCamera, purkinje, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinCamera, purkinje_kappa1, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinCamera, purkinje_kappa2, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinCamera, russian_roulette_threshold, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinCamera, use_color_correction, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinCamera, color_correction, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinCamera, film_grain, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinCamera, camera_scale, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinCamera, object_distance, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinCamera, use_physical_camera, 1, LUM_VERSION_CURRENT),
};

static const LumBuiltinTypeMember _lum_builtin_member_ocean[] = {
  _LUM_BUILTIN_MEMBER(LumBuiltinOcean, active, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinOcean, height, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinOcean, amplitude, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinOcean, frequency, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinOcean, refractive_index, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinOcean, water_type, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinOcean, caustics_active, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinOcean, caustics_ris_sample_count, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinOcean, caustics_domain_scale, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinOcean, multiscattering, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinOcean, triangle_light_contribution, 1, LUM_VERSION_CURRENT),
};

static const LumBuiltinTypeMember _lum_builtin_member_sky[] = {
  _LUM_BUILTIN_MEMBER(LumBuiltinSky, geometry_offset, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinSky, azimuth, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinSky, altitude, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinSky, moon_azimuth, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinSky, moon_altitude, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinSky, sun_strength, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinSky, base_density, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinSky, ozone_absorption, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinSky, steps, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinSky, stars_count, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinSky, stars_seed, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinSky, stars_intensity, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinSky, rayleigh_density, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinSky, mie_density, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinSky, ozone_density, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinSky, rayleigh_falloff, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinSky, mie_falloff, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinSky, mie_diameter, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinSky, ground_visibility, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinSky, ozone_layer_thickness, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinSky, multiscattering_factor, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinSky, hdri_dim, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinSky, hdri_samples, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinSky, aerial_perspective, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinSky, constant_color, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinSky, mode, 1, LUM_VERSION_CURRENT),
};

static const LumBuiltinTypeMember _lum_builtin_member_cloud[] = {
  _LUM_BUILTIN_MEMBER(LumBuiltinCloud, active, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinCloud, atmosphere_scattering, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinCloud, low, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinCloud, mid, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinCloud, top, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinCloud, offset_x, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinCloud, offset_z, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinCloud, density, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinCloud, seed, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinCloud, droplet_diameter, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinCloud, steps, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinCloud, shadow_steps, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinCloud, noise_shape_scale, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinCloud, noise_detail_scale, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinCloud, noise_weather_scale, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinCloud, mipmap_bias, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinCloud, octaves, 1, LUM_VERSION_CURRENT),
};

static const LumBuiltinTypeMember _lum_builtin_member_fog[] = {
  _LUM_BUILTIN_MEMBER(LumBuiltinFog, active, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinFog, density, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinFog, droplet_diameter, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinFog, height, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinFog, dist, 1, LUM_VERSION_CURRENT),
};

static const LumBuiltinTypeMember _lum_builtin_member_particles[] = {
  _LUM_BUILTIN_MEMBER(LumBuiltinParticles, active, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinParticles, seed, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinParticles, count, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinParticles, albedo, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinParticles, speed, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinParticles, direction_altitude, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinParticles, direction_azimuth, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinParticles, phase_diameter, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinParticles, scale, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinParticles, size, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinParticles, size_variation, 1, LUM_VERSION_CURRENT),
};

static const LumBuiltinTypeMember _lum_builtin_member_material[] = {
  _LUM_BUILTIN_MEMBER(LumBuiltinMaterial, base_substrate, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinMaterial, albedo, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinMaterial, opacity, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinMaterial, emission, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinMaterial, emission_scale, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinMaterial, roughness, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinMaterial, roughness_clamp, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinMaterial, refraction_index, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinMaterial, emission_active, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinMaterial, thin_walled, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinMaterial, metallic, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinMaterial, colored_transparency, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinMaterial, roughness_as_smoothness, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinMaterial, normal_map_is_compressed, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinMaterial, bidirectional_emission, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinMaterial, albedo_tex, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinMaterial, luminance_tex, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinMaterial, roughness_tex, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinMaterial, metallic_tex, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinMaterial, normal_tex, 1, LUM_VERSION_CURRENT),
};

static const LumBuiltinTypeMember _lum_builtin_member_instance[] = {
  _LUM_BUILTIN_MEMBER(LumBuiltinInstance, mesh_id, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinInstance, position, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinInstance, rotation, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinInstance, scale, 1, LUM_VERSION_CURRENT),
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

static const LumBuiltinTypeMember _lum_builtin_member_cloud_layer[] = {
  _LUM_BUILTIN_MEMBER(LumBuiltinCloudLayer, active, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinCloudLayer, height_max, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinCloudLayer, height_min, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinCloudLayer, coverage, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinCloudLayer, coverage_min, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinCloudLayer, type, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinCloudLayer, type_min, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinCloudLayer, wind_speed, 1, LUM_VERSION_CURRENT),
  _LUM_BUILTIN_MEMBER(LumBuiltinCloudLayer, wind_angle, 1, LUM_VERSION_CURRENT),
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
  [LUM_BUILTIN_TYPE_OCEAN]            = sizeof(_lum_builtin_member_ocean) / sizeof(LumBuiltinTypeMember),
  [LUM_BUILTIN_TYPE_SKY]              = sizeof(_lum_builtin_member_sky) / sizeof(LumBuiltinTypeMember),
  [LUM_BUILTIN_TYPE_CLOUD]            = sizeof(_lum_builtin_member_cloud) / sizeof(LumBuiltinTypeMember),
  [LUM_BUILTIN_TYPE_FOG]              = sizeof(_lum_builtin_member_fog) / sizeof(LumBuiltinTypeMember),
  [LUM_BUILTIN_TYPE_PARTICLES]        = sizeof(_lum_builtin_member_particles) / sizeof(LumBuiltinTypeMember),
  [LUM_BUILTIN_TYPE_MATERIAL]         = sizeof(_lum_builtin_member_material) / sizeof(LumBuiltinTypeMember),
  [LUM_BUILTIN_TYPE_INSTANCE]         = sizeof(_lum_builtin_member_instance) / sizeof(LumBuiltinTypeMember),
  [LUM_BUILTIN_TYPE_STRING]           = 0,
  [LUM_BUILTIN_TYPE_LUMINARY]         = sizeof(_lum_builtin_member_luminary) / sizeof(LumBuiltinTypeMember),
  [LUM_BUILTIN_TYPE_WAVEFRONTOBJFILE] = 0,
  [LUM_BUILTIN_TYPE_ADAPTIVESAMPLING] = sizeof(_lum_builtin_member_adaptive_sampling) / sizeof(LumBuiltinTypeMember),
  [LUM_BUILTIN_TYPE_CLOUDLAYER]       = sizeof(_lum_builtin_member_cloud_layer) / sizeof(LumBuiltinTypeMember),
};

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
  [LUM_BUILTIN_TYPE_OCEAN]            = _lum_builtin_member_ocean,
  [LUM_BUILTIN_TYPE_SKY]              = _lum_builtin_member_sky,
  [LUM_BUILTIN_TYPE_CLOUD]            = _lum_builtin_member_cloud,
  [LUM_BUILTIN_TYPE_FOG]              = _lum_builtin_member_fog,
  [LUM_BUILTIN_TYPE_PARTICLES]        = _lum_builtin_member_particles,
  [LUM_BUILTIN_TYPE_MATERIAL]         = _lum_builtin_member_material,
  [LUM_BUILTIN_TYPE_INSTANCE]         = _lum_builtin_member_instance,
  [LUM_BUILTIN_TYPE_STRING]           = 0,
  [LUM_BUILTIN_TYPE_LUMINARY]         = _lum_builtin_member_luminary,
  [LUM_BUILTIN_TYPE_WAVEFRONTOBJFILE] = 0,
  [LUM_BUILTIN_TYPE_ADAPTIVESAMPLING] = _lum_builtin_member_adaptive_sampling,
  [LUM_BUILTIN_TYPE_CLOUDLAYER]       = _lum_builtin_member_cloud_layer,
};

LuminaryResult lum_builtin_settings_init(LumBuiltinSettings* settings, uint32_t version) {
  __CHECK_NULL_ARGUMENT(settings);

  memset(settings, 0, sizeof(LumBuiltinSettings));

  if (version < 1)
    return LUMINARY_SUCCESS;

  settings->width                   = 2560;
  settings->height                  = 1440;
  settings->max_ray_depth           = 4;
  settings->bridge_max_num_vertices = 15;
  settings->undersampling           = 2;
  settings->supersampling           = 1;
  settings->shading_mode            = LUMINARY_SHADING_MODE_DEFAULT;
  settings->region_x                = 0.0f;
  settings->region_y                = 0.0f;
  settings->region_width            = 1.0f;
  settings->region_height           = 1.0f;

  settings->adaptive_sampling_settings = (LumBuiltinAdaptiveSampling) {.enable            = true,
                                                                       .max_sampling_rate = 256,
                                                                       .avg_sampling_rate = 2,
                                                                       .update_interval   = 64,
                                                                       .exposure_aware    = true,
                                                                       .output_mode       = LUMINARY_ADAPTIVE_SAMPLING_OUTPUT_MODE_BEAUTY};

  return LUMINARY_SUCCESS;
}

LuminaryResult lum_builtin_camera_init(LumBuiltinCamera* camera, uint32_t version) {
  __CHECK_NULL_ARGUMENT(camera);

  memset(camera, 0, sizeof(LumBuiltinCamera));

  if (version < 1)
    return LUMINARY_SUCCESS;

  camera->pos.x                        = 0.0f;
  camera->pos.y                        = 0.0f;
  camera->pos.z                        = 0.0f;
  camera->rotation.x                   = 0.0f;
  camera->rotation.y                   = 0.0f;
  camera->rotation.z                   = 0.0f;
  camera->aperture_shape               = LUMINARY_APERTURE_ROUND;
  camera->aperture_blade_count         = 7;
  camera->exposure                     = 0.0f;
  camera->bloom_blend                  = 0.01f;
  camera->dithering                    = 1;
  camera->tonemap                      = LUMINARY_TONEMAP_AGX;
  camera->use_local_error_minimization = false;
  camera->agx_custom_slope             = 1.0f;
  camera->agx_custom_power             = 1.0f;
  camera->agx_custom_saturation        = 1.0f;
  camera->filter                       = LUMINARY_FILTER_NONE;
  camera->purkinje                     = 1;
  camera->purkinje_kappa1              = 0.2f;
  camera->purkinje_kappa2              = 0.29f;
  camera->russian_roulette_threshold   = 0.1f;
  camera->use_color_correction         = 0;
  camera->color_correction.r           = 0.0f;
  camera->color_correction.g           = 0.0f;
  camera->color_correction.b           = 0.0f;
  camera->film_grain                   = 0.0f;
  camera->camera_scale                 = 1.0f;
  camera->object_distance              = 1.0f;
  camera->use_physical_camera          = false;

  return LUMINARY_SUCCESS;
}

LuminaryResult lum_builtin_ocean_init(LumBuiltinOcean* ocean, uint32_t version) {
  __CHECK_NULL_ARGUMENT(ocean);

  memset(ocean, 0, sizeof(LumBuiltinOcean));

  if (version < 1)
    return LUMINARY_SUCCESS;

  ocean->active                      = false;
  ocean->height                      = 0.0f;
  ocean->amplitude                   = 0.2f;
  ocean->frequency                   = 0.12f;
  ocean->refractive_index            = 1.333f;
  ocean->water_type                  = LUMINARY_JERLOV_WATER_TYPE_IB;
  ocean->caustics_active             = false;
  ocean->caustics_ris_sample_count   = 32;
  ocean->caustics_domain_scale       = 0.5f;
  ocean->multiscattering             = false;
  ocean->triangle_light_contribution = false;

  return LUMINARY_SUCCESS;
}

LuminaryResult lum_builtin_sky_init(LumBuiltinSky* sky, uint32_t version) {
  __CHECK_NULL_ARGUMENT(sky);

  memset(sky, 0, sizeof(LumBuiltinSky));

  if (version < 1)
    return LUMINARY_SUCCESS;

  sky->geometry_offset.x      = 0.0f;
  sky->geometry_offset.y      = 0.1f;
  sky->geometry_offset.z      = 0.0f;
  sky->altitude               = 0.5f;
  sky->azimuth                = 3.141f;
  sky->moon_altitude          = -0.5f;
  sky->moon_azimuth           = 0.0f;
  sky->moon_tex_offset        = 0.0f;
  sky->sun_strength           = 1.0f;
  sky->base_density           = 1.0f;
  sky->rayleigh_density       = 1.0f;
  sky->mie_density            = 1.0f;
  sky->ozone_density          = 1.0f;
  sky->ground_visibility      = 60.0f;
  sky->mie_diameter           = 2.0f;
  sky->ozone_layer_thickness  = 15.0f;
  sky->rayleigh_falloff       = 8.0f;
  sky->mie_falloff            = 1.7f;
  sky->multiscattering_factor = 1.0f;
  sky->steps                  = 40;
  sky->ozone_absorption       = true;
  sky->aerial_perspective     = false;
  sky->hdri_dim               = 2048;
  sky->hdri_samples           = 32;
  sky->stars_seed             = 0;
  sky->stars_count            = 10000;
  sky->stars_intensity        = 1.0f;
  sky->constant_color.r       = 1.0f;
  sky->constant_color.g       = 1.0f;
  sky->constant_color.b       = 1.0f;
  sky->mode                   = LUMINARY_SKY_MODE_DEFAULT;

  return LUMINARY_SUCCESS;
}

LuminaryResult lum_builtin_cloud_init(LumBuiltinCloud* cloud, uint32_t version) {
  __CHECK_NULL_ARGUMENT(cloud);

  memset(cloud, 0, sizeof(LumBuiltinCloud));

  if (version < 1)
    return LUMINARY_SUCCESS;

  cloud->active                = false;
  cloud->steps                 = 96;
  cloud->shadow_steps          = 8;
  cloud->atmosphere_scattering = true;
  cloud->seed                  = 0;
  cloud->offset_x              = 0.0f;
  cloud->offset_z              = 0.0f;
  cloud->noise_shape_scale     = 1.0f;
  cloud->noise_detail_scale    = 1.0f;
  cloud->noise_weather_scale   = 1.0f;
  cloud->octaves               = 9;
  cloud->droplet_diameter      = 25.0f;
  cloud->density               = 1.0f;
  cloud->mipmap_bias           = 0.0f;

  cloud->low = (LumBuiltinCloudLayer) {
    .active       = true,
    .height_max   = 5.0f,
    .height_min   = 1.5f,
    .coverage     = 1.0f,
    .coverage_min = 0.0f,
    .type         = 1.0f,
    .type_min     = 0.0f,
    .wind_speed   = 2.5f,
    .wind_angle   = 0.0f,
  };

  cloud->mid = (LumBuiltinCloudLayer) {
    .active       = true,
    .height_max   = 6.0f,
    .height_min   = 5.5f,
    .coverage     = 1.0f,
    .coverage_min = 0.0f,
    .type         = 1.0f,
    .type_min     = 0.0f,
    .wind_speed   = 2.5f,
    .wind_angle   = 0.0f,
  };

  cloud->top = (LumBuiltinCloudLayer) {
    .active       = true,
    .height_max   = 8.0f,
    .height_min   = 7.95f,
    .coverage     = 1.0f,
    .coverage_min = 0.0f,
    .type         = 1.0f,
    .type_min     = 0.0f,
    .wind_speed   = 1.0f,
    .wind_angle   = 0.0f,
  };

  return LUMINARY_SUCCESS;
}

LuminaryResult lum_builtin_fog_init(LumBuiltinFog* fog, uint32_t version) {
  __CHECK_NULL_ARGUMENT(fog);

  memset(fog, 0, sizeof(LumBuiltinFog));

  if (version < 1)
    return LUMINARY_SUCCESS;

  fog->active           = false;
  fog->density          = 1.0f;
  fog->droplet_diameter = 10.0f;
  fog->height           = 500.0f;
  fog->dist             = 500.0f;

  return LUMINARY_SUCCESS;
}

LuminaryResult lum_builtin_particles_init(LumBuiltinParticles* particles, uint32_t version) {
  __CHECK_NULL_ARGUMENT(particles);

  memset(particles, 0, sizeof(LumBuiltinParticles));

  if (version < 1)
    return LUMINARY_SUCCESS;

  particles->active             = false;
  particles->scale              = 10.0f;
  particles->albedo.r           = 1.0f;
  particles->albedo.g           = 1.0f;
  particles->albedo.b           = 1.0f;
  particles->direction_altitude = 1.234f;
  particles->direction_azimuth  = 0.0f;
  particles->speed              = 0.0f;
  particles->phase_diameter     = 50.0f;
  particles->seed               = 0;
  particles->count              = 8192;
  particles->size               = 1.0f;
  particles->size_variation     = 0.1f;

  return LUMINARY_SUCCESS;
}

LuminaryResult lum_builtin_material_init(LumBuiltinMaterial* material, uint32_t version) {
  __CHECK_NULL_ARGUMENT(material);

  memset(material, 0, sizeof(LumBuiltinMaterial));

  if (version < 1)
    return LUMINARY_SUCCESS;

  material->base_substrate           = LUMINARY_MATERIAL_BASE_SUBSTRATE_OPAQUE;
  material->albedo                   = (RGBF) {.r = 0.9f, .g = 0.9f, .b = 0.9f};
  material->opacity                  = 0.9f;
  material->emission                 = (RGBF) {.r = 0.0f, .g = 0.0f, .b = 0.0f};
  material->emission_scale           = 1.0f;
  material->roughness                = 0.7f;
  material->roughness_clamp          = 0.25f;
  material->refraction_index         = 1.0f;
  material->emission_active          = false;
  material->thin_walled              = false;
  material->metallic                 = false;
  material->colored_transparency     = false;
  material->normal_map_is_compressed = true;
  material->bidirectional_emission   = false;
  material->albedo_tex               = TEXTURE_NONE;
  material->luminance_tex            = TEXTURE_NONE;
  material->roughness_tex            = TEXTURE_NONE;
  material->metallic_tex             = TEXTURE_NONE;
  material->normal_tex               = TEXTURE_NONE;

  return LUMINARY_SUCCESS;
}

LuminaryResult lum_builtin_instance_init(LumBuiltinInstance* instance, uint32_t version) {
  __CHECK_NULL_ARGUMENT(instance);

  memset(instance, 0, sizeof(LumBuiltinInstance));

  if (version < 1)
    return LUMINARY_SUCCESS;

  instance->mesh_id = 0;
  instance->scale.x = 1.0f;
  instance->scale.y = 1.0f;
  instance->scale.z = 1.0f;

  return LUMINARY_SUCCESS;
}
