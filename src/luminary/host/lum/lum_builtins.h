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
  LUM_BUILTIN_TYPE_WAVEFRONTOBJFILE,
  LUM_BUILTIN_TYPE_ADAPTIVESAMPLING,
  LUM_BUILTIN_TYPE_CLOUDLAYER,
  LUM_BUILTIN_TYPE_CAMERATHINLENS,
  LUM_BUILTIN_TYPE_CAMERAPHYSICAL,
  LUM_BUILTIN_TYPE_COUNT_VERSION_1,

  LUM_BUILTIN_TYPE_COUNT = LUM_BUILTIN_TYPE_COUNT_VERSION_1
} typedef LumBuiltinType;

extern const char* lum_builtin_types_strings[LUM_BUILTIN_TYPE_COUNT];
extern const size_t lum_builtin_types_sizes[LUM_BUILTIN_TYPE_COUNT];
extern const char* lum_builtin_types_mnemonic[LUM_BUILTIN_TYPE_COUNT];
extern const bool lum_builtin_types_addressable[LUM_BUILTIN_TYPE_COUNT];
extern const bool lum_builtin_types_accessible[LUM_BUILTIN_TYPE_COUNT];

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

struct LumBuiltinCameraThinLens {
  float fov;
  float aperture_size;
} typedef LumBuiltinCameraThinLens;

struct LumBuiltinCameraPhysical {
  bool allow_reflections;
  bool use_spectral_rendering;
  float focal_length;
  float front_focal_point;
  float back_focal_point;
  float front_principal_point;
  float back_principal_point;
  float aperture_point;
  float aperture_diameter;
  float exit_pupil_point;
  float exit_pupil_diameter;
  float image_plane_distance;
  float sensor_width;
} typedef LumBuiltinCameraPhysical;

struct LumBuiltinCamera {
  LuminaryVec3 pos;
  LuminaryVec3 rotation;
  LuminaryApertureShape aperture_shape;
  uint32_t aperture_blade_count;
  float exposure;
  LuminaryToneMap tonemap;
  float agx_custom_slope;
  float agx_custom_power;
  float agx_custom_saturation;
  LuminaryFilter filter;
  bool use_local_error_minimization;
  float bloom_blend;
  bool dithering;
  bool purkinje;
  float purkinje_kappa1;
  float purkinje_kappa2;
  float russian_roulette_threshold;
  bool use_color_correction;
  LuminaryRGBF color_correction;
  float film_grain;
  float camera_scale;
  float object_distance;
  bool use_physical_camera;
  LumBuiltinCameraThinLens thin_lens;
  LumBuiltinCameraPhysical physical;
} typedef LumBuiltinCamera;

struct LumBuiltinOcean {
  bool active;
  float height;
  float amplitude;
  float frequency;
  float refractive_index;
  LuminaryJerlovWaterType water_type;
  bool caustics_active;
  uint32_t caustics_ris_sample_count;
  float caustics_domain_scale;
  bool multiscattering;
  bool triangle_light_contribution;
} typedef LumBuiltinOcean;

struct LumBuiltinSky {
  LuminaryVec3 geometry_offset;
  float azimuth;
  float altitude;
  float moon_azimuth;
  float moon_altitude;
  float moon_tex_offset;
  float sun_strength;
  float base_density;
  bool ozone_absorption;
  uint32_t steps;
  uint32_t stars_count;
  uint32_t stars_seed;
  float stars_intensity;
  float rayleigh_density;
  float mie_density;
  float ozone_density;
  float rayleigh_falloff;
  float mie_falloff;
  float mie_diameter;
  float ground_visibility;
  float ozone_layer_thickness;
  float multiscattering_factor;
  uint32_t hdri_dim;
  uint32_t hdri_samples;
  bool aerial_perspective;
  LuminaryRGBF constant_color;
  LuminarySkyMode mode;
} typedef LumBuiltinSky;

struct LumBuiltinCloudLayer {
  bool active;
  float height_max;
  float height_min;
  float coverage;
  float coverage_min;
  float type;
  float type_min;
  float wind_speed;
  float wind_angle;
} typedef LumBuiltinCloudLayer;

struct LumBuiltinCloud {
  bool active;
  bool atmosphere_scattering;
  LumBuiltinCloudLayer low;
  LumBuiltinCloudLayer mid;
  LumBuiltinCloudLayer top;
  float offset_x;
  float offset_z;
  float density;
  uint32_t seed;
  float droplet_diameter;
  uint32_t steps;
  uint32_t shadow_steps;
  float noise_shape_scale;
  float noise_detail_scale;
  float noise_weather_scale;
  float mipmap_bias;
  uint32_t octaves;
} typedef LumBuiltinCloud;

struct LumBuiltinFog {
  bool active;
  float density;
  float droplet_diameter;
  float height;
  float dist;
} typedef LumBuiltinFog;

struct LumBuiltinParticles {
  bool active;
  uint32_t seed;
  uint32_t count;
  LuminaryRGBF albedo;
  float speed;
  float direction_altitude;
  float direction_azimuth;
  float phase_diameter;
  float scale;
  float size;
  float size_variation;
} typedef LumBuiltinParticles;

struct LumBuiltinMaterial {
  LuminaryMaterialBaseSubstrate base_substrate;
  LuminaryRGBF albedo;
  float opacity;
  LuminaryRGBF emission;
  float emission_scale;
  float roughness;
  float roughness_clamp;
  float refraction_index;
  bool emission_active;
  bool thin_walled;
  bool metallic;
  bool colored_transparency;
  bool roughness_as_smoothness;
  bool normal_map_is_compressed;
  bool bidirectional_emission;
  uint32_t albedo_tex;
  uint32_t luminance_tex;
  uint32_t roughness_tex;
  uint32_t metallic_tex;
  uint32_t normal_tex;
} typedef LumBuiltinMaterial;

struct LumBuiltinInstance {
  uint32_t mesh_id;
  LuminaryVec3 position;
  LuminaryVec3 rotation;
  LuminaryVec3 scale;
} typedef LumBuiltinInstance;

////////////////////////////////////////////////////////////////////
// LumBuiltin Default Initializers
////////////////////////////////////////////////////////////////////

LuminaryResult lum_builtin_settings_init(LumBuiltinSettings* settings, uint32_t version);
LuminaryResult lum_builtin_camera_init(LumBuiltinCamera* camera, uint32_t version);
LuminaryResult lum_builtin_ocean_init(LumBuiltinOcean* ocean, uint32_t version);
LuminaryResult lum_builtin_sky_init(LumBuiltinSky* sky, uint32_t version);
LuminaryResult lum_builtin_cloud_init(LumBuiltinCloud* cloud, uint32_t version);
LuminaryResult lum_builtin_fog_init(LumBuiltinFog* fog, uint32_t version);
LuminaryResult lum_builtin_particles_init(LumBuiltinParticles* particles, uint32_t version);
LuminaryResult lum_builtin_material_init(LumBuiltinMaterial* material, uint32_t version);
LuminaryResult lum_builtin_instance_init(LumBuiltinInstance* instance, uint32_t version);

////////////////////////////////////////////////////////////////////
// LumBuiltin Conversion
////////////////////////////////////////////////////////////////////

LuminaryResult lum_builtin_settings_convert(const LumBuiltinSettings* settings, LuminaryRendererSettings* dst_settings, uint32_t version);
LuminaryResult lum_builtin_camera_convert(const LumBuiltinCamera* camera, LuminaryCamera* dst_camera, uint32_t version);
LuminaryResult lum_builtin_ocean_convert(const LumBuiltinOcean* ocean, LuminaryOcean* dst_ocean, uint32_t version);
LuminaryResult lum_builtin_sky_convert(const LumBuiltinSky* sky, LuminarySky* dst_sky, uint32_t version);
LuminaryResult lum_builtin_cloud_convert(const LumBuiltinCloud* cloud, LuminaryCloud* dst_cloud, uint32_t version);
LuminaryResult lum_builtin_fog_convert(const LumBuiltinFog* fog, LuminaryFog* dst_fog, uint32_t version);
LuminaryResult lum_builtin_particles_convert(const LumBuiltinParticles* particles, LuminaryParticles* dst_particles, uint32_t version);
LuminaryResult lum_builtin_material_convert(const LumBuiltinMaterial* material, LuminaryMaterial* dst_material, uint32_t version);
LuminaryResult lum_builtin_instance_convert(const LumBuiltinInstance* instance, LuminaryInstance* dst_instance, uint32_t version);

#endif /* LUMINARY_LUM_BUILTINS_H */
