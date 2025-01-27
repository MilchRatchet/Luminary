/*
  Copyright (C) 2021-2024 Max Jenke

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU Affero General Public License as published
  by the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU Affero General Public License for more details.

  You should have received a copy of the GNU Affero General Public License
  along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

#ifndef LUMINARY_API_STRUCTS_H
#define LUMINARY_API_STRUCTS_H

#include <luminary/api_utils.h>

////////////////////////////////////////////////////////////////////
// General
////////////////////////////////////////////////////////////////////

// 3 bits reserved
LUMINARY_API enum LuminaryShadingMode {
  LUMINARY_SHADING_MODE_DEFAULT        = 0,
  LUMINARY_SHADING_MODE_ALBEDO         = 1,
  LUMINARY_SHADING_MODE_DEPTH          = 2,
  LUMINARY_SHADING_MODE_NORMAL         = 3,
  LUMINARY_SHADING_MODE_IDENTIFICATION = 4,
  LUMINARY_SHADING_MODE_LIGHTS         = 5,
  LUMINARY_SHADING_MODE_COUNT
} typedef LuminaryShadingMode;

LUMINARY_API struct LuminaryRendererSettings {
  uint32_t width;
  uint32_t height;
  uint32_t max_ray_depth;
  uint32_t bridge_max_num_vertices;
  uint32_t bridge_num_ris_samples;
  uint32_t light_num_ris_samples;
  uint32_t light_num_rays;
  LUMINARY_DEPRECATED bool use_opacity_micromaps;
  LUMINARY_DEPRECATED bool use_displacement_micromaps;
  uint32_t undersampling;
  uint32_t supersampling;
  LuminaryShadingMode shading_mode;
  uint32_t max_sample_count;
  uint32_t num_indirect_buckets;
} typedef LuminaryRendererSettings;

LUMINARY_API struct LuminaryDeviceInfo {
  char name[256];
  size_t memory_size;
  size_t allocated_memory_size;
} typedef LuminaryDeviceInfo;

LUMINARY_API struct LuminaryOutputProperties {
  bool enabled;
  uint32_t width;
  uint32_t height;
} typedef LuminaryOutputProperties;

#define LUMINARY_OUTPUT_HANDLE_INVALID 0xFFFFFFFF
LUMINARY_API typedef uint32_t LuminaryOutputHandle;

LUMINARY_API struct LuminaryOutputRequestProperties {
  uint32_t sample_count;
  uint32_t width;
  uint32_t height;
} typedef LuminaryOutputRequestProperties;

LUMINARY_API typedef uint32_t LuminaryOutputPromiseHandle;

LUMINARY_API struct LuminaryPixelQueryResult {
  bool pixel_query_is_valid;
  uint32_t instance_id;
  uint16_t material_id;
  float depth;
  LuminaryVec3 rel_hit_pos;
} typedef LuminaryPixelQueryResult;

LUMINARY_API struct LuminaryImage {
  uint8_t* buffer;
  uint32_t width;
  uint32_t height;
  size_t ld;
  struct {
    float time;
    uint32_t sample_count;
  } meta_data;
} typedef LuminaryImage;

////////////////////////////////////////////////////////////////////
// Camera
////////////////////////////////////////////////////////////////////

// 3 bits reserved
LUMINARY_API enum LuminaryFilter {
  LUMINARY_FILTER_NONE       = 0,
  LUMINARY_FILTER_GRAY       = 1,
  LUMINARY_FILTER_SEPIA      = 2,
  LUMINARY_FILTER_GAMEBOY    = 3,
  LUMINARY_FILTER_2BITGRAY   = 4,
  LUMINARY_FILTER_CRT        = 5,
  LUMINARY_FILTER_BLACKWHITE = 6,
  LUMINARY_FILTER_COUNT
} typedef LuminaryFilter;

// 3 bits reserved
LUMINARY_API enum LuminaryToneMap {
  LUMINARY_TONEMAP_NONE       = 0,
  LUMINARY_TONEMAP_ACES       = 1,
  LUMINARY_TONEMAP_REINHARD   = 2,
  LUMINARY_TONEMAP_UNCHARTED2 = 3,
  LUMINARY_TONEMAP_AGX        = 4,
  LUMINARY_TONEMAP_AGX_PUNCHY = 5,
  LUMINARY_TONEMAP_AGX_CUSTOM = 6,
  LUMINARY_TONEMAP_COUNT
} typedef LuminaryToneMap;

LUMINARY_API enum LuminaryApertureShape {
  LUMINARY_APERTURE_ROUND  = 0,
  LUMINARY_APERTURE_BLADED = 1,
  LUMINARY_APERTURE_COUNT
} typedef LuminaryApertureShape;

LUMINARY_API struct LuminaryCamera {
  LuminaryVec3 pos;
  LuminaryVec3 rotation;
  float fov;
  float focal_length;
  float aperture_size;
  LuminaryApertureShape aperture_shape;
  uint32_t aperture_blade_count;
  float exposure;
  float max_exposure;
  float min_exposure;
  bool auto_exposure;
  LuminaryToneMap tonemap;
  float agx_custom_slope;
  float agx_custom_power;
  float agx_custom_saturation;
  LuminaryFilter filter;
  float bloom_blend;
  bool lens_flare;
  float lens_flare_threshold;
  bool dithering;
  bool purkinje;
  float purkinje_kappa1;
  float purkinje_kappa2;
  float wasd_speed;
  float mouse_speed;
  bool smooth_movement;
  float smoothing_factor;
  float russian_roulette_threshold;
  bool use_color_correction;
  LuminaryRGBF color_correction;
  bool do_firefly_clamping;
  float film_grain;
  bool indirect_only;
} typedef LuminaryCamera;

////////////////////////////////////////////////////////////////////
// Ocean
////////////////////////////////////////////////////////////////////

// 4 bits reserved
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
  LUMINARY_JERLOV_WATER_TYPE_9C  = 9,
  LUMINARY_JERLOV_WATER_TYPE_COUNT
} typedef LuminaryJerlovWaterType;

LUMINARY_API struct LuminaryOcean {
  bool active;
  float height;
  float amplitude;
  float frequency;
  float choppyness;
  float refractive_index;
  LuminaryJerlovWaterType water_type;
  bool caustics_active;
  uint32_t caustics_ris_sample_count;
  float caustics_domain_scale;
  bool multiscattering;
  bool triangle_light_contribution;
} typedef LuminaryOcean;

////////////////////////////////////////////////////////////////////
// Sky
////////////////////////////////////////////////////////////////////

// 2 bits reserved
enum LuminarySkyMode {
  LUMINARY_SKY_MODE_DEFAULT        = 0,
  LUMINARY_SKY_MODE_HDRI           = 1,
  LUMINARY_SKY_MODE_CONSTANT_COLOR = 2,
  LUMINARY_SKY_MODE_COUNT
} typedef LuminarySkyMode;

struct LuminarySky {
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
  uint32_t settings_stars_count;
  uint32_t current_stars_count;
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
  bool lut_initialized;
  bool hdri_initialized;
  uint32_t hdri_dim;
  uint32_t settings_hdri_dim;
  uint32_t hdri_samples;
  LuminaryVec3 hdri_origin;
  float hdri_mip_bias;
  bool aerial_perspective;
  LuminaryRGBF constant_color;
  bool ambient_sampling;
  LuminarySkyMode mode;
} typedef LuminarySky;

////////////////////////////////////////////////////////////////////
// Clouds
////////////////////////////////////////////////////////////////////

LUMINARY_API struct LuminaryCloudLayer {
  bool active;
  float height_max;
  float height_min;
  float coverage;
  float coverage_min;
  float type;
  float type_min;
  float wind_speed;
  float wind_angle;
} typedef LuminaryCloudLayer;

LUMINARY_API struct LuminaryCloud {
  bool active;
  bool initialized;
  bool atmosphere_scattering;
  LuminaryCloudLayer low;
  LuminaryCloudLayer mid;
  LuminaryCloudLayer top;
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
} typedef LuminaryCloud;

////////////////////////////////////////////////////////////////////
// Fog
////////////////////////////////////////////////////////////////////

LUMINARY_API struct LuminaryFog {
  bool active;
  float density;
  float droplet_diameter;
  float height;
  float dist;
} typedef LuminaryFog;

////////////////////////////////////////////////////////////////////
// Particles
////////////////////////////////////////////////////////////////////

LUMINARY_API struct LuminaryParticles {
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
} typedef LuminaryParticles;

////////////////////////////////////////////////////////////////////
// Material
////////////////////////////////////////////////////////////////////

enum LuminaryMaterialBaseSubstrate {
  LUMINARY_MATERIAL_BASE_SUBSTRATE_OPAQUE,
  LUMINARY_MATERIAL_BASE_SUBSTRATE_TRANSLUCENT,
  LUMINARY_MATERIAL_BASE_SUBSTRATE_COUNT
} typedef LuminaryMaterialBaseSubstrate;

LUMINARY_API struct LuminaryMaterial {
  uint32_t id;
  LuminaryMaterialBaseSubstrate base_substrate;
  LuminaryRGBAF albedo;
  LuminaryRGBF emission;
  float emission_scale;
  float roughness;
  float roughness_clamp;
  float refraction_index;
  bool emission_active;
  bool thin_walled;
  bool metallic;
  bool colored_transparency;
  uint16_t albedo_tex;
  uint16_t luminance_tex;
  uint16_t roughness_tex;
  uint16_t metallic_tex;
  uint16_t normal_tex;
} typedef LuminaryMaterial;

////////////////////////////////////////////////////////////////////
// LuminaryInstance
////////////////////////////////////////////////////////////////////

LUMINARY_API struct LuminaryInstance {
  uint32_t mesh_id;
  LuminaryVec3 position;
  LuminaryVec3 rotation;
  LuminaryVec3 scale;
} typedef LuminaryInstance;

#endif /* LUMINARY_STRUCTS_H */
