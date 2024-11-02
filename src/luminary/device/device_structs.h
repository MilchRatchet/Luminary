#ifndef LUMINARY_DEVICE_STRUCTS_H
#define LUMINARY_DEVICE_STRUCTS_H

#include "device_nv_includes.h"
#include "scene.h"
#include "utils.h"

struct DeviceRendererSettings {
  uint32_t max_ray_depth : 8;
  uint32_t shading_mode : 3;
  uint32_t bridge_max_num_vertices : 4;
  uint32_t bridge_num_ris_samples : 6;
  uint32_t light_num_rays : 5;
  uint32_t light_num_ris_samples : 6;

  uint16_t width;
  uint16_t height;
} typedef DeviceRendererSettings;
LUM_STATIC_SIZE_ASSERT(DeviceRendererSettings, 0x08u);

struct DeviceCamera {
  uint32_t aperture_shape : 1;
  uint32_t aperture_blade_count : 3;
  uint32_t tonemap : 3;
  uint32_t dithering : 1;
  uint32_t purkinje : 1;
  uint32_t use_color_correction : 1;
  uint32_t do_firefly_clamping : 1;

  vec3 pos;
  vec3 rotation;
  float fov;
  float focal_length;
  float aperture_size;
  float exposure;
  float purkinje_kappa1;
  float purkinje_kappa2;
  float russian_roulette_threshold;
  float film_grain;
} typedef DeviceCamera;
LUM_STATIC_SIZE_ASSERT(DeviceCamera, 0x3Cu);

struct DeviceOcean {
  uint32_t active : 1;
  uint32_t water_type : 4;
  uint32_t caustics_active : 1;
  uint32_t caustics_ris_sample_count : 7;
  uint32_t multiscattering : 1;
  uint32_t triangle_light_contribution : 1;

  float height;
  float amplitude;
  float frequency;
  float choppyness;
  float refractive_index;
  float caustics_domain_scale;
} typedef DeviceOcean;
LUM_STATIC_SIZE_ASSERT(DeviceOcean, 0x1Cu);

struct DeviceSky {
  uint32_t ozone_absorption : 1;
  uint32_t steps : 10;
  uint32_t aerial_perspective : 1;
  uint32_t ambient_sampling : 1;
  uint32_t mode : 2;

  vec3 geometry_offset;
  float moon_tex_offset;
  float sun_strength;
  float base_density;
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
  vec3 sun_pos;
  vec3 moon_pos;
  vec3 hdri_origin;
  float hdri_mip_bias;
  RGBF constant_color;
} typedef DeviceSky;
LUM_STATIC_SIZE_ASSERT(DeviceSky, 0x78u);

struct DeviceCloudLayer {
  float height_max;
  float height_min;
  float coverage;
  float coverage_min;
  float type;
  float type_min;
  float wind_speed;
  float wind_angle;
} typedef DeviceCloudLayer;
LUM_STATIC_SIZE_ASSERT(DeviceCloudLayer, 0x20u);

struct DeviceCloud {
  uint32_t active : 1;
  uint32_t atmosphere_scattering : 1;
  uint32_t steps : 10;
  uint32_t shadow_steps : 10;
  uint32_t octaves : 4;
  uint32_t low_active : 1;
  uint32_t mid_active : 1;
  uint32_t top_active : 1;

  float offset_x;
  float offset_z;
  float density;
  float droplet_diameter;
  float noise_shape_scale;
  float noise_detail_scale;
  float noise_weather_scale;
  float mipmap_bias;
  DeviceCloudLayer low;
  DeviceCloudLayer mid;
  DeviceCloudLayer top;
} typedef DeviceCloud;
LUM_STATIC_SIZE_ASSERT(DeviceCloud, 0x84u);

struct DeviceFog {
  uint32_t active : 1;

  float density;
  float droplet_diameter;
  float height;
  float dist;
} typedef DeviceFog;
LUM_STATIC_SIZE_ASSERT(DeviceFog, 0x14u);

struct DeviceParticles {
  uint32_t active : 1;
  uint32_t count : 31;

  RGBF albedo;
  float speed;
  float direction_altitude;
  float direction_azimuth;
  float phase_diameter;
  float scale;
  float size;
  float size_variation;
} typedef DeviceParticles;
LUM_STATIC_SIZE_ASSERT(DeviceParticles, 0x2Cu);

struct DeviceToy {
  uint32_t active : 1;
  uint32_t emissive : 1;

  vec3 position;
  vec3 rotation;
  float scale;
  float refractive_index;
  RGBAF albedo;
  RGBAF material;
  RGBF emission;
} typedef DeviceToy;
LUM_STATIC_SIZE_ASSERT(DeviceToy, 0x50u);

enum DeviceMaterialFlags {
  DEVICE_MATERIAL_FLAG_EMISSION             = 0x01,
  DEVICE_MATERIAL_FLAG_IOR_SHADOWING        = 0x02,
  DEVICE_MATERIAL_FLAG_THIN_WALLED          = 0x04,
  DEVICE_MATERIAL_FLAG_COLORED_TRANSPARENCY = 0x08
} typedef DeviceMaterialFlags;

struct DeviceMaterialCompressed {
  uint8_t flags;
  uint8_t roughness_clamp;
  uint16_t metallic;
  uint16_t roughness;
  uint16_t refraction_index;

  uint16_t albedo_r;
  uint16_t albedo_g;
  uint16_t albedo_b;
  uint16_t albedo_a;

  uint16_t emission_r;
  uint16_t emission_g;
  uint16_t emission_b;
  uint16_t emission_scale;

  uint16_t albedo_tex;
  uint16_t luminance_tex;
  uint16_t material_tex;
  uint16_t normal_tex;
} typedef DeviceMaterialCompressed;
LUM_STATIC_SIZE_ASSERT(DeviceMaterialCompressed, 0x20u);

struct DeviceMaterial {
  uint8_t flags;
  float roughness_clamp;
  float metallic;
  float roughness;
  float refraction_index;

  RGBAF albedo;
  RGBF emission;
  float emission_scale;

  uint16_t albedo_tex;
  uint16_t luminance_tex;
  uint16_t material_tex;
  uint16_t normal_tex;
} typedef DeviceMaterial;

struct DeviceTriangle {
  vec3 vertex;
  vec3 edge1;
  vec3 edge2;
  uint32_t vertex_texture;
  uint32_t edge1_texture;
  uint32_t edge2_texture;
  uint32_t vertex_normal;
  uint32_t edge1_normal;
  uint32_t edge2_normal;
  uint32_t padding;
} typedef DeviceTriangle;
LUM_STATIC_SIZE_ASSERT(DeviceTriangle, 0x40u);

typedef CUtexObject DeviceTextureHandle;

struct DeviceTextureObject {
  DeviceTextureHandle handle;
  float gamma;
  float padding;
} typedef DeviceTextureObject;
LUM_STATIC_SIZE_ASSERT(DeviceTextureObject, 0x10u);

struct DeviceInstancelet {
  uint32_t triangles_offset;
  uint16_t material_id;
  uint16_t padding;
} typedef DeviceInstancelet;
LUM_STATIC_SIZE_ASSERT(DeviceInstancelet, 0x08u);

struct DeviceTransform {
  vec3 offset;
  vec3 scale;
  Quaternion16 rotation;
} typedef DeviceTransform;
LUM_STATIC_SIZE_ASSERT(DeviceTransform, 0x20u);

struct DeviceTexture typedef DeviceTexture;
struct DeviceMesh typedef DeviceMesh;

LuminaryResult device_struct_settings_convert(const RendererSettings* settings, DeviceRendererSettings* device_settings);
LuminaryResult device_struct_camera_convert(const Camera* camera, DeviceCamera* device_camera);
LuminaryResult device_struct_ocean_convert(const Ocean* ocean, DeviceOcean* device_ocean);
LuminaryResult device_struct_sky_convert(const Sky* sky, DeviceSky* device_sky);
LuminaryResult device_struct_cloud_convert(const Cloud* cloud, DeviceCloud* device_cloud);
LuminaryResult device_struct_fog_convert(const Fog* fog, DeviceFog* device_fog);
LuminaryResult device_struct_particles_convert(const Particles* particles, DeviceParticles* device_particles);
LuminaryResult device_struct_toy_convert(const Toy* toy, DeviceToy* device_toy);
LuminaryResult device_struct_material_convert(const Material* material, DeviceMaterialCompressed* device_material);
LuminaryResult device_struct_scene_entity_convert(const void* source, void* dst, SceneEntity entity);
LuminaryResult device_struct_triangle_convert(const Triangle* triangle, DeviceTriangle* device_triangle);
LuminaryResult device_struct_texture_object_convert(const struct DeviceTexture* texture, DeviceTextureObject* texture_object);
LuminaryResult device_struct_instance_convert(
  const MeshInstance* instance, DeviceInstancelet* device_instance, DeviceTransform* device_transform, const DeviceMesh* device_mesh,
  const uint32_t meshlet_id);

#endif /* LUMINARY_DEVICE_STRUCTS_H */
