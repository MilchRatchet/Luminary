#ifndef LUMINARY_DEVICE_STRUCTS_H
#define LUMINARY_DEVICE_STRUCTS_H

#include "device_nv_includes.h"
#include "scene.h"
#include "utils.h"

struct DeviceRendererSettings {
  uint32_t max_ray_depth : 6;
  uint32_t shading_mode : 3;
  uint32_t bridge_max_num_vertices : 4;
  uint32_t supersampling : 2;
  // 12 bits spare

  uint16_t width;
  uint16_t height;
  uint16_t window_x;
  uint16_t window_y;
  uint16_t window_width;
  uint16_t window_height;
} typedef DeviceRendererSettings;
LUM_STATIC_SIZE_ASSERT(DeviceRendererSettings, 0x10u);

struct DeviceCameraMedium {
  float design_ior;
  float abbe;
} typedef DeviceCameraMedium;
LUM_STATIC_SIZE_ASSERT(DeviceCameraMedium, 0x08u);

struct DeviceCameraInterface {
  float diameter;
  float radius;
  float vertex;
  float padding;
} typedef DeviceCameraInterface;
LUM_STATIC_SIZE_ASSERT(DeviceCameraInterface, 0x10u);

struct DeviceCamera {
  uint32_t aperture_shape : 1;
  uint32_t aperture_blade_count : 3;
  uint32_t tonemap : 3;
  uint32_t dithering : 1;
  uint32_t purkinje : 1;
  uint32_t use_color_correction : 1;
  uint32_t do_firefly_rejection : 1;
  uint32_t indirect_only : 1;
  uint32_t allow_reflections : 1;
  uint32_t use_spectral_rendering : 1;
  uint32_t use_physical_camera : 1;
  // 17 bits spare

  vec3 pos;
  Quaternion rotation;
  float exposure;
  float purkinje_kappa1;
  float purkinje_kappa2;
  float russian_roulette_threshold;
  float film_grain;
  float camera_scale;
  float object_distance;

  union {
    struct {
      uint32_t num_interfaces;
      float focal_length;
      float front_focal_point;
      float back_focal_point;
      float front_principal_point;
      float back_principal_point;
      float aperture_point;
      float aperture_radius;
      float exit_pupil_point;
      float exit_pupil_radius;
      float image_plane_distance;
      float sensor_width;
    } physical;
    struct {
      float fov;
      float aperture_size;
    } thin_lens;
  };
} typedef DeviceCamera;
LUM_STATIC_SIZE_ASSERT(DeviceCamera, 0x6Cu);

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
  float refractive_index;
  float caustics_domain_scale;
} typedef DeviceOcean;
LUM_STATIC_SIZE_ASSERT(DeviceOcean, 0x18u);

struct DeviceSky {
  uint32_t ozone_absorption : 1;
  uint32_t steps : 10;
  uint32_t aerial_perspective : 1;
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
  RGBF constant_color;
} typedef DeviceSky;
LUM_STATIC_SIZE_ASSERT(DeviceSky, 0x68u);

struct DeviceCloudLayer {
  float height_max;
  float height_min;
  float coverage;
  float coverage_min;
  float type;
  float type_min;
  float wind_speed;
  float wind_angle_cos;
  float wind_angle_sin;
} typedef DeviceCloudLayer;
LUM_STATIC_SIZE_ASSERT(DeviceCloudLayer, 0x24u);

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
LUM_STATIC_SIZE_ASSERT(DeviceCloud, 0x90u);

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

union DeviceSceneEntityCover {
  DeviceRendererSettings settings;
  DeviceCamera camera;
  DeviceOcean ocean;
  DeviceSky sky;
  DeviceCloud cloud;
  DeviceFog fog;
  DeviceParticles particles;
} typedef DeviceSceneEntityCover;

enum DeviceMaterialFlags {
  // 1 bit for the base substrate
  DEVICE_MATERIAL_BASE_SUBSTRATE_OPAQUE      = 0x00,
  DEVICE_MATERIAL_BASE_SUBSTRATE_TRANSLUCENT = 0x01,
  DEVICE_MATERIAL_BASE_SUBSTRATE_MASK        = 0x01,

  DEVICE_MATERIAL_FLAG_EMISSION                = 0x02,
  DEVICE_MATERIAL_FLAG_THIN_WALLED             = 0x04,
  DEVICE_MATERIAL_FLAG_METALLIC                = 0x08,
  DEVICE_MATERIAL_FLAG_COLORED_TRANSPARENCY    = 0x10,
  DEVICE_MATERIAL_FLAG_ROUGHNESS_AS_SMOOTHNESS = 0x20,
  DEVICE_MATERIAL_FLAG_NORMAL_MAP_COMPRESSED   = 0x40
  // 1 bits unused
} typedef DeviceMaterialFlags;

struct DeviceMaterialCompressed {
  uint8_t flags;
  uint8_t roughness_clamp;
  uint16_t metallic_tex;
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
  uint16_t roughness_tex;
  uint16_t normal_tex;
} typedef DeviceMaterialCompressed;
LUM_STATIC_SIZE_ASSERT(DeviceMaterialCompressed, 0x20u);

struct DeviceMaterial {
  uint8_t flags;
  float roughness_clamp;
  float roughness;
  float refraction_index;

  RGBAF albedo;
  RGBF emission;
  float emission_scale;

  uint16_t albedo_tex;
  uint16_t luminance_tex;
  uint16_t roughness_tex;
  uint16_t metallic_tex;
  uint16_t normal_tex;
} typedef DeviceMaterial;

struct DeviceTriangle {
  vec3 vertex;
  vec3 edge1;
  vec3 edge2;
  uint32_t vertex_texture;
  uint32_t vertex1_texture;
  uint32_t vertex2_texture;
  uint32_t vertex_normal;
  uint32_t vertex1_normal;
  uint32_t vertex2_normal;
  uint16_t material_id;
  uint16_t padding;
} typedef DeviceTriangle;
LUM_STATIC_SIZE_ASSERT(DeviceTriangle, 0x40u);

typedef CUtexObject DeviceTextureHandle;

struct DeviceTextureObject {
  DeviceTextureHandle handle;
  float gamma;
  uint16_t width;
  uint16_t height;
} typedef DeviceTextureObject;
LUM_STATIC_SIZE_ASSERT(DeviceTextureObject, 0x10u);

struct DeviceTransform {
  vec3 translation;
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
LuminaryResult device_struct_material_convert(const Material* material, DeviceMaterialCompressed* device_material);
LuminaryResult device_struct_scene_entity_convert(const void* source, void* dst, SceneEntity entity);
LuminaryResult device_struct_triangle_convert(const Triangle* triangle, DeviceTriangle* device_triangle);
LuminaryResult device_struct_texture_object_convert(const struct DeviceTexture* texture, DeviceTextureObject* texture_object);
LuminaryResult device_struct_instance_transform_convert(const MeshInstance* instance, DeviceTransform* device_transform);

#endif /* LUMINARY_DEVICE_STRUCTS_H */
