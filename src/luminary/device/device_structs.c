#include "device_structs.h"

#include <math.h>

#include "device_mesh.h"
#include "device_texture.h"
#include "internal_error.h"

LuminaryResult device_struct_settings_convert(const RendererSettings* settings, DeviceRendererSettings* device_settings) {
  __CHECK_NULL_ARGUMENT(settings);
  __CHECK_NULL_ARGUMENT(device_settings);

  device_settings->max_ray_depth           = settings->max_ray_depth;
  device_settings->shading_mode            = settings->shading_mode;
  device_settings->bridge_max_num_vertices = settings->bridge_max_num_vertices;
  device_settings->bridge_num_ris_samples  = settings->bridge_num_ris_samples;
  device_settings->light_num_rays          = settings->light_num_rays;
  device_settings->light_num_ris_samples   = settings->light_num_ris_samples;

  device_settings->width  = settings->width;
  device_settings->height = settings->height;

  return LUMINARY_SUCCESS;
}

/*
 * Computes a rotation quaternion from euler angles.
 * @param rotation Euler angles defining the rotation.
 * @result Rotation quaternion
 */
static LuminaryResult _device_struct_get_quaternion(Quaternion* q, const vec3 rotation) {
  const float alpha = rotation.x;
  const float beta  = rotation.y;
  const float gamma = rotation.z;

  const float cy = cosf(gamma * 0.5f);
  const float sy = sinf(gamma * 0.5f);
  const float cp = cosf(beta * 0.5f);
  const float sp = sinf(beta * 0.5f);
  const float cr = cosf(alpha * 0.5f);
  const float sr = sinf(alpha * 0.5f);

  q->w = cr * cp * cy + sr * sp * sy;
  q->x = sr * cp * cy - cr * sp * sy;
  q->y = cr * sp * cy + sr * cp * sy;
  q->z = cr * cp * sy - sr * sp * cy;

  return LUMINARY_SUCCESS;
}

LuminaryResult device_struct_camera_convert(const Camera* camera, DeviceCamera* device_camera) {
  __CHECK_NULL_ARGUMENT(camera);
  __CHECK_NULL_ARGUMENT(device_camera);

  device_camera->aperture_shape       = camera->aperture_shape;
  device_camera->aperture_blade_count = camera->aperture_blade_count;
  device_camera->tonemap              = camera->tonemap;
  device_camera->dithering            = camera->dithering;
  device_camera->purkinje             = camera->purkinje;
  device_camera->use_color_correction = camera->use_color_correction;
  device_camera->do_firefly_clamping  = camera->do_firefly_clamping;

  device_camera->pos = camera->pos;
  __FAILURE_HANDLE(_device_struct_get_quaternion(&device_camera->rotation, camera->rotation));
  device_camera->fov                        = camera->fov;
  device_camera->focal_length               = camera->focal_length;
  device_camera->aperture_size              = camera->aperture_size;
  device_camera->exposure                   = camera->exposure;
  device_camera->purkinje_kappa1            = camera->purkinje_kappa1;
  device_camera->purkinje_kappa2            = camera->purkinje_kappa2;
  device_camera->russian_roulette_threshold = camera->russian_roulette_threshold;
  device_camera->film_grain                 = camera->film_grain;

  return LUMINARY_SUCCESS;
}
LuminaryResult device_struct_ocean_convert(const Ocean* ocean, DeviceOcean* device_ocean) {
  __CHECK_NULL_ARGUMENT(ocean);
  __CHECK_NULL_ARGUMENT(device_ocean);

  device_ocean->active                      = ocean->active;
  device_ocean->water_type                  = ocean->water_type;
  device_ocean->caustics_active             = ocean->caustics_active;
  device_ocean->caustics_ris_sample_count   = ocean->caustics_ris_sample_count;
  device_ocean->multiscattering             = ocean->multiscattering;
  device_ocean->triangle_light_contribution = ocean->triangle_light_contribution;

  device_ocean->height                = ocean->height;
  device_ocean->amplitude             = ocean->amplitude;
  device_ocean->frequency             = ocean->frequency;
  device_ocean->choppyness            = ocean->choppyness;
  device_ocean->refractive_index      = ocean->refractive_index;
  device_ocean->caustics_domain_scale = ocean->caustics_domain_scale;

  return LUMINARY_SUCCESS;
}

LuminaryResult device_struct_sky_convert(const Sky* sky, DeviceSky* device_sky) {
  __CHECK_NULL_ARGUMENT(sky);
  __CHECK_NULL_ARGUMENT(device_sky);

  device_sky->ozone_absorption   = sky->ozone_absorption;
  device_sky->steps              = sky->steps;
  device_sky->aerial_perspective = sky->aerial_perspective;
  device_sky->ambient_sampling   = sky->ambient_sampling;
  device_sky->mode               = sky->mode;

  device_sky->geometry_offset        = sky->geometry_offset;
  device_sky->moon_tex_offset        = sky->moon_tex_offset;
  device_sky->sun_strength           = sky->sun_strength;
  device_sky->base_density           = sky->base_density;
  device_sky->stars_intensity        = sky->stars_intensity;
  device_sky->rayleigh_density       = sky->rayleigh_density;
  device_sky->mie_density            = sky->mie_density;
  device_sky->ozone_density          = sky->ozone_density;
  device_sky->rayleigh_falloff       = sky->rayleigh_falloff;
  device_sky->mie_falloff            = sky->mie_falloff;
  device_sky->mie_diameter           = sky->mie_diameter;
  device_sky->ground_visibility      = sky->ground_visibility;
  device_sky->ozone_layer_thickness  = sky->ozone_layer_thickness;
  device_sky->multiscattering_factor = sky->multiscattering_factor;
  device_sky->hdri_origin            = sky->hdri_origin;
  device_sky->hdri_mip_bias          = sky->hdri_mip_bias;
  device_sky->constant_color         = sky->constant_color;

  ////////////////////////////////////////////////////////////////////
  // Precompute sun and moon positions
  ////////////////////////////////////////////////////////////////////

  double sun_x = cos(sky->azimuth) * cos(sky->altitude);
  double sun_y = sin(sky->altitude);
  double sun_z = sin(sky->azimuth) * cos(sky->altitude);

  const double scale_sun = 1.0 / (sqrt(sun_x * sun_x + sun_y * sun_y + sun_z * sun_z));
  sun_x *= scale_sun * SKY_SUN_DISTANCE;
  sun_y *= scale_sun * SKY_SUN_DISTANCE;
  sun_z *= scale_sun * SKY_SUN_DISTANCE;
  sun_y -= SKY_EARTH_RADIUS;
  sun_x -= sky->geometry_offset.x;
  sun_y -= sky->geometry_offset.y;
  sun_z -= sky->geometry_offset.z;

  device_sky->sun_pos.x = sun_x;
  device_sky->sun_pos.y = sun_y;
  device_sky->sun_pos.z = sun_z;

  double moon_x = cos(sky->moon_azimuth) * cos(sky->moon_altitude);
  double moon_y = sin(sky->moon_altitude);
  double moon_z = sin(sky->moon_azimuth) * cos(sky->moon_altitude);

  const double scale_moon = 1.0 / (sqrt(moon_x * moon_x + moon_y * moon_y + moon_z * moon_z));
  moon_x *= scale_moon * SKY_MOON_DISTANCE;
  moon_y *= scale_moon * SKY_MOON_DISTANCE;
  moon_z *= scale_moon * SKY_MOON_DISTANCE;
  moon_y -= SKY_EARTH_RADIUS;
  moon_x -= sky->geometry_offset.x;
  moon_y -= sky->geometry_offset.y;
  moon_z -= sky->geometry_offset.z;

  device_sky->moon_pos.x = moon_x;
  device_sky->moon_pos.y = moon_y;
  device_sky->moon_pos.z = moon_z;

  return LUMINARY_SUCCESS;
}

static LuminaryResult _device_struct_cloud_layer_convert(const CloudLayer* cloud_layer, DeviceCloudLayer* device_cloud_layer) {
  __CHECK_NULL_ARGUMENT(cloud_layer);
  __CHECK_NULL_ARGUMENT(device_cloud_layer);

  device_cloud_layer->height_max   = cloud_layer->height_max;
  device_cloud_layer->height_min   = cloud_layer->height_min;
  device_cloud_layer->coverage     = cloud_layer->coverage;
  device_cloud_layer->coverage_min = cloud_layer->coverage_min;
  device_cloud_layer->type         = cloud_layer->type;
  device_cloud_layer->type_min     = cloud_layer->type_min;
  device_cloud_layer->wind_speed   = cloud_layer->wind_speed;
  device_cloud_layer->wind_angle   = cloud_layer->wind_angle;

  return LUMINARY_SUCCESS;
}

LuminaryResult device_struct_cloud_convert(const Cloud* cloud, DeviceCloud* device_cloud) {
  __CHECK_NULL_ARGUMENT(cloud);
  __CHECK_NULL_ARGUMENT(device_cloud);

  device_cloud->active                = cloud->active;
  device_cloud->atmosphere_scattering = cloud->atmosphere_scattering;
  device_cloud->steps                 = cloud->steps;
  device_cloud->shadow_steps          = cloud->shadow_steps;
  device_cloud->octaves               = cloud->octaves;
  device_cloud->low_active            = cloud->low.active;
  device_cloud->mid_active            = cloud->mid.active;
  device_cloud->top_active            = cloud->top.active;

  device_cloud->offset_x            = cloud->offset_x;
  device_cloud->offset_z            = cloud->offset_z;
  device_cloud->density             = cloud->density;
  device_cloud->droplet_diameter    = cloud->droplet_diameter;
  device_cloud->noise_shape_scale   = cloud->noise_shape_scale;
  device_cloud->noise_detail_scale  = cloud->noise_detail_scale;
  device_cloud->noise_weather_scale = cloud->noise_weather_scale;
  device_cloud->mipmap_bias         = cloud->mipmap_bias;

  __FAILURE_HANDLE(_device_struct_cloud_layer_convert(&cloud->low, &device_cloud->low));
  __FAILURE_HANDLE(_device_struct_cloud_layer_convert(&cloud->mid, &device_cloud->mid));
  __FAILURE_HANDLE(_device_struct_cloud_layer_convert(&cloud->top, &device_cloud->top));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_struct_fog_convert(const Fog* fog, DeviceFog* device_fog) {
  __CHECK_NULL_ARGUMENT(fog);
  __CHECK_NULL_ARGUMENT(device_fog);

  device_fog->active = fog->active;

  device_fog->density          = fog->density;
  device_fog->droplet_diameter = fog->droplet_diameter;
  device_fog->height           = fog->height;
  device_fog->dist             = fog->dist;

  return LUMINARY_SUCCESS;
}

LuminaryResult device_struct_particles_convert(const Particles* particles, DeviceParticles* device_particles) {
  __CHECK_NULL_ARGUMENT(particles);
  __CHECK_NULL_ARGUMENT(device_particles);

  device_particles->active = particles->active;
  device_particles->count  = particles->count;

  device_particles->albedo             = particles->albedo;
  device_particles->speed              = particles->speed;
  device_particles->direction_altitude = particles->direction_altitude;
  device_particles->direction_azimuth  = particles->direction_azimuth;
  device_particles->phase_diameter     = particles->phase_diameter;
  device_particles->scale              = particles->scale;
  device_particles->size               = particles->size;
  device_particles->size_variation     = particles->size_variation;

  return LUMINARY_SUCCESS;
}

LuminaryResult device_struct_toy_convert(const Toy* toy, DeviceToy* device_toy) {
  __CHECK_NULL_ARGUMENT(toy);
  __CHECK_NULL_ARGUMENT(device_toy);

  device_toy->active   = toy->active;
  device_toy->emissive = toy->emissive;

  device_toy->position         = toy->position;
  device_toy->rotation         = toy->rotation;
  device_toy->scale            = toy->scale;
  device_toy->refractive_index = toy->refractive_index;
  device_toy->albedo           = toy->albedo;
  device_toy->material         = toy->material;
  device_toy->emission         = toy->emission;

  return LUMINARY_SUCCESS;
}

static uint16_t _device_struct_convert_float01_to_uint16(const float f) {
  return (uint16_t) (f * 0xFFFFu + 0.5f);
}

LuminaryResult device_struct_material_convert(const Material* material, DeviceMaterialCompressed* device_material) {
  __CHECK_NULL_ARGUMENT(material);
  __CHECK_NULL_ARGUMENT(device_material);

  device_material->flags = 0;
  device_material->flags |= material->flags.emission_active ? DEVICE_MATERIAL_FLAG_EMISSION : 0;
  device_material->flags |= material->flags.ior_shadowing ? DEVICE_MATERIAL_FLAG_IOR_SHADOWING : 0;
  device_material->flags |= material->flags.thin_walled ? DEVICE_MATERIAL_FLAG_THIN_WALLED : 0;
  device_material->flags |= material->flags.colored_transparency ? DEVICE_MATERIAL_FLAG_COLORED_TRANSPARENCY : 0;

  device_material->roughness_clamp = _device_struct_convert_float01_to_uint16(material->roughness_clamp) >> 8;

  device_material->metallic         = _device_struct_convert_float01_to_uint16(material->metallic);
  device_material->roughness        = _device_struct_convert_float01_to_uint16(material->roughness);
  device_material->refraction_index = _device_struct_convert_float01_to_uint16(0.5f * (material->refraction_index - 1.0f));

  RGBF emission                 = material->emission;
  const uint16_t emission_scale = (uint16_t) fminf(fmaxf(fmaxf(emission.r, emission.g), emission.b) + 1.0f, (float) 0xFFFFu);
  emission.r /= (float) emission_scale;
  emission.g /= (float) emission_scale;
  emission.b /= (float) emission_scale;

  device_material->albedo_r       = _device_struct_convert_float01_to_uint16(material->albedo.r);
  device_material->albedo_g       = _device_struct_convert_float01_to_uint16(material->albedo.g);
  device_material->albedo_b       = _device_struct_convert_float01_to_uint16(material->albedo.b);
  device_material->albedo_a       = _device_struct_convert_float01_to_uint16(material->albedo.a);
  device_material->emission_r     = _device_struct_convert_float01_to_uint16(emission.r);
  device_material->emission_g     = _device_struct_convert_float01_to_uint16(emission.g);
  device_material->emission_b     = _device_struct_convert_float01_to_uint16(emission.b);
  device_material->emission_scale = emission_scale;
  device_material->albedo_tex     = material->albedo_tex;
  device_material->luminance_tex  = material->luminance_tex;
  device_material->material_tex   = material->material_tex;
  device_material->normal_tex     = material->normal_tex;

  return LUMINARY_SUCCESS;
}

LuminaryResult device_struct_scene_entity_convert(const void* restrict source, void* restrict dst, SceneEntity entity) {
  __CHECK_NULL_ARGUMENT(source);
  __CHECK_NULL_ARGUMENT(dst);

  switch (entity) {
    case SCENE_ENTITY_SETTINGS:
      __FAILURE_HANDLE(device_struct_settings_convert(source, dst));
      break;
    case SCENE_ENTITY_CAMERA:
      __FAILURE_HANDLE(device_struct_camera_convert(source, dst));
      break;
    case SCENE_ENTITY_OCEAN:
      __FAILURE_HANDLE(device_struct_ocean_convert(source, dst));
      break;
    case SCENE_ENTITY_SKY:
      __FAILURE_HANDLE(device_struct_sky_convert(source, dst));
      break;
    case SCENE_ENTITY_CLOUD:
      __FAILURE_HANDLE(device_struct_cloud_convert(source, dst));
      break;
    case SCENE_ENTITY_FOG:
      __FAILURE_HANDLE(device_struct_fog_convert(source, dst));
      break;
    case SCENE_ENTITY_PARTICLES:
      __FAILURE_HANDLE(device_struct_particles_convert(source, dst));
      break;
    case SCENE_ENTITY_TOY:
      __FAILURE_HANDLE(device_struct_toy_convert(source, dst));
      break;
    case SCENE_ENTITY_MATERIALS:
      __FAILURE_HANDLE(device_struct_material_convert(source, dst));
      break;
    default:
      __RETURN_ERROR(LUMINARY_ERROR_NOT_IMPLEMENTED, "Scene entity does not support conversion to device format yet.");
  }

  return LUMINARY_SUCCESS;
}

// Octahedron encoding, for example: https://www.shadertoy.com/view/clXXD8
static uint32_t _device_vec3_to_uint(const vec3 normal) {
  double x = normal.x;
  double y = normal.y;
  double z = normal.z;

  const double recip_norm = 1.0 / (x * x + y * y + z * z);

  x *= recip_norm;
  y *= recip_norm;
  z *= recip_norm;

  const double t = fmax(fmin(-z, 1.0f), 0.0f);

  x += (x >= 0.0) ? t : -t;
  y += (y >= 0.0) ? t : -t;

  const uint32_t x_u16 = (uint16_t) round(x * 0x7FFF);
  const uint32_t y_u16 = (uint16_t) round(y * 0x7FFF);

  return (x_u16 << 16) | y_u16;
}

/*
 * Each component gets 1 sign bit, 8 exponent bit and 7 mantissa bits.
 */
static uint32_t _device_UV_to_uint(const UV uv) {
  const uint32_t u = *((uint32_t*) (&uv.u));
  const uint32_t v = *((uint32_t*) (&uv.v));

  const uint32_t compressed = (u & 0xFFFF0000) | (v >> 16);

  return compressed;
}

LuminaryResult device_struct_triangle_convert(const Triangle* triangle, DeviceTriangle* device_triangle) {
  __CHECK_NULL_ARGUMENT(triangle);
  __CHECK_NULL_ARGUMENT(device_triangle);

  device_triangle->vertex = triangle->vertex;
  device_triangle->edge1  = triangle->edge1;
  device_triangle->edge2  = triangle->edge2;

  device_triangle->vertex_normal = _device_vec3_to_uint(triangle->vertex_normal);
  device_triangle->edge1_normal  = _device_vec3_to_uint(triangle->edge1_normal);
  device_triangle->edge2_normal  = _device_vec3_to_uint(triangle->edge2_normal);

  device_triangle->vertex_texture = _device_UV_to_uint(triangle->vertex_texture);
  device_triangle->edge1_texture  = _device_UV_to_uint(triangle->edge1_texture);
  device_triangle->edge2_texture  = _device_UV_to_uint(triangle->edge2_texture);

  device_triangle->material_id = triangle->material_id;
  device_triangle->padding     = 0;

  return LUMINARY_SUCCESS;
}

LuminaryResult device_struct_texture_object_convert(const struct DeviceTexture* texture, DeviceTextureObject* texture_object) {
  __CHECK_NULL_ARGUMENT(texture);
  __CHECK_NULL_ARGUMENT(texture_object);

  texture_object->handle  = texture->tex;
  texture_object->gamma   = texture->gamma;
  texture_object->padding = 0.0f;

  return LUMINARY_SUCCESS;
}

LuminaryResult _device_struct_quaternion_to_quaternion16(Quaternion16* dst, const Quaternion* src) {
  __CHECK_NULL_ARGUMENT(dst);
  __CHECK_NULL_ARGUMENT(src);

  const int64_t x = (int64_t) ((src->x * 0x7FFF) + 0.5f);
  dst->x          = (int16_t) ((x >> 48) | (x & 0x7FFF));

  const int64_t y = (int64_t) ((src->y * 0x7FFF) + 0.5f);
  dst->y          = (int16_t) ((y >> 48) | (y & 0x7FFF));

  const int64_t z = (int64_t) ((src->z * 0x7FFF) + 0.5f);
  dst->z          = (int16_t) ((z >> 48) | (z & 0x7FFF));

  const int64_t w = (int64_t) ((src->w * 0x7FFF) + 0.5f);
  dst->w          = (int16_t) ((w >> 48) | (w & 0x7FFF));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_struct_instance_transform_convert(const MeshInstance* instance, DeviceTransform* device_transform) {
  __CHECK_NULL_ARGUMENT(instance);
  __CHECK_NULL_ARGUMENT(device_transform);

  device_transform->offset = instance->offset;
  device_transform->scale  = instance->scale;
  __FAILURE_HANDLE(_device_struct_quaternion_to_quaternion16(&device_transform->rotation, &instance->rotation));

  return LUMINARY_SUCCESS;
}
