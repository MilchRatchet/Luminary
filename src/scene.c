#include "scene.h"

#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "light.h"
#include "log.h"
#include "lum.h"
#include "png.h"
#include "raytrace.h"
#include "stars.h"
#include "texture.h"
#include "utils.h"
#include "wavefront.h"

static const int LINE_SIZE = 4096;

void scene_init(Scene** _scene) {
  Scene* scene = calloc(1, sizeof(Scene));

  scene->material.lights_active           = 0;
  scene->material.light_tree_active       = 1;
  scene->material.default_material.r      = 0.3f;
  scene->material.default_material.g      = 0.0f;
  scene->material.default_material.b      = 1.0f;
  scene->material.alpha_cutoff            = 0.0f;
  scene->material.colored_transparency    = 0;
  scene->material.override_materials      = 0;
  scene->material.invert_roughness        = 0;
  scene->material.enable_ior_shadowing    = 1;
  scene->material.caustic_roughness_clamp = 0.25f;

  scene->camera.pos.x                      = 0.0f;
  scene->camera.pos.y                      = 0.0f;
  scene->camera.pos.z                      = 0.0f;
  scene->camera.rotation.x                 = 0.0f;
  scene->camera.rotation.y                 = 0.0f;
  scene->camera.rotation.z                 = 0.0f;
  scene->camera.fov                        = 1.0f;
  scene->camera.focal_length               = 1.0f;
  scene->camera.aperture_size              = 0.0f;
  scene->camera.aperture_shape             = CAMERA_APERTURE_ROUND;
  scene->camera.aperture_blade_count       = 7;
  scene->camera.exposure                   = 1.0f;
  scene->camera.min_exposure               = 40.0f;
  scene->camera.max_exposure               = 300.0f;
  scene->camera.auto_exposure              = 1;
  scene->camera.bloom                      = 1;
  scene->camera.bloom_blend                = 0.01f;
  scene->camera.lens_flare                 = 0;
  scene->camera.lens_flare_threshold       = 1.0f;
  scene->camera.dithering                  = 1;
  scene->camera.far_clip_distance          = 50000.0f;
  scene->camera.tonemap                    = TONEMAP_ACES;
  scene->camera.agx_custom_slope           = 1.0f;
  scene->camera.agx_custom_power           = 1.0f;
  scene->camera.agx_custom_saturation      = 1.0f;
  scene->camera.filter                     = FILTER_NONE;
  scene->camera.wasd_speed                 = 1.0f;
  scene->camera.mouse_speed                = 1.0f;
  scene->camera.smooth_movement            = 0;
  scene->camera.smoothing_factor           = 0.1f;
  scene->camera.temporal_blend_factor      = 0.15f;
  scene->camera.purkinje                   = 1;
  scene->camera.purkinje_kappa1            = 0.2f;
  scene->camera.purkinje_kappa2            = 0.29f;
  scene->camera.russian_roulette_threshold = 0.1f;
  scene->camera.use_color_correction       = 0;
  scene->camera.color_correction.r         = 0.0f;
  scene->camera.color_correction.g         = 0.0f;
  scene->camera.color_correction.b         = 0.0f;
  scene->camera.do_firefly_clamping        = 1;
  scene->camera.film_grain                 = 0.0f;

  scene->ocean.active                    = 0;
  scene->ocean.height                    = 0.0f;
  scene->ocean.amplitude                 = 0.2f;
  scene->ocean.frequency                 = 0.12f;
  scene->ocean.choppyness                = 4.0f;
  scene->ocean.refractive_index          = 1.333f;
  scene->ocean.water_type                = JERLOV_WATER_TYPE_IB;
  scene->ocean.caustics_active           = 0;
  scene->ocean.caustics_ris_sample_count = 32;
  scene->ocean.caustics_domain_scale     = 0.5f;
  scene->ocean.multiscattering           = 0;

  scene->toy.active           = 0;
  scene->toy.emissive         = 0;
  scene->toy.shape            = TOY_SPHERE;
  scene->toy.position.x       = 0.0f;
  scene->toy.position.y       = 10.0f;
  scene->toy.position.z       = 0.0f;
  scene->toy.rotation.x       = 0.0f;
  scene->toy.rotation.y       = 0.0f;
  scene->toy.rotation.z       = 0.0f;
  scene->toy.scale            = 1.0f;
  scene->toy.refractive_index = 1.0f;
  scene->toy.albedo.r         = 0.9f;
  scene->toy.albedo.g         = 0.9f;
  scene->toy.albedo.b         = 0.9f;
  scene->toy.albedo.a         = 1.0f;
  scene->toy.material.r       = 0.3f;
  scene->toy.material.g       = 0.0f;
  scene->toy.material.b       = 1.0f;
  scene->toy.material.a       = 0.0f;
  scene->toy.emission.r       = 0.0f;
  scene->toy.emission.g       = 0.0f;
  scene->toy.emission.b       = 0.0f;
  scene->toy.flashlight_mode  = 0;

  scene->sky.geometry_offset.x           = 0.0f;
  scene->sky.geometry_offset.y           = 0.1f;
  scene->sky.geometry_offset.z           = 0.0f;
  scene->sky.altitude                    = 0.5f;
  scene->sky.azimuth                     = 3.141f;
  scene->sky.moon_altitude               = -0.5f;
  scene->sky.moon_azimuth                = 0.0f;
  scene->sky.moon_tex_offset             = 0.0f;
  scene->sky.sun_strength                = 1.0f;
  scene->sky.base_density                = 1.0f;
  scene->sky.rayleigh_density            = 1.0f;
  scene->sky.mie_density                 = 1.0f;
  scene->sky.ozone_density               = 1.0f;
  scene->sky.ground_visibility           = 60.0f;
  scene->sky.mie_diameter                = 2.0f;
  scene->sky.ozone_layer_thickness       = 15.0f;
  scene->sky.rayleigh_falloff            = 8.0f;
  scene->sky.mie_falloff                 = 1.7f;
  scene->sky.multiscattering_factor      = 1.0f;
  scene->sky.steps                       = 40;
  scene->sky.ozone_absorption            = 1;
  scene->sky.aerial_perspective          = 0;
  scene->sky.lut_initialized             = 0;
  scene->sky.hdri_initialized            = 0;
  scene->sky.hdri_dim                    = 0;
  scene->sky.settings_hdri_dim           = 2048;
  scene->sky.hdri_samples                = 50;
  scene->sky.hdri_origin.x               = 0.0f;
  scene->sky.hdri_origin.y               = 1.0f;
  scene->sky.hdri_origin.z               = 0.0f;
  scene->sky.hdri_mip_bias               = 0.0f;
  scene->sky.stars_seed                  = 0;
  scene->sky.stars_intensity             = 1.0f;
  scene->sky.settings_stars_count        = 10000;
  scene->sky.constant_color.r            = 1.0f;
  scene->sky.constant_color.g            = 1.0f;
  scene->sky.constant_color.b            = 1.0f;
  scene->sky.ambient_sampling            = 1;
  scene->sky.mode                        = SKY_MODE_DEFAULT;
  scene->sky.cloud.active                = 0;
  scene->sky.cloud.initialized           = 0;
  scene->sky.cloud.steps                 = 96;
  scene->sky.cloud.shadow_steps          = 8;
  scene->sky.cloud.atmosphere_scattering = 1;
  scene->sky.cloud.seed                  = 1;
  scene->sky.cloud.offset_x              = 0.0f;
  scene->sky.cloud.offset_z              = 0.0f;
  scene->sky.cloud.noise_shape_scale     = 1.0f;
  scene->sky.cloud.noise_detail_scale    = 1.0f;
  scene->sky.cloud.noise_weather_scale   = 1.0f;
  scene->sky.cloud.octaves               = 9;
  scene->sky.cloud.droplet_diameter      = 25.0f;
  scene->sky.cloud.density               = 1.0f;
  scene->sky.cloud.mipmap_bias           = 0.0f;
  scene->sky.cloud.low.active            = 1;
  scene->sky.cloud.low.height_max        = 5.0f;
  scene->sky.cloud.low.height_min        = 1.5f;
  scene->sky.cloud.low.coverage          = 1.0f;
  scene->sky.cloud.low.coverage_min      = 0.0f;
  scene->sky.cloud.low.type              = 1.0f;
  scene->sky.cloud.low.type_min          = 0.0f;
  scene->sky.cloud.low.wind_speed        = 2.5f;
  scene->sky.cloud.low.wind_angle        = 0.0f;
  scene->sky.cloud.mid.active            = 1;
  scene->sky.cloud.mid.height_max        = 6.0f;
  scene->sky.cloud.mid.height_min        = 5.5f;
  scene->sky.cloud.mid.coverage          = 1.0f;
  scene->sky.cloud.mid.coverage_min      = 0.0f;
  scene->sky.cloud.mid.type              = 1.0f;
  scene->sky.cloud.mid.type_min          = 0.0f;
  scene->sky.cloud.mid.wind_speed        = 2.5f;
  scene->sky.cloud.mid.wind_angle        = 0.0f;
  scene->sky.cloud.top.active            = 1;
  scene->sky.cloud.top.height_max        = 8.0f;
  scene->sky.cloud.top.height_min        = 7.95f;
  scene->sky.cloud.top.coverage          = 1.0f;
  scene->sky.cloud.top.coverage_min      = 0.0f;
  scene->sky.cloud.top.type              = 1.0f;
  scene->sky.cloud.top.type_min          = 0.0f;
  scene->sky.cloud.top.wind_speed        = 1.0f;
  scene->sky.cloud.top.wind_angle        = 0.0f;

  scene->fog.active           = 0;
  scene->fog.density          = 1.0f;
  scene->fog.droplet_diameter = 10.0f;
  scene->fog.height           = 500.0f;
  scene->fog.dist             = 500.0f;

  scene->particles.active             = 0;
  scene->particles.scale              = 10.0f;
  scene->particles.albedo.r           = 1.0f;
  scene->particles.albedo.g           = 1.0f;
  scene->particles.albedo.b           = 1.0f;
  scene->particles.direction_altitude = 1.234f;
  scene->particles.direction_azimuth  = 0.0f;
  scene->particles.speed              = 0.0f;
  scene->particles.phase_diameter     = 50.0f;
  scene->particles.seed               = 0;
  scene->particles.count              = 8192;
  scene->particles.size               = 1.0f;
  scene->particles.size_variation     = 0.1f;

  *_scene = scene;
}

static General get_default_settings() {
  General general = {
    .width             = 1280,
    .height            = 720,
    .max_ray_depth     = 8,
    .samples           = 64,
    .num_light_ray     = 1,
    .denoiser          = DENOISING_OFF,
    .output_path       = malloc(LINE_SIZE),
    .mesh_files        = malloc(sizeof(char*) * 10),
    .mesh_files_count  = 0,
    .mesh_files_length = 10};

  return general;
}

void scene_create_from_wavefront(Scene* scene, WavefrontContent* content) {
  wavefront_convert_content(content, &scene->triangles, &scene->triangle_data);

  scene->materials_count = content->materials_count;
  scene->materials       = wavefront_generate_texture_assignments(content);
}

RaytraceInstance* scene_load_lum(const char* filename, CommandlineOptions options) {
  FILE* file = fopen(filename, "rb");

  if (!file) {
    crash_message("Scene file \"%s\" could not be opened.", filename);
    return (RaytraceInstance*) 0;
  }

  if (lum_validate_file(file)) {
    crash_message("Scene file \"%s\" is not a supported *.lum file.", filename);
    return (RaytraceInstance*) 0;
  }

  Scene* scene;
  scene_init(&scene);

  General general = get_default_settings();

  strcpy(general.output_path, "output");

  WavefrontContent* content;
  wavefront_init(&content);

  lum_parse_file(file, scene, &general, content);

  fclose(file);

  assert(general.mesh_files_count, "No mesh files where loaded.", 1);

  scene_create_from_wavefront(scene, content);

  TextureAtlas tex_atlas = {
    .albedo           = (DeviceBuffer*) 0,
    .albedo_length    = content->maps_count[WF_ALBEDO],
    .luminance        = (DeviceBuffer*) 0,
    .luminance_length = content->maps_count[WF_LUMINANCE],
    .material         = (DeviceBuffer*) 0,
    .material_length  = content->maps_count[WF_MATERIAL],
    .normal           = (DeviceBuffer*) 0,
    .normal_length    = content->maps_count[WF_NORMAL]};

  texture_create_atlas(&tex_atlas.albedo, content->maps[WF_ALBEDO], content->maps_count[WF_ALBEDO]);
  texture_create_atlas(&tex_atlas.luminance, content->maps[WF_LUMINANCE], content->maps_count[WF_LUMINANCE]);
  texture_create_atlas(&tex_atlas.material, content->maps[WF_MATERIAL], content->maps_count[WF_MATERIAL]);
  texture_create_atlas(&tex_atlas.normal, content->maps[WF_NORMAL], content->maps_count[WF_NORMAL]);

  RaytraceInstance* instance;
  raytrace_init(&instance, general, tex_atlas, scene, options);

  wavefront_clear(&content);
  scene_clear(&scene);

  return instance;
}

RaytraceInstance* scene_load_obj(char* filename, CommandlineOptions options) {
  Scene* scene;
  scene_init(&scene);

  General general = get_default_settings();

  general.mesh_files[0] = malloc(LINE_SIZE);
  strcpy(general.mesh_files[0], filename);
  strcpy(general.output_path, "output");

  WavefrontContent* content;
  wavefront_init(&content);

  assert(!wavefront_read_file(content, filename), "Mesh file could not be loaded.", 1);

  general.mesh_files[general.mesh_files_count++] = filename;

  scene_create_from_wavefront(scene, content);

  TextureAtlas tex_atlas = {
    .albedo           = (DeviceBuffer*) 0,
    .albedo_length    = content->maps_count[WF_ALBEDO],
    .luminance        = (DeviceBuffer*) 0,
    .luminance_length = content->maps_count[WF_LUMINANCE],
    .material         = (DeviceBuffer*) 0,
    .material_length  = content->maps_count[WF_MATERIAL],
    .normal           = (DeviceBuffer*) 0,
    .normal_length    = content->maps_count[WF_NORMAL]};

  texture_create_atlas(&tex_atlas.albedo, content->maps[WF_ALBEDO], content->maps_count[WF_ALBEDO]);
  texture_create_atlas(&tex_atlas.luminance, content->maps[WF_LUMINANCE], content->maps_count[WF_LUMINANCE]);
  texture_create_atlas(&tex_atlas.material, content->maps[WF_MATERIAL], content->maps_count[WF_MATERIAL]);
  texture_create_atlas(&tex_atlas.normal, content->maps[WF_NORMAL], content->maps_count[WF_NORMAL]);

  RaytraceInstance* instance;
  raytrace_init(&instance, general, tex_atlas, scene, options);

  wavefront_clear(&content);
  scene_clear(&scene);

  return instance;
}

void scene_serialize(RaytraceInstance* instance) {
  FILE* file = fopen("generated.lum", "wb");

  if (!file) {
    error_message("Could not export settings! Failed to open file.");
    return;
  }

  lum_write_file(file, instance);

  fclose(file);
}

void free_atlases(RaytraceInstance* instance) {
  texture_free_atlas(instance->tex_atlas.albedo, instance->tex_atlas.albedo_length);
  texture_free_atlas(instance->tex_atlas.luminance, instance->tex_atlas.luminance_length);
  texture_free_atlas(instance->tex_atlas.material, instance->tex_atlas.material_length);
  texture_free_atlas(instance->tex_atlas.normal, instance->tex_atlas.normal_length);
}

void free_strings(RaytraceInstance* instance) {
  for (int i = 0; i < instance->settings.mesh_files_count; i++) {
    free(instance->settings.mesh_files[i]);
  }
  free(instance->settings.mesh_files);
  free(instance->settings.output_path);
}

void scene_clear(Scene** scene) {
  if (!scene) {
    error_message("Scene is NULL.");
    return;
  }

  free((*scene)->triangles);
  free((*scene)->triangle_lights);
  free((*scene)->materials);

  free(*scene);
}
