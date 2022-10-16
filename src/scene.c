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
#include "utils.h"
#include "wavefront.h"

static const int LINE_SIZE = 4096;

static Scene get_default_scene() {
  Scene scene;

  memset(&scene, 0, sizeof(Scene));

  scene.material.lights_active      = 0;
  scene.material.default_material.r = 0.3f;
  scene.material.default_material.g = 0.0f;
  scene.material.default_material.b = 1.0f;
  scene.material.fresnel            = FDEZ_AGUERA;
  scene.material.bvh_alpha_cutoff   = 1;
  scene.material.alpha_cutoff       = 0.0f;

  scene.camera.pos.x                 = 0.0f;
  scene.camera.pos.y                 = 0.0f;
  scene.camera.pos.z                 = 0.0f;
  scene.camera.rotation.x            = 0.0f;
  scene.camera.rotation.y            = 0.0f;
  scene.camera.rotation.z            = 0.0f;
  scene.camera.fov                   = 1.0f;
  scene.camera.focal_length          = 1.0f;
  scene.camera.aperture_size         = 0.00f;
  scene.camera.exposure              = 1.0f;
  scene.camera.auto_exposure         = 1;
  scene.camera.bloom                 = 1;
  scene.camera.bloom_strength        = 1.0f;
  scene.camera.bloom_threshold       = 1.0f;
  scene.camera.dithering             = 1;
  scene.camera.far_clip_distance     = 50000.0f;
  scene.camera.tonemap               = TONEMAP_ACES;
  scene.camera.filter                = FILTER_NONE;
  scene.camera.wasd_speed            = 1.0f;
  scene.camera.mouse_speed           = 1.0f;
  scene.camera.smooth_movement       = 0;
  scene.camera.smoothing_factor      = 0.1f;
  scene.camera.temporal_blend_factor = 0.15f;
  scene.camera.purkinje              = 1;
  scene.camera.purkinje_kappa1       = 0.2f;
  scene.camera.purkinje_kappa2       = 0.29f;

  scene.ocean.active              = 0;
  scene.ocean.emissive            = 0;
  scene.ocean.update              = 0;
  scene.ocean.height              = 0.0f;
  scene.ocean.amplitude           = 0.2f;
  scene.ocean.frequency           = 0.12f;
  scene.ocean.choppyness          = 4.0f;
  scene.ocean.speed               = 1.0f;
  scene.ocean.time                = 0.0f;
  scene.ocean.albedo.r            = 0.0f;
  scene.ocean.albedo.g            = 0.0f;
  scene.ocean.albedo.b            = 0.0f;
  scene.ocean.albedo.a            = 0.9f;
  scene.ocean.refractive_index    = 1.333f;
  scene.ocean.anisotropy          = 0.0f;
  scene.ocean.scattering.r        = 0.0f;
  scene.ocean.scattering.g        = 0.2f;
  scene.ocean.scattering.b        = 1.0f;
  scene.ocean.absorption.r        = 1.0f;
  scene.ocean.absorption.g        = 0.15f;
  scene.ocean.absorption.b        = 0.01f;
  scene.ocean.pollution           = 0.5f;
  scene.ocean.absorption_strength = 1.0f;

  scene.toy.active           = 0;
  scene.toy.emissive         = 0;
  scene.toy.shape            = TOY_SPHERE;
  scene.toy.position.x       = 0.0f;
  scene.toy.position.y       = 10.0f;
  scene.toy.position.z       = 0.0f;
  scene.toy.rotation.x       = 0.0f;
  scene.toy.rotation.y       = 0.0f;
  scene.toy.rotation.z       = 0.0f;
  scene.toy.scale            = 1.0f;
  scene.toy.refractive_index = 1.0f;
  scene.toy.albedo.r         = 0.9f;
  scene.toy.albedo.g         = 0.9f;
  scene.toy.albedo.b         = 0.9f;
  scene.toy.albedo.a         = 1.0f;
  scene.toy.material.r       = 0.3f;
  scene.toy.material.g       = 0.0f;
  scene.toy.material.b       = 1.0f;
  scene.toy.material.a       = 0.0f;
  scene.toy.emission.r       = 0.0f;
  scene.toy.emission.g       = 0.0f;
  scene.toy.emission.b       = 0.0f;
  scene.toy.emission.a       = 0.0f;
  scene.toy.flashlight_mode  = 0;

  scene.sky.geometry_offset.x         = 0.0f;
  scene.sky.geometry_offset.y         = 0.1f;
  scene.sky.geometry_offset.z         = 0.0f;
  scene.sky.sun_color.r               = 1.0f;
  scene.sky.sun_color.g               = 0.9f;
  scene.sky.sun_color.b               = 0.8f;
  scene.sky.altitude                  = 0.5f;
  scene.sky.azimuth                   = 3.141f;
  scene.sky.moon_altitude             = -0.5f;
  scene.sky.moon_azimuth              = 0.0f;
  scene.sky.moon_albedo               = 0.12f;
  scene.sky.sun_strength              = 15000.0f;
  scene.sky.base_density              = 1.0f;
  scene.sky.steps                     = 8;
  scene.sky.shadow_steps              = 8;
  scene.sky.ozone_absorption          = 1;
  scene.sky.stars_seed                = 0;
  scene.sky.stars_intensity           = 1.0f;
  scene.sky.settings_stars_count      = 10000;
  scene.sky.cloud.active              = 0;
  scene.sky.cloud.initialized         = 0;
  scene.sky.cloud.seed                = 1;
  scene.sky.cloud.offset_x            = 0.0f;
  scene.sky.cloud.offset_z            = 0.0f;
  scene.sky.cloud.height_max          = 4000.0f;
  scene.sky.cloud.height_min          = 1500.0f;
  scene.sky.cloud.noise_shape_scale   = 1.0f;
  scene.sky.cloud.noise_detail_scale  = 1.0f;
  scene.sky.cloud.noise_weather_scale = 1.0f;
  scene.sky.cloud.noise_curl_scale    = 1.0f;
  scene.sky.cloud.coverage            = 1.0f;
  scene.sky.cloud.anvil               = 0.0f;
  scene.sky.cloud.coverage_min        = 1.05f;
  scene.sky.cloud.forward_scattering  = 0.8f;
  scene.sky.cloud.backward_scattering = -0.2f;
  scene.sky.cloud.lobe_lerp           = 0.5f;
  scene.sky.cloud.wetness             = 0.0f;
  scene.sky.cloud.powder              = 0.5f;
  scene.sky.cloud.shadow_steps        = 16;
  scene.sky.cloud.density             = 1.0f;

  scene.fog.active     = 0;
  scene.fog.density    = 1.0f;
  scene.fog.anisotropy = 0.0f;
  scene.fog.height     = 500.0f;
  scene.fog.dist       = 500.0f;

  return scene;
}

static General get_default_settings() {
  General general = {
    .width             = 1280,
    .height            = 720,
    .max_ray_depth     = 3,
    .samples           = 16,
    .denoiser          = DENOISING_ON,
    .reservoir_size    = 8,
    .output_path       = malloc(LINE_SIZE),
    .mesh_files        = malloc(sizeof(char*) * 10),
    .mesh_files_count  = 0,
    .mesh_files_length = 10};

  return general;
}

static void convert_wavefront_to_internal(Wavefront_Content content, Scene* scene) {
  scene->triangles_length = convert_wavefront_content(&scene->triangles, content);

  Node2* initial_nodes = build_bvh_structure(&scene->triangles, &scene->triangles_length, &scene->nodes_length);

  if (!scene->triangles_length) {
    crash_message("No triangles are left. Did the scene not contain any faces?");
  }

  scene->nodes = collapse_bvh(initial_nodes, scene->nodes_length, &scene->triangles, scene->triangles_length, &scene->nodes_length);

  free(initial_nodes);

  sort_traversal_elements(&scene->nodes, scene->nodes_length, &scene->triangles, scene->triangles_length);

  scene->materials_length    = content.materials_length;
  scene->texture_assignments = get_texture_assignments(content);

  scene->traversal_triangles = malloc(sizeof(TraversalTriangle) * scene->triangles_length);

  for (unsigned int i = 0; i < scene->triangles_length; i++) {
    const Triangle triangle   = scene->triangles[i];
    const uint32_t albedo_tex = scene->texture_assignments[triangle.object_maps].albedo_map;
    TraversalTriangle tt      = {
           .vertex     = {.x = triangle.vertex.x, .y = triangle.vertex.y, .z = triangle.vertex.z},
           .edge1      = {.x = triangle.edge1.x, .y = triangle.edge1.y, .z = triangle.edge1.z},
           .edge2      = {.x = triangle.edge2.x, .y = triangle.edge2.y, .z = triangle.edge2.z},
           .albedo_tex = albedo_tex};
    scene->traversal_triangles[i] = tt;
  }
}

RaytraceInstance* load_scene(const char* filename) {
  FILE* file = fopen(filename, "rb");

  if (!file) {
    crash_message("Scene file \"%s\" could not be opened.", filename);
    return (RaytraceInstance*) 0;
  }

  if (lum_validate_file(file)) {
    crash_message("Scene file \"%s\" is not a supported *.lum file.", filename);
    return (RaytraceInstance*) 0;
  }

  Scene scene     = get_default_scene();
  General general = get_default_settings();

  strcpy(general.output_path, "output");

  Wavefront_Content content = create_wavefront_content();

  lum_parse_file(file, &scene, &general, &content);

  fclose(file);

  assert(general.mesh_files_count, "No mesh files where loaded.", 1);

  convert_wavefront_to_internal(content, &scene);

  process_lights(&scene, content.illuminance_maps);

  DeviceBuffer* albedo_atlas      = cudatexture_allocate_to_buffer(content.albedo_maps, content.albedo_maps_length);
  DeviceBuffer* illuminance_atlas = cudatexture_allocate_to_buffer(content.illuminance_maps, content.illuminance_maps_length);
  DeviceBuffer* material_atlas    = cudatexture_allocate_to_buffer(content.material_maps, content.material_maps_length);

  RaytraceInstance* instance = init_raytracing(
    general, albedo_atlas, content.albedo_maps_length, illuminance_atlas, content.illuminance_maps_length, material_atlas,
    content.material_maps_length, scene);

  free_wavefront_content(content);
  free_scene(scene);

  generate_stars(instance);

  return instance;
}

RaytraceInstance* load_obj_as_scene(char* filename) {
  Scene scene     = get_default_scene();
  General general = get_default_settings();

  general.mesh_files[0] = malloc(LINE_SIZE);
  strcpy(general.mesh_files[0], filename);
  strcpy(general.output_path, "output");

  Wavefront_Content content = create_wavefront_content();

  assert(!read_wavefront_file(filename, &content), "Mesh file could not be loaded.", 1);

  general.mesh_files[general.mesh_files_count++] = filename;

  convert_wavefront_to_internal(content, &scene);

  process_lights(&scene, content.illuminance_maps);

  DeviceBuffer* albedo_atlas      = cudatexture_allocate_to_buffer(content.albedo_maps, content.albedo_maps_length);
  DeviceBuffer* illuminance_atlas = cudatexture_allocate_to_buffer(content.illuminance_maps, content.illuminance_maps_length);
  DeviceBuffer* material_atlas    = cudatexture_allocate_to_buffer(content.material_maps, content.material_maps_length);

  RaytraceInstance* instance = init_raytracing(
    general, albedo_atlas, content.albedo_maps_length, illuminance_atlas, content.illuminance_maps_length, material_atlas,
    content.material_maps_length, scene);

  free_wavefront_content(content);
  free_scene(scene);

  generate_stars(instance);

  return instance;
}

void serialize_scene(RaytraceInstance* instance) {
  FILE* file = fopen("generated.lum", "wb");

  if (!file) {
    error_message("Could not export settings! Failed to open file.");
    return;
  }

  lum_write_file(file, instance);

  fclose(file);
}

void free_atlases(RaytraceInstance* instance) {
  cudatexture_free_buffer(instance->albedo_atlas, instance->albedo_atlas_length);
  cudatexture_free_buffer(instance->illuminance_atlas, instance->illuminance_atlas_length);
  cudatexture_free_buffer(instance->material_atlas, instance->material_atlas_length);
}

void free_strings(RaytraceInstance* instance) {
  for (int i = 0; i < instance->settings.mesh_files_count; i++) {
    free(instance->settings.mesh_files[i]);
  }
  free(instance->settings.mesh_files);
  free(instance->settings.output_path);
}

void free_scene(Scene scene) {
  free(scene.triangles);
  free(scene.traversal_triangles);
  free(scene.nodes);
  free(scene.triangle_lights);
  free(scene.texture_assignments);
}
