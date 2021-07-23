#include <stdlib.h>
#include <stdio.h>
#include <immintrin.h>
#include <math.h>
#include "scene.h"
#include "raytrace.h"
#include "error.h"
#include "png.h"
#include "wavefront.h"
#include "light.h"

static const int LINE_SIZE       = 4096;
static const int CURRENT_VERSION = 3;

static int validate_filetype(const char* line) {
  int result = 0;

  result += line[0] ^ 'L';
  result += line[1] ^ 'u';
  result += line[2] ^ 'm';
  result += line[3] ^ 'i';
  result += line[4] ^ 'n';
  result += line[5] ^ 'a';
  result += line[6] ^ 'r';
  result += line[7] ^ 'y';

  return result;
}

Scene load_scene(const char* filename, RaytraceInstance** instance, char** output_name) {
  FILE* file;
  fopen_s(&file, filename, "rb");

  char* line = (char*) malloc(LINE_SIZE);

  sprintf(line, "Scene file \"%s\" could not be opened.", filename);

  assert((unsigned long long) file, line, 1);

  fgets(line, LINE_SIZE, file);

  assert(!validate_filetype(line), "Scene file is not a Luminary scene file!", 1);

  fgets(line, LINE_SIZE, file);

  if (line[0] == 'v') {
    int version = 0;
    sscanf_s(line, "%*c %d\n", &version);
    assert(
      version == CURRENT_VERSION,
      "Incompatible Scene version! Update the file or use an older version of Luminary!", 1);
  }
  else {
    print_error("Scene file has no version information, assuming version!")
  }

  Scene scene;

  scene.camera.pos.x             = 0.0f;
  scene.camera.pos.y             = 0.0f;
  scene.camera.pos.z             = 0.0f;
  scene.camera.rotation.x        = 0.0f;
  scene.camera.rotation.y        = 0.0f;
  scene.camera.rotation.z        = 0.0f;
  scene.camera.fov               = 1.0f;
  scene.camera.focal_length      = 1.0f;
  scene.camera.aperture_size     = 0.00f;
  scene.camera.exposure          = 1.0f;
  scene.camera.auto_exposure     = 0;
  scene.camera.alpha_cutoff      = 0.0f;
  scene.camera.far_clip_distance = 1000000.0f;

  scene.ocean.active     = 0;
  scene.ocean.emissive   = 0;
  scene.ocean.height     = 0.0f;
  scene.ocean.amplitude  = 0.6f;
  scene.ocean.frequency  = 0.16f;
  scene.ocean.choppyness = 4.0f;
  scene.ocean.speed      = 0.8f;
  scene.ocean.time       = 0.0f;
  scene.ocean.albedo.r   = 0.0f;
  scene.ocean.albedo.g   = 0.0f;
  scene.ocean.albedo.b   = 0.0f;
  scene.ocean.albedo.a   = 0.9f;

  scene.sky.base_density     = 0.8f;
  scene.sky.rayleigh_falloff = 0.125f;
  scene.sky.mie_falloff      = 0.833333f;

  int width          = 1280;
  int height         = 720;
  int bounces        = 5;
  int samples        = 16;
  float azimuth      = 3.141f;
  float altitude     = 0.5f;
  float sun_strength = 30.0f;

  int denoiser = 1;

  Wavefront_Content content = create_wavefront_content();

  char* source = (char*) malloc(LINE_SIZE);

  while (!feof(file)) {
    fgets(line, LINE_SIZE, file);

    if (line[0] == 'm' && line[1] == ' ') {
      sscanf_s(line, "%*c %s\n", source, LINE_SIZE);
      if (read_wavefront_file(source, &content)) {
        print_error("Mesh file could not be loaded!");
      }
    }
    else if (line[0] == 'c' && line[1] == ' ') {
      sscanf_s(
        line, "%*c %f %f %f %f %f %f %f\n", &scene.camera.pos.x, &scene.camera.pos.y,
        &scene.camera.pos.z, &scene.camera.rotation.x, &scene.camera.rotation.y,
        &scene.camera.rotation.z, &scene.camera.fov);
    }
    else if (line[0] == 'l' && line[1] == ' ') {
      sscanf_s(
        line, "%*c %f %f %f\n", &scene.camera.focal_length, &scene.camera.aperture_size,
        &scene.camera.exposure);
    }
    else if (line[0] == 's' && line[1] == ' ') {
      sscanf_s(line, "%*c %f %f %f\n", &azimuth, &altitude, &sun_strength);
    }
    else if (line[0] == 'i' && line[1] == ' ') {
      sscanf_s(line, "%*c %d %d %d %d\n", &width, &height, &bounces, &samples);
    }
    else if (line[0] == 'o' && line[1] == ' ') {
      sscanf_s(line, "%*c %s\n", *output_name, 4096);
    }
    else if (line[0] == 'f') {
      sscanf_s(line, "%*c %f\n", &scene.camera.far_clip_distance);
    }
    else if (line[0] == 'w') {
      sscanf_s(
        line, "%*c %d %d %f %f %f %f %f %f %f %f %f\n", &scene.ocean.active, &scene.ocean.emissive,
        &scene.ocean.albedo.r, &scene.ocean.albedo.g, &scene.ocean.albedo.b, &scene.ocean.albedo.a,
        &scene.ocean.height, &scene.ocean.amplitude, &scene.ocean.frequency,
        &scene.ocean.choppyness, &scene.ocean.speed);
    }
    else if (line[0] == 'd') {
      sscanf_s(line, "%*c %d\n", &denoiser);
    }
    else if (line[0] == '#') {
      continue;
    }
    else if (line[0] == 'x') {
      break;
    }
    else {
      sprintf(source, "Scene file contains unknown line!\n Content: %s\n", line);
      print_error(source);
    }
  }

  Triangle* triangles;

  unsigned int triangle_count = convert_wavefront_content(&triangles, content);

  int nodes_length;

  Node2* initial_nodes = build_bvh_structure(&triangles, &triangle_count, &nodes_length);

  Node8* nodes =
    collapse_bvh(initial_nodes, nodes_length, &triangles, triangle_count, &nodes_length);

  free(initial_nodes);

  Traversal_Triangle* traversal_triangles = malloc(sizeof(Traversal_Triangle) * triangle_count);

  for (unsigned int i = 0; i < triangle_count; i++) {
    Triangle triangle     = triangles[i];
    Traversal_Triangle tt = {
      .vertex = {.x = triangle.vertex.x, .y = triangle.vertex.y, .z = triangle.vertex.z},
      .edge1  = {.x = triangle.edge1.x, .y = triangle.edge1.y, .z = triangle.edge1.z},
      .edge2  = {.x = triangle.edge2.x, .y = triangle.edge2.y, .z = triangle.edge2.z}};
    traversal_triangles[i] = tt;
    triangles[i]           = triangle;
  }

  scene.triangles           = triangles;
  scene.traversal_triangles = traversal_triangles;
  scene.triangles_length    = triangle_count;
  scene.nodes               = nodes;
  scene.nodes_length        = nodes_length;
  scene.materials_length    = content.materials_length;
  scene.texture_assignments = get_texture_assignments(content);

  scene.altitude     = altitude;
  scene.azimuth      = azimuth;
  scene.sun_strength = sun_strength;

  process_lights(&scene);

  void* albedo_atlas = initialize_textures(content.albedo_maps, content.albedo_maps_length);
  void* illuminance_atlas =
    initialize_textures(content.illuminance_maps, content.illuminance_maps_length);
  void* material_atlas = initialize_textures(content.material_maps, content.material_maps_length);

  *instance = init_raytracing(
    width, height, bounces, samples, albedo_atlas, content.albedo_maps_length, illuminance_atlas,
    content.illuminance_maps_length, material_atlas, content.material_maps_length, scene, denoiser);

  free(source);
  free(line);

  free_wavefront_content(content);

  fclose(file);

  return scene;
}

void free_scene(Scene scene, RaytraceInstance* instance) {
  free_textures(instance->albedo_atlas, instance->albedo_atlas_length);
  free_textures(instance->illuminance_atlas, instance->illuminance_atlas_length);
  free_textures(instance->material_atlas, instance->material_atlas_length);

  free(scene.triangles);
  free(scene.nodes);
  free(scene.lights);
}
