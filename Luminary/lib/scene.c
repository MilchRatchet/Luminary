#include <stdlib.h>
#include <stdio.h>
#include "scene.h"
#include "raytrace.h"
#include "error.h"
#include "png.h"
#include "wavefront.h"
#include "light.h"

static const int LINE_SIZE       = 4096;
static const int CURRENT_VERSION = 2;

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

Scene load_scene(const char* filename, raytrace_instance** instance, char** output_name) {
  FILE* file = fopen(filename, "rb");

  assert(file, "Scene file could not be opened", 1);

  char* line = (char*) malloc(LINE_SIZE);

  fgets(line, LINE_SIZE, file);

  assert(!validate_filetype(line), "Scene file is not a Luminary scene file!", 1);

  fgets(line, LINE_SIZE, file);

  if (line[0] == 'v') {
    int version = 0;
    sscanf(line, "%*c %d\n", &version);
    assert(
      version == CURRENT_VERSION,
      "Incompatible Scene version! Update the file or use an older version of Luminary!", 1);
  }
  else {
    print_error("Scene file has no version information, assuming version!")
  }

  Scene scene;

  int width           = 1280;
  int height          = 720;
  int bounces         = 5;
  int samples         = 16;
  float azimuth       = 3.141f;
  float altitude      = 0.5f;
  float sun_strength  = 30.0f;
  int bvh_depth       = 18;
  float clip_distance = 1000000.0f;

  int denoiser = 1;

  Wavefront_Content content = create_wavefront_content();

  char* source = (char*) malloc(LINE_SIZE);

  while (!feof(file)) {
    fgets(line, LINE_SIZE, file);

    if (line[0] == 'm' && line[1] == ' ') {
      sscanf(line, "%*c %s\n", source);
      if (read_wavefront_file(source, &content)) {
        print_error("Mesh file could not be loaded!");
      }
    }
    else if (line[0] == 'c' && line[1] == ' ') {
      sscanf(
        line, "%*c %f %f %f %f %f %f %f\n", &scene.camera.pos.x, &scene.camera.pos.y,
        &scene.camera.pos.z, &scene.camera.rotation.x, &scene.camera.rotation.y,
        &scene.camera.rotation.z, &scene.camera.fov);
    }
    else if (line[0] == 's' && line[1] == ' ') {
      sscanf(line, "%*c %f %f %f\n", &azimuth, &altitude, &sun_strength);
    }
    else if (line[0] == 'i' && line[1] == ' ') {
      sscanf(line, "%*c %d %d %d %d\n", &width, &height, &bounces, &samples);
    }
    else if (line[0] == 'o' && line[1] == ' ') {
      sscanf(line, "%*c %s\n", *output_name);
    }
    else if (line[0] == 'b') {
      sscanf(line, "%*c %d\n", &bvh_depth);
    }
    else if (line[0] == 'f') {
      sscanf(line, "%*c %f\n", &clip_distance);
    }
    else if (line[0] == 'd') {
      sscanf(line, "%*c %d\n", &denoiser);
    }
    else if (line[0] == '#') {
      continue;
    }
    else if (line[0] == 'x') {
      break;
    }
    else {
      print_error("Scene file contains illegal lines!");
    }
  }

  Triangle* triangles;

  unsigned int triangle_count = convert_wavefront_content(&triangles, content);

  int nodes_length;

  Node* nodes = build_bvh_structure(&triangles, triangle_count, bvh_depth, &nodes_length);

  scene.triangles           = triangles;
  scene.triangles_length    = triangle_count;
  scene.nodes               = nodes;
  scene.nodes_length        = nodes_length;
  scene.materials_length    = content.materials_length;
  scene.texture_assignments = get_texture_assignments(content);
  scene.far_clip_distance   = clip_distance;

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

void free_scene(Scene scene, raytrace_instance* instance) {
  free_textures(instance->albedo_atlas, instance->albedo_atlas_length);
  free_textures(instance->illuminance_atlas, instance->illuminance_atlas_length);
  free_textures(instance->material_atlas, instance->material_atlas_length);

  free(scene.triangles);
  free(scene.nodes);
  free(scene.lights);
}