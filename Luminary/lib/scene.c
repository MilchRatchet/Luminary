#include <stdlib.h>
#include <stdio.h>
#include "scene.h"
#include "raytrace.h"
#include "error.h"
#include "png.h"
#include "wavefront.h"
#include "light.h"

static const int LINE_SIZE       = 4096;
static const int CURRENT_VERSION = 1;

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

  Wavefront_Mesh* meshes;
  int meshes_length = 0;

  int assignments_length = 0;

  texture_assignment* texture_assignments =
    (texture_assignment*) malloc(sizeof(texture_assignment) * assignments_length);

  Scene scene;

  int width          = 1280;
  int height         = 720;
  int bounces        = 5;
  int samples        = 16;
  float azimuth      = 3.141f;
  float altitude     = 0.5f;
  float sun_strength = 30.0f;
  int bvh_depth      = 18;

  unsigned int albedo_maps_count      = 1;
  unsigned int illuminance_maps_count = 1;
  unsigned int material_maps_count    = 1;

  TextureRGBA* albedo_maps = (TextureRGBA*) malloc(sizeof(TextureRGBA) * albedo_maps_count);
  TextureRGBA* illuminance_maps =
    (TextureRGBA*) malloc(sizeof(TextureRGBA) * illuminance_maps_count);
  TextureRGBA* material_maps = (TextureRGBA*) malloc(sizeof(TextureRGBA) * material_maps_count);

  albedo_maps[0].width          = 1;
  albedo_maps[0].height         = 1;
  albedo_maps[0].data           = (RGBAF*) malloc(sizeof(RGBAF));
  albedo_maps[0].data[0].r      = 0.9f;
  albedo_maps[0].data[0].g      = 0.9f;
  albedo_maps[0].data[0].b      = 0.9f;
  albedo_maps[0].data[0].a      = 1.0f;
  illuminance_maps[0].width     = 1;
  illuminance_maps[0].height    = 1;
  illuminance_maps[0].data      = (RGBAF*) malloc(sizeof(RGBAF));
  illuminance_maps[0].data[0].r = 0.0f;
  illuminance_maps[0].data[0].g = 0.0f;
  illuminance_maps[0].data[0].b = 0.0f;
  illuminance_maps[0].data[0].a = 0.0f;
  material_maps[0].width        = 1;
  material_maps[0].height       = 1;
  material_maps[0].data         = (RGBAF*) malloc(sizeof(RGBAF));
  material_maps[0].data[0].r    = 0.2f;
  material_maps[0].data[0].g    = 0.0f;
  material_maps[0].data[0].b    = 1.0f / 255.0f;
  material_maps[0].data[0].a    = 0.0f;

  char* source = (char*) malloc(LINE_SIZE);

  while (!feof(file)) {
    fgets(line, LINE_SIZE, file);

    if (line[0] == 'm' && line[1] == ' ') {
      sscanf(line, "%*c %s\n", source);
      meshes_length = read_mesh_from_file(source, &meshes, meshes_length);
      if (meshes_length > assignments_length) {
        const int old_length = assignments_length;
        assignments_length   = meshes_length;
        texture_assignments  = (texture_assignment*) realloc(
          texture_assignments, sizeof(texture_assignment) * assignments_length);

        for (int i = old_length; i < assignments_length; i++) {
          texture_assignments[i].albedo_map      = 0;
          texture_assignments[i].illuminance_map = 0;
          texture_assignments[i].material_map    = 0;
        }
      }
    }
    else if (line[0] == 't' && line[1] == 'a') {
      sscanf(line, "%*2c %s\n", source);
      albedo_maps_count++;
      albedo_maps = (TextureRGBA*) realloc(albedo_maps, sizeof(TextureRGBA) * albedo_maps_count);
      albedo_maps[albedo_maps_count - 1] = load_texture_from_png(source);
    }
    else if (line[0] == 't' && line[1] == 'i') {
      sscanf(line, "%*2c %s\n", source);
      illuminance_maps_count++;
      illuminance_maps =
        (TextureRGBA*) realloc(illuminance_maps, sizeof(TextureRGBA) * illuminance_maps_count);
      illuminance_maps[illuminance_maps_count - 1] = load_texture_from_png(source);
    }
    else if (line[0] == 't' && line[1] == 'm') {
      sscanf(line, "%*2c %s\n", source);
      material_maps_count++;
      material_maps =
        (TextureRGBA*) realloc(material_maps, sizeof(TextureRGBA) * material_maps_count);
      material_maps[material_maps_count - 1] = load_texture_from_png(source);
    }
    else if (line[0] == 'a' && line[1] == ' ') {
      int mesh;
      int albedo;
      int illuminance;
      int material;
      sscanf(line, "%*c %d %d %d %d\n", &mesh, &albedo, &illuminance, &material);

      if (assignments_length <= mesh) {
        const int old_length = assignments_length;
        assignments_length   = mesh + 1;
        texture_assignments  = (texture_assignment*) realloc(
          texture_assignments, sizeof(texture_assignment) * assignments_length);

        for (int i = old_length; i < assignments_length - 1; i++) {
          texture_assignments[i].albedo_map      = 0;
          texture_assignments[i].illuminance_map = 0;
          texture_assignments[i].material_map    = 0;
        }
      }
      texture_assignments[mesh].albedo_map      = albedo;
      texture_assignments[mesh].illuminance_map = illuminance;
      texture_assignments[mesh].material_map    = material;
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

  unsigned int triangle_count = convert_wavefront_mesh(&triangles, meshes, meshes_length);

  int nodes_length;

  Node* nodes = build_bvh_structure(&triangles, triangle_count, bvh_depth, &nodes_length);

  scene.triangles           = triangles;
  scene.triangles_length    = triangle_count;
  scene.nodes               = nodes;
  scene.nodes_length        = nodes_length;
  scene.meshes_length       = meshes_length;
  scene.texture_assignments = texture_assignments;

  scene.altitude     = altitude;
  scene.azimuth      = azimuth;
  scene.sun_strength = sun_strength;

  process_lights(&scene);

  void* albedo_atlas      = initialize_textures(albedo_maps, albedo_maps_count);
  void* illuminance_atlas = initialize_textures(illuminance_maps, illuminance_maps_count);
  void* material_atlas    = initialize_textures(material_maps, material_maps_count);

  *instance = init_raytracing(
    width, height, bounces, samples, albedo_atlas, albedo_maps_count, illuminance_atlas,
    illuminance_maps_count, material_atlas, material_maps_count, scene);

  free(source);
  free(line);

  for (int i = 0; i < scene.meshes_length; i++) {
    free(meshes[i].triangles);
    free(meshes[i].normals);
    free(meshes[i].uvs);
    free(meshes[i].vertices);
  }

  free(meshes);

  for (int i = 0; i < albedo_maps_count; i++) {
    free(albedo_maps[i].data);
  }
  for (int i = 0; i < illuminance_maps_count; i++) {
    free(illuminance_maps[i].data);
  }
  for (int i = 0; i < material_maps_count; i++) {
    free(material_maps[i].data);
  }

  fclose(file);

  return scene;
}

void free_scene(Scene scene, raytrace_instance* instance) {
  free_textures(instance->albedo_atlas, instance->albedo_atlas_length);
  free_textures(instance->illuminance_atlas, instance->illuminance_atlas_length);
  free_textures(instance->material_atlas, instance->material_atlas_length);

  free(scene.triangles);
  free(scene.nodes);
}
