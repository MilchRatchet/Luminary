#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "Luminary.h"
#include "lib/png.h"
#include "lib/image.h"
#include "lib/device.h"
#include "lib/scene.h"
#include "lib/raytrace.h"
#include "lib/mesh.h"
#include "lib/wavefront.h"
#include "lib/bvh.h"
#include "lib/texture.h"
#include "lib/error.h"
#include "lib/processing.h"

int main() {
  initialize_device();

  clock_t time = clock();

  Wavefront_Mesh* meshes;

  int meshes_length = read_mesh_from_file("Field.obj", &meshes, 0);

  printf("[%.3fs] Mesh loaded from file.\n", ((double) (clock() - time)) / CLOCKS_PER_SEC);

  Triangle* triangles;

  unsigned int triangle_count = convert_wavefront_mesh(&triangles, meshes, meshes_length);

  printf(
    "[%.3fs] Mesh converted. Total Triangles: %u\n", ((double) (clock() - time)) / CLOCKS_PER_SEC,
    triangle_count);

  int nodes_length;

  Node* nodes = build_bvh_structure(&triangles, triangle_count, 18, &nodes_length);

  printf("[%.3fs] BVH structure built.\n", ((double) (clock() - time)) / CLOCKS_PER_SEC);

  TextureRGBA* albedo_maps      = (TextureRGBA*) malloc(sizeof(TextureRGBA) * 7);
  TextureRGBA* illuminance_maps = (TextureRGBA*) malloc(sizeof(TextureRGBA) * 1);
  TextureRGBA* material_maps    = (TextureRGBA*) malloc(sizeof(TextureRGBA) * 2);

  albedo_maps[0].width          = 1;
  albedo_maps[0].height         = 1;
  albedo_maps[0].data           = (RGBAF*) malloc(sizeof(RGBAF));
  albedo_maps[0].data[0].r      = 0.2f;
  albedo_maps[0].data[0].g      = 0.2f;
  albedo_maps[0].data[0].b      = 0.2f;
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
  material_maps[0].data[0].r    = 0.1f;
  material_maps[0].data[0].g    = 1.0f;
  material_maps[0].data[0].b    = 1.0f;
  material_maps[0].data[0].a    = 0.0f;

  albedo_maps[1] = load_texture_from_png("Textures/AMX_50B_chassis_01_AM_hd.png");
  albedo_maps[2] = load_texture_from_png("Textures/AMX_50B_guns_AM_hd.png");
  albedo_maps[3] = load_texture_from_png("Textures/AMX_50B_hull_01_AM_hd.png");
  albedo_maps[4] = load_texture_from_png("Textures/AMX_50B_turret_01_AM_hd.png");
  albedo_maps[5] = load_texture_from_png("Textures/Bat_Chatillon25t_track_AM_hd.png");

  albedo_maps[6].width     = 1;
  albedo_maps[6].height    = 1;
  albedo_maps[6].data      = (RGBAF*) malloc(sizeof(RGBAF));
  albedo_maps[6].data[0].r = 0.4f;
  albedo_maps[6].data[0].g = 0.35f;
  albedo_maps[6].data[0].b = 0.2f;
  albedo_maps[6].data[0].a = 1.0f;

  material_maps[1].width     = 1;
  material_maps[1].height    = 1;
  material_maps[1].data      = (RGBAF*) malloc(sizeof(RGBAF));
  material_maps[1].data[0].r = 0.95f;
  material_maps[1].data[0].g = 0.0f;
  material_maps[1].data[0].b = 1.0f;
  material_maps[1].data[0].a = 0.0f;

  void* albedo_atlas      = initialize_textures(albedo_maps, 7);
  void* illuminance_atlas = initialize_textures(illuminance_maps, 1);
  void* material_atlas    = initialize_textures(material_maps, 2);

  texture_assignment* texture_assignments =
    (texture_assignment*) malloc(sizeof(texture_assignment) * meshes_length);

  for (int i = 0; i < meshes_length; i++) {
    texture_assignments[i].albedo_map      = 0;
    texture_assignments[i].illuminance_map = 0;
    texture_assignments[i].material_map    = 0;
  }

  texture_assignments[0].albedo_map = 5;
  texture_assignments[1].albedo_map = 5;
  texture_assignments[2].albedo_map = 1;
  texture_assignments[3].albedo_map = 1;
  texture_assignments[4].albedo_map = 1;
  texture_assignments[5].albedo_map = 1;
  texture_assignments[6].albedo_map = 2;
  texture_assignments[7].albedo_map = 3;
  texture_assignments[8].albedo_map = 4;

  texture_assignments[9].albedo_map = 6;

  texture_assignments[10].material_map = 1;

  printf("[%.3fs] Textures loaded.\n", ((double) (clock() - time)) / CLOCKS_PER_SEC);

  Scene scene;

  scene.far_clip_distance = 1000.0f;

  scene.camera.fov   = 3.0f;
  scene.camera.pos.x = 9.0f;
  scene.camera.pos.y = 3.0f;
  scene.camera.pos.z = 12.0f;

  scene.camera.rotation.x = 0.2f;
  scene.camera.rotation.y = 0.5f;
  scene.camera.rotation.z = 0.0f;

  scene.triangles        = triangles;
  scene.triangles_length = triangle_count;
  scene.nodes            = nodes;
  scene.nodes_length     = nodes_length;

  const int width  = 2560;
  const int height = 1440;

  raytrace_instance* instance = init_raytracing(width, height, 10, 250);

  printf("[%.3fs] Instance set up.\n", ((double) (clock() - time)) / CLOCKS_PER_SEC);

  trace_scene(
    scene, instance, albedo_atlas, illuminance_atlas, material_atlas, texture_assignments,
    meshes_length);

  printf("[%.3fs] Raytracing done.\n", ((double) (clock() - time)) / CLOCKS_PER_SEC);

  post_bloom(instance, 2.0f);

  printf("[%.3fs] Applied Bloom.\n", ((double) (clock() - time)) / CLOCKS_PER_SEC);

  RGB8* frame = (RGB8*) malloc(sizeof(RGB8) * instance->width * instance->height);

  frame_buffer_to_8bit_image(scene.camera, instance, frame);

  RGB16* frame_16 = (RGB16*) malloc(sizeof(RGB16) * instance->width * instance->height);

  frame_buffer_to_16bit_image(scene.camera, instance, frame_16);

  printf(
    "[%.3fs] Converted frame buffer to image.\n", ((double) (clock() - time)) / CLOCKS_PER_SEC);

  free_raytracing(instance);

  printf("[%.3fs] Instance freed.\n", ((double) (clock() - time)) / CLOCKS_PER_SEC);

  store_as_png(
    "test.png", (uint8_t*) frame, sizeof(RGB8) * width * height, width, height,
    PNG_COLORTYPE_TRUECOLOR, PNG_BITDEPTH_8);

  store_as_png(
    "test16.png", (uint8_t*) frame_16, sizeof(RGB16) * width * height, width, height,
    PNG_COLORTYPE_TRUECOLOR, PNG_BITDEPTH_16);

  printf("[%.3fs] PNG file created.\n", ((double) (clock() - time)) / CLOCKS_PER_SEC);

  free_textures(albedo_atlas, 1);
  free_textures(illuminance_atlas, 1);
  free_textures(material_atlas, 1);

  free(triangles);
  free(nodes);
  for (int i = 0; i < meshes_length; i++) {
    free(meshes[i].triangles);
    free(meshes[i].normals);
    free(meshes[i].uvs);
    free(meshes[i].vertices);
  }
  free(meshes);

  return 0;
}
