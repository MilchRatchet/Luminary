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

int main() {
  display_gpu_information();

  clock_t time = clock();

  Wavefront_Mesh* meshes;

  int meshes_length = read_mesh_from_file("Garage.obj", &meshes, 0);

  for (int i = 0; i < meshes_length; i++) {
    printf("Mesh %d\n", i);
    printf("  Vertices %d\n", meshes[i].vertices_length);
    printf("  UVs %d\n", meshes[i].uvs_length);
    printf("  Normals %d\n", meshes[i].normals_length);
    printf("  Triangles %d\n", meshes[i].triangles_length);
  }

  printf("[%.3fs] Mesh loaded from file.\n", ((double) (clock() - time)) / CLOCKS_PER_SEC);

  Triangle* triangles;

  unsigned int triangle_count = convert_wavefront_mesh(&triangles, meshes, meshes_length);

  printf("Total Triangles: %u\n", triangle_count);

  printf("[%.3fs] Mesh converted.\n", ((double) (clock() - time)) / CLOCKS_PER_SEC);

  int nodes_length;

  Node* nodes = build_bvh_structure(&triangles, triangle_count, 16, &nodes_length);

  printf("[%.3fs] BVH structure built.\n", ((double) (clock() - time)) / CLOCKS_PER_SEC);

  TextureRGBA* albedo_maps      = (TextureRGBA*) malloc(sizeof(TextureRGBA) * 17);
  TextureRGBA* illuminance_maps = (TextureRGBA*) malloc(sizeof(TextureRGBA) * 3);
  TextureRGBA* material_maps    = (TextureRGBA*) malloc(sizeof(TextureRGBA) * 3);

  albedo_maps[0].width     = 1;
  albedo_maps[0].height    = 1;
  albedo_maps[0].data      = (RGBAF*) malloc(sizeof(RGBAF));
  albedo_maps[0].data[0].r = 0.2f;
  albedo_maps[0].data[0].g = 0.2f;
  albedo_maps[0].data[0].b = 0.2f;
  albedo_maps[0].data[0].a = 1.0f;

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

  albedo_maps[1]  = load_texture_from_png("Textures/Type59_chassis_01_AM_hd.png");
  albedo_maps[2]  = load_texture_from_png("Textures/Type59_guns_AM_hd.png");
  albedo_maps[3]  = load_texture_from_png("Textures/Type59_hull_01_AM_hd.png");
  albedo_maps[4]  = load_texture_from_png("Textures/Type59_turret_01_AM_hd.png");
  albedo_maps[5]  = load_texture_from_png("Textures/Projet_4_1_chassis_01_AM_hd.png");
  albedo_maps[6]  = load_texture_from_png("Textures/Projet_4_1_guns_AM_hd.png");
  albedo_maps[7]  = load_texture_from_png("Textures/Projet_4_1_hull_01_2_AM_hd.png");
  albedo_maps[8]  = load_texture_from_png("Textures/Projet_4_1_hull_01_AM_hd.png");
  albedo_maps[9]  = load_texture_from_png("Textures/Projet_4_1_turret_01_AM_hd.png");
  albedo_maps[10] = load_texture_from_png("Textures/Concept_1b_chassis_01_AM_hd.png");
  albedo_maps[11] = load_texture_from_png("Textures/Concept_1b_guns_AM_hd.png");
  albedo_maps[12] = load_texture_from_png("Textures/Concept_1b_hull_01_AM_hd.png");
  albedo_maps[13] = load_texture_from_png("Textures/Concept_1b_track_AM_hd.png");
  albedo_maps[14] = load_texture_from_png("Textures/Concept_1b_turret_01_AM_hd.png");
  albedo_maps[15] = load_texture_from_png("Textures/WZ_111_track_AM_hd.png");
  albedo_maps[16] = load_texture_from_png("Textures/Projet_4_1_track_AM_hd.png");

  material_maps[1].width     = 1;
  material_maps[1].height    = 1;
  material_maps[1].data      = (RGBAF*) malloc(sizeof(RGBAF));
  material_maps[1].data[0].r = 0.8f;
  material_maps[1].data[0].g = 1.0f;
  material_maps[1].data[0].b = 0.0f;
  material_maps[1].data[0].a = 0.0f;

  material_maps[2].width     = 1;
  material_maps[2].height    = 1;
  material_maps[2].data      = (RGBAF*) malloc(sizeof(RGBAF));
  material_maps[2].data[0].r = 0.8f;
  material_maps[2].data[0].g = 0.0f;
  material_maps[2].data[0].b = 0.0f;
  material_maps[2].data[0].a = 0.0f;

  illuminance_maps[1].width     = 1;
  illuminance_maps[1].height    = 1;
  illuminance_maps[1].data      = (RGBAF*) malloc(sizeof(RGBAF));
  illuminance_maps[1].data[0].r = 0.9f;
  illuminance_maps[1].data[0].g = 0.9f;
  illuminance_maps[1].data[0].b = 1.0f;
  illuminance_maps[1].data[0].a = 0.0f;

  illuminance_maps[2].width     = 1;
  illuminance_maps[2].height    = 1;
  illuminance_maps[2].data      = (RGBAF*) malloc(sizeof(RGBAF));
  illuminance_maps[2].data[0].r = 1.0f;
  illuminance_maps[2].data[0].g = 0.9f;
  illuminance_maps[2].data[0].b = 0.9f;
  illuminance_maps[2].data[0].a = 0.0f;

  void* albedo_atlas      = initialize_textures(albedo_maps, 17);
  void* illuminance_atlas = initialize_textures(illuminance_maps, 3);
  void* material_atlas    = initialize_textures(material_maps, 3);

  texture_assignment* texture_assignments =
    (texture_assignment*) malloc(sizeof(texture_assignment) * meshes_length);

  for (int i = 0; i < meshes_length; i++) {
    texture_assignments[i].albedo_map      = 0;
    texture_assignments[i].illuminance_map = 0;
    texture_assignments[i].material_map    = 0;
  }

  texture_assignments[0].albedo_map = 15;
  texture_assignments[1].albedo_map = 15;
  texture_assignments[2].albedo_map = 1;
  texture_assignments[3].albedo_map = 1;
  texture_assignments[4].albedo_map = 2;
  texture_assignments[5].albedo_map = 3;
  texture_assignments[6].albedo_map = 4;

  texture_assignments[7].albedo_map  = 13;
  texture_assignments[8].albedo_map  = 13;
  texture_assignments[9].albedo_map  = 10;
  texture_assignments[10].albedo_map = 10;
  texture_assignments[11].albedo_map = 11;
  texture_assignments[12].albedo_map = 12;
  texture_assignments[13].albedo_map = 14;

  texture_assignments[14].albedo_map = 16;
  texture_assignments[15].albedo_map = 16;
  texture_assignments[16].albedo_map = 5;
  texture_assignments[17].albedo_map = 5;
  texture_assignments[18].albedo_map = 6;
  texture_assignments[19].albedo_map = 8;
  texture_assignments[20].albedo_map = 7;
  texture_assignments[21].albedo_map = 9;

  texture_assignments[23].illuminance_map = 1;
  texture_assignments[24].illuminance_map = 2;

  texture_assignments[2].material_map = 1;
  texture_assignments[3].material_map = 1;
  texture_assignments[4].material_map = 1;
  texture_assignments[5].material_map = 1;
  texture_assignments[6].material_map = 1;

  texture_assignments[22].material_map = 2;

  printf("[%.3fs] Textures loaded.\n", ((double) (clock() - time)) / CLOCKS_PER_SEC);

  Scene scene;

  scene.far_clip_distance = 1000.0f;

  scene.camera.fov   = 1.0f;
  scene.camera.pos.x = 9.0f;
  scene.camera.pos.y = 4.0f;
  scene.camera.pos.z = -12.0f;

  scene.camera.rotation.x = -0.3f;
  scene.camera.rotation.y = 2.5f;
  scene.camera.rotation.z = 0.0f;

  scene.triangles        = triangles;
  scene.triangles_length = triangle_count;
  scene.nodes            = nodes;
  scene.nodes_length     = nodes_length;

  const int width  = 1920;
  const int height = 1080;

  raytrace_instance* instance = init_raytracing(width, height, 10, 10);

  printf("[%.3fs] Instance set up.\n", ((double) (clock() - time)) / CLOCKS_PER_SEC);

  trace_scene(
    scene, instance, albedo_atlas, illuminance_atlas, material_atlas, texture_assignments,
    meshes_length);

  printf("[%.3fs] Raytracing done.\n", ((double) (clock() - time)) / CLOCKS_PER_SEC);

  RGB8* frame = (RGB8*) malloc(sizeof(RGB8) * instance->width * instance->height);

  frame_buffer_to_image(scene.camera, instance, frame);

  printf(
    "[%.3fs] Converted frame buffer to image.\n", ((double) (clock() - time)) / CLOCKS_PER_SEC);

  free_raytracing(instance);

  printf("[%.3fs] Instance freed.\n", ((double) (clock() - time)) / CLOCKS_PER_SEC);

  store_as_png(
    "test.png", (uint8_t*) frame, sizeof(RGB8) * width * height, width, height,
    PNG_COLORTYPE_TRUECOLOR, PNG_BITDEPTH_8);

  printf("[%.3fs] PNG file created.\n", ((double) (clock() - time)) / CLOCKS_PER_SEC);

  free_textures(albedo_atlas, 2);
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
