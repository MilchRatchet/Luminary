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

  Node* nodes = build_bvh_structure(&triangles, &triangle_count, 14, &nodes_length);

  /*for (int i = 0; i < nodes_length; i++) {
    printf("Node: %d\n", i);
    printf("  Triangle Count: %d\n", nodes[i].triangle_count);
  }*/

  printf("Total Triangles: %u\n", triangle_count);

  printf("[%.3fs] BVH structure built.\n", ((double) (clock() - time)) / CLOCKS_PER_SEC);

  Scene scene;

  scene.far_clip_distance = 1000.0f;

  scene.camera.fov   = 1.0f;
  scene.camera.pos.x = 9.0f;
  scene.camera.pos.y = 3.0f;
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

  raytrace_instance* instance = init_raytracing(width, height, 5, 1);

  printf("[%.3fs] Instance set up.\n", ((double) (clock() - time)) / CLOCKS_PER_SEC);

  trace_scene(scene, instance);

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
