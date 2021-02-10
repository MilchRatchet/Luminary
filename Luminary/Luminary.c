#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "Luminary.h"
#include "lib/png.h"
#include "lib/image.h"
#include "lib/cudatest.h"
#include "lib/device.h"
#include "lib/scene.h"
#include "lib/raytrace.h"

int main() {
  display_gpu_information();

  clock_t time = clock();

  const int width  = 3840;
  const int height = 2160;
  Camera camera    = {.pos = {.x = 0.0f, .y = 0.0f, .z = 0.0f}, .fov = 1.56079632679};

  const int rows = 10;

  Sphere* spheres = (Sphere*) malloc(sizeof(Sphere) * (9 * rows + 1));

  int ptr = 0;

  for (int i = 0; i < 9; i++) {
    for (int j = 0; j < rows; j++) {
      spheres[ptr].id     = ptr + 1;
      spheres[ptr].pos.x  = -10 + 10 * (i % 3);
      spheres[ptr].pos.y  = -10 + 10 * (i / 3);
      spheres[ptr].pos.z  = -10 - 10 * j;
      spheres[ptr].radius = 1;
      spheres[ptr].sign   = 1;

      spheres[ptr].color.r = 1.0f;
      spheres[ptr].color.g = 0.1f;
      spheres[ptr].color.b = 0.1f;

      ptr++;
    }
  }

  spheres[ptr].id     = ptr + 1;
  spheres[ptr].pos.x  = 0;
  spheres[ptr].pos.y  = 0;
  spheres[ptr].pos.z  = 0;
  spheres[ptr].radius = 100;
  spheres[ptr].sign   = -1;

  spheres[ptr].color.r = 0.1f;
  spheres[ptr].color.g = 0.1f;
  spheres[ptr].color.b = 1.0f;

  ptr++;

  Light* lights = (Light*) malloc(sizeof(Light) * 3);

  lights[0].id    = ptr++;
  lights[0].pos.x = 20;
  lights[0].pos.y = 10;
  lights[0].pos.z = 0;

  lights[0].color.r = 1.0f;
  lights[0].color.g = 0;
  lights[0].color.b = 0;

  lights[1].id    = ptr++;
  lights[1].pos.x = -20;
  lights[1].pos.y = 10;
  lights[1].pos.z = 0;

  lights[1].color.r = 0;
  lights[1].color.g = 0;
  lights[1].color.b = 1.0f;

  lights[2].id    = ptr++;
  lights[2].pos.x = 0;
  lights[2].pos.y = -20;
  lights[2].pos.z = 0;

  lights[2].color.r = 0;
  lights[2].color.g = 1.0f;
  lights[2].color.b = 0;

  Scene scene = {
    .camera            = camera,
    .far_clip_distance = 1000,
    .spheres           = spheres,
    .spheres_length    = 1 + 9 * rows,
    .lights            = lights,
    .lights_length     = 1};

  raytrace_instance* instance = init_raytracing(width, height);

  printf("[%.3fs] Instance set up.\n", ((double) (clock() - time)) / CLOCKS_PER_SEC);

  trace_scene(scene, instance);

  printf("[%.3fs] Raytracing done.\n", ((double) (clock() - time)) / CLOCKS_PER_SEC);

  RGB8* frame = (RGB8*) malloc(sizeof(RGB8) * instance->width * instance->height);

  frame_buffer_to_image(camera, instance, frame);

  printf(
    "[%.3fs] Converted frame buffer to image.\n", ((double) (clock() - time)) / CLOCKS_PER_SEC);

  free_raytracing(instance);

  printf("[%.3fs] Instance freed.\n", ((double) (clock() - time)) / CLOCKS_PER_SEC);

  store_as_png(
    "test.png", (uint8_t*) frame, sizeof(RGB8) * width * height, width, height,
    PNG_COLORTYPE_TRUECOLOR, PNG_BITDEPTH_8);

  printf("[%.3fs] PNG file created.\n", ((double) (clock() - time)) / CLOCKS_PER_SEC);

  return 0;
}
