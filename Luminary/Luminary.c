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
  Camera camera    = {.x = 0, .y = 0, .z = 0, .fov = 1.56079632679};

  const int rows = 10;

  Sphere* spheres = (Sphere*) malloc(sizeof(Sphere) * 9 * rows);

  int ptr = 0;

  for (int i = 0; i < 9; i++) {
    for (int j = 0; j < rows; j++) {
      spheres[ptr].id     = ptr + 1;
      spheres[ptr].x      = -10 + 10 * (i % 3);
      spheres[ptr].y      = -10 + 10 * (i / 3);
      spheres[ptr].z      = -10 - 10 * j;
      spheres[ptr].radius = 1;

      spheres[ptr].color.r = 25;
      spheres[ptr].color.g = 25;
      spheres[ptr].color.b = 25;

      ptr++;
    }
  }

  Light* lights = (Light*) malloc(sizeof(Light) * 3);

  lights[0].id = ptr++;
  lights[0].x  = 20;
  lights[0].y  = 10;
  lights[0].z  = 0;

  lights[0].color.r = 100;
  lights[0].color.g = 0;
  lights[0].color.b = 0;

  lights[1].id = ptr++;
  lights[1].x  = -20;
  lights[1].y  = 10;
  lights[1].z  = 0;

  lights[1].color.r = 0;
  lights[1].color.g = 0;
  lights[1].color.b = 100;

  lights[2].id = ptr++;
  lights[2].x  = 0;
  lights[2].y  = -20;
  lights[2].z  = 0;

  lights[2].color.r = 0;
  lights[2].color.g = 100;
  lights[2].color.b = 0;

  Scene scene = {
    .camera            = camera,
    .far_clip_distance = 1000,
    .spheres           = spheres,
    .spheres_length    = 9 * rows,
    .lights            = lights,
    .lights_length     = 3};

  uint8_t* frame = scene_to_frame(scene, width, height);

  // uint8_t* frame = test_frame(width, height);

  printf("[%.3fs] Frame Buffer set by GPU.\n", ((double) (clock() - time)) / CLOCKS_PER_SEC);

  store_as_png(
    "test.png", (uint8_t*) frame, sizeof(RGB8) * width * height, width, height,
    PNG_COLORTYPE_TRUECOLOR, PNG_BITDEPTH_8);

  printf("[%.3fs] PNG file created.\n", ((double) (clock() - time)) / CLOCKS_PER_SEC);

  return 0;
}
