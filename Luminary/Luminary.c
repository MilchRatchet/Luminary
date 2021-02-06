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
  Sphere* spheres  = (Sphere*) malloc(sizeof(Sphere) * 4);

  spheres[0].x      = 0;
  spheres[0].y      = 0;
  spheres[0].z      = -20;
  spheres[0].radius = 7;

  spheres[0].color.r = 255;
  spheres[0].color.g = 255;
  spheres[0].color.b = 0;

  spheres[1].x      = -30;
  spheres[1].y      = 0;
  spheres[1].z      = -25;
  spheres[1].radius = 10;

  spheres[1].color.r = 0;
  spheres[1].color.g = 0;
  spheres[1].color.b = 255;

  spheres[2].x      = 10;
  spheres[2].y      = 0;
  spheres[2].z      = -10;
  spheres[2].radius = 3;

  spheres[2].color.r = 255;
  spheres[2].color.g = 0;
  spheres[2].color.b = 0;

  spheres[3].x      = 10;
  spheres[3].y      = 6;
  spheres[3].z      = -12;
  spheres[3].radius = 3;

  spheres[3].color.r = 0;
  spheres[3].color.g = 255;
  spheres[3].color.b = 255;

  Scene scene = {
    .camera = camera, .far_clip_distance = 1000, .spheres = spheres, .spheres_length = 4};

  uint8_t* frame = scene_to_frame(scene, width, height);

  // uint8_t* frame = test_frame(width, height);

  printf("[%.3fs] Frame Buffer set by GPU.\n", ((double) (clock() - time)) / CLOCKS_PER_SEC);

  store_as_png(
    "test.png", (uint8_t*) frame, sizeof(RGB8) * width * height, width, height,
    PNG_COLORTYPE_TRUECOLOR, PNG_BITDEPTH_8);

  printf("[%.3fs] PNG file created.\n", ((double) (clock() - time)) / CLOCKS_PER_SEC);

  return 0;
}
