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
#include "examples.h"

int main() {
  display_gpu_information();

  clock_t time = clock();

  Scene scene = example_scene2();

  const int width  = 3840;
  const int height = 2160;

  raytrace_instance* instance = init_raytracing(width, height, 5, 10000);

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

  return 0;
}
