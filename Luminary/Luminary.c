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

  raytrace_instance* instance;

  char* output_name = (char*) malloc(4096);

  Scene scene = load_scene("Sponza.lum", &instance, &output_name);

  printf("[%.3fs] Instance set up.\n", ((double) (clock() - time)) / CLOCKS_PER_SEC);

  trace_scene(scene, instance);

  printf("[%.3fs] Raytracing done.\n", ((double) (clock() - time)) / CLOCKS_PER_SEC);

  post_median_filter(instance, 0.9f);

  printf("[%.3fs] Applied Median Filter.\n", ((double) (clock() - time)) / CLOCKS_PER_SEC);

  post_bloom(instance, 3.0f);

  printf("[%.3fs] Applied Bloom.\n", ((double) (clock() - time)) / CLOCKS_PER_SEC);

  post_tonemapping(instance);

  printf("[%.3fs] Applied Tonemapping.\n", ((double) (clock() - time)) / CLOCKS_PER_SEC);

  RGB8* frame = (RGB8*) malloc(sizeof(RGB8) * instance->width * instance->height);

  frame_buffer_to_8bit_image(scene.camera, instance, frame);

  RGB16* frame_16 = (RGB16*) malloc(sizeof(RGB16) * instance->width * instance->height);

  frame_buffer_to_16bit_image(scene.camera, instance, frame_16);

  printf(
    "[%.3fs] Converted frame buffer to image.\n", ((double) (clock() - time)) / CLOCKS_PER_SEC);

  store_as_png(
    output_name, (uint8_t*) frame, sizeof(RGB8) * instance->width * instance->height,
    instance->width, instance->height, PNG_COLORTYPE_TRUECOLOR, PNG_BITDEPTH_8);

  printf("[%.3fs] PNG file created.\n", ((double) (clock() - time)) / CLOCKS_PER_SEC);

  free(output_name);

  free_scene(scene, instance);

  free_raytracing(instance);

  printf("[%.3fs] Instance freed.\n", ((double) (clock() - time)) / CLOCKS_PER_SEC);

  return 0;
}
