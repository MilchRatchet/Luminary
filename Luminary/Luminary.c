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
#include "lib/output.h"

int main(int argc, char* argv[]) {
  assert(argc >= 2, "No scene description was given!", 1);

  initialize_device();

  clock_t time = clock();

  raytrace_instance* instance;

  char* output_name = (char*) malloc(4096);

  Scene scene = load_scene(argv[1], &instance, &output_name);

  printf("[%.3fs] Instance set up.\n", ((double) (clock() - time)) / CLOCKS_PER_SEC);

  if (argc > 2 && argv[2][0] == 'r') {
    if (argc > 3 && argv[3][0] != '0') {
      realtime_output(scene, instance, 1);
    }
    else {
      realtime_output(scene, instance, 0);
    }
  }
  else {
    offline_output(scene, instance, output_name, 1, time);
  }

  free(output_name);

  free_scene(scene, instance);

  free_outputs(instance);

  printf("[%.3fs] Instance freed.\n", ((double) (clock() - time)) / CLOCKS_PER_SEC);

  return 0;
}
