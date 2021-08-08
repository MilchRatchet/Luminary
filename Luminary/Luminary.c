#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "lib/scene.h"
#include "lib/raytrace.h"
#include "lib/error.h"
#include "lib/output.h"

static int parse_command(char* arg, char* opt1, char* opt2) {
  int ptr1    = -1;
  int result1 = 1;

  do {
    ptr1++;
    result1 &= (arg[ptr1] == opt1[ptr1]);
  } while (arg[ptr1] != '\0' && opt1[ptr1] != '\0');

  result1 &= (opt1[ptr1] == '\0');

  if (result1)
    return 1;

  int ptr2    = -1;
  int result2 = 1;

  do {
    ptr2++;
    result2 &= (arg[ptr2] == opt2[ptr2]);
  } while (arg[ptr2] != '\0' && opt2[ptr2] != '\0');

  result2 &= (opt2[ptr2] == '\0');

  return result2;
}

int main(int argc, char* argv[]) {
  assert(argc >= 2, "No scene description was given!", 1);

  initialize_device();

  clock_t time = clock();

  RaytraceInstance* instance;

  char* output_name = (char*) malloc(4096);

  Scene scene = load_scene(argv[1], &instance, &output_name);

  printf("[%.3fs] Instance set up.\n", ((double) (clock() - time)) / CLOCKS_PER_SEC);

  int realtime = 0;

  for (int i = 2; i < argc; i++) {
    realtime |= parse_command(argv[i], "-r", "--realtime");
  }

  if (realtime) {
    realtime_output(scene, instance);
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
