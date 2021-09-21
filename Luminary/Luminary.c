#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "lib/bench.h"
#include "lib/error.h"
#include "lib/output.h"
#include "lib/raytrace.h"
#include "lib/scene.h"

static int parse_command(const char* arg, char* opt1, char* opt2) {
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

#define LUM_FILE 0
#define OBJ_FILE 1
#define BAKED_FILE 2

// We figure out what kind of file was given only by the last char in the path
static int magic(char* path) {
  int ptr        = 0;
  char last_char = 0;

  while (path[ptr] != '\0') {
    last_char = path[ptr];
    ptr++;
  }

  switch (last_char) {
    case 'm':
      return LUM_FILE;
    case 'j':
      return OBJ_FILE;
    case 'd':
      return BAKED_FILE;
    default:
      return LUM_FILE;
  }
}

int main(int argc, char* argv[]) {
  initialize_device();

  assert(argc >= 2, "No scene description was given!", 1);

  int file_type = magic(argv[1]);

  int offline = 0;
  int bench   = 0;

  for (int i = 2; i < argc; i++) {
    offline |= parse_command(argv[i], "-o", "--offline");
    bench |= parse_command(argv[i], "-t", "--timings");
  }

  if (bench)
    bench_activate();

  clock_t time = clock();

  RaytraceInstance* instance;
  Scene scene;

  switch (file_type) {
    case LUM_FILE:
      scene = load_scene(argv[1], &instance);
      break;
    case OBJ_FILE:
      scene = load_obj_as_scene(argv[1], &instance);
      break;
    default:
      scene = load_scene(argv[1], &instance);
      break;
  }

  printf("[%.3fs] Instance set up.\n", ((double) (clock() - time)) / CLOCKS_PER_SEC);

  if (offline) {
    offline_output(scene, instance, time);
  }
  else {
    realtime_output(scene, instance);
  }

  free_scene(scene, instance);
  free_outputs(instance);

  printf("[%.3fs] Instance freed.\n", ((double) (clock() - time)) / CLOCKS_PER_SEC);

  return 0;
}
