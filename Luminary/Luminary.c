#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "lib/baked.h"
#include "lib/bench.h"
#include "lib/error.h"
#include "lib/log.h"
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
      warn_message("Input file (%s) is of unknown type ending with %c. Assuming *.lum file.", path, last_char);
      return LUM_FILE;
  }
}

int main(int argc, char* argv[]) {
  init_log();

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

  switch (file_type) {
    case LUM_FILE:
      instance = load_scene(argv[1]);
      break;
    case OBJ_FILE:
      instance = load_obj_as_scene(argv[1]);
      break;
    case BAKED_FILE:
      instance = load_baked(argv[1]);
      break;
    default:
      instance = load_scene(argv[1]);
      break;
  }

  info_message("Instance set up.");

  if (offline) {
    offline_output(instance, time);
  }
  else {
    realtime_output(instance);
  }

  free_atlases(instance);
  free_outputs(instance);

  info_message("Instance freed.");

  write_log();

  return 0;
}
