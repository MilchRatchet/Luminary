#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "baked.h"
#include "bench.h"
#include "log.h"
#include "output.h"
#include "raytrace.h"
#include "scene.h"
#include "utils.h"

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

static int parse_number(const char* arg) {
  int contains_invalid_char = 0;

  int ptr = 0;

  while (arg[ptr] != '\0') {
    unsigned char c = ((unsigned char*) arg)[ptr++];
    if (c < 48 || c > 57) {
      contains_invalid_char |= 1;
    }
  }

  return (contains_invalid_char) ? 0 : atoi(arg);
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
  int offline         = 0;
  int bench           = 0;
  int write_logs      = 0;
  int custom_samples  = 0;
  int offline_samples = 0;
  int custom_width    = 0;
  int custom_height   = 0;
  int width           = 0;
  int height          = 0;

  for (int i = 2; i < argc; i++) {
    if (custom_samples) {
      offline_samples = parse_number(argv[i]);
    }

    if (custom_width) {
      width = parse_number(argv[i]);
    }

    if (custom_height) {
      height = parse_number(argv[i]);
    }

    offline |= parse_command(argv[i], "-o", "--offline");
    bench |= parse_command(argv[i], "-t", "--timings");
    write_logs |= parse_command(argv[i], "-l", "--logs");
    custom_samples = parse_command(argv[i], "-s", "--samples");
    custom_width   = parse_command(argv[i], "-w", "--width");
    custom_height  = parse_command(argv[i], "-h", "--height");
  }

  init_log(write_logs);

  initialize_device();

  assert(argc >= 2, "No scene description was given!", 1);

  int file_type = magic(argv[1]);

  if (bench)
    bench_activate();

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

  instance->realtime = !offline;

  if (offline_samples)
    instance->offline_samples = offline_samples;

  int custom_res = width || height;

  if (custom_res) {
    instance->settings.width  = (width) ? width : instance->width;
    instance->settings.height = (height) ? height : instance->height;

    reset_raytracing(instance);
  }

  info_message("Instance set up.");

  if (offline) {
    offline_output(instance);
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