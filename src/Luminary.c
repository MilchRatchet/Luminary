#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "baked.h"
#include "bench.h"
#include "config.h"
#include "device.h"
#include "log.h"
#include "output.h"
#include "raytrace.h"
#include "scene.h"
#include "utils.h"

static int parse_command(const char* arg, char* opt1, char* opt2) {
  if (opt1) {
    int ptr1    = -1;
    int result1 = 1;

    do {
      ptr1++;
      result1 &= (arg[ptr1] == opt1[ptr1]);
    } while (arg[ptr1] != '\0' && opt1[ptr1] != '\0');

    result1 &= (opt1[ptr1] == '\0');

    if (result1)
      return 1;
  }

  if (opt2) {
    int ptr2    = -1;
    int result2 = 1;

    do {
      ptr2++;
      result2 &= (arg[ptr2] == opt2[ptr2]);
    } while (arg[ptr2] != '\0' && opt2[ptr2] != '\0');

    result2 &= (opt2[ptr2] == '\0');

    if (result2)
      return 1;
  }

  return 0;
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

static void luminary_version_output() {
  printf("Luminary %s (Branch: %s)\n", LUMINARY_VERSION_DATE, LUMINARY_BRANCH_NAME);
  printf("(%s, %s, CUDA %s, OptiX %s)\n", LUMINARY_COMPILER, LUMINARY_OS, LUMINARY_CUDA_VERSION, LUMINARY_OPTIX_VERSION);
  printf("Copyright (c) 2023 MilchRatchet\n");
}

int main(int argc, char* argv[]) {
  int offline                  = 0;
  int bench                    = 0;
  int write_logs               = 0;
  int custom_samples           = 0;
  int offline_samples          = 0;
  int custom_width             = 0;
  int custom_height            = 0;
  int width                    = 0;
  int height                   = 0;
  OutputImageFormat img_format = IMGFORMAT_PNG;
  int post_process_menu        = 0;
  int unittest                 = 0;
  int version_output           = 0;
  int force_displacement       = 0;
  int disable_omm              = 0;
  int aov_mode                 = 0;

  for (int i = 1; i < argc; i++) {
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
    post_process_menu |= parse_command(argv[i], "-p", "--post-menu");
    unittest |= parse_command(argv[i], "-u", "--unittest");
    version_output |= parse_command(argv[i], "-v", "--version");
    force_displacement |= parse_command(argv[i], (char*) 0, "--force-displacement");
    disable_omm |= parse_command(argv[i], (char*) 0, "--no-omm");
    aov_mode |= parse_command(argv[i], (char*) 0, "--aov-mode");

    if (parse_command(argv[i], (char*) 0, "--png")) {
      img_format = IMGFORMAT_PNG;
    }

    if (parse_command(argv[i], (char*) 0, "--qoi")) {
      img_format = IMGFORMAT_QOI;
    }
  }

  if (version_output) {
    luminary_version_output();
    return 0;
  }

  init_log(write_logs);

  device_init();

  assert(argc >= 2, "No scene description was given!", 1);

  int file_type = magic(argv[1]);

  if (bench)
    bench_activate();

  CommandlineOptions options = {.aov_mode = aov_mode, .width = width, .height = height};

  RaytraceInstance* instance;

  switch (file_type) {
    case LUM_FILE:
      instance = scene_load_lum(argv[1], options);
      break;
    case OBJ_FILE:
      instance = scene_load_obj(argv[1], options);
      break;
    case BAKED_FILE:
      instance = load_baked(argv[1], options);
      break;
    default:
      instance = scene_load_lum(argv[1], options);
      break;
  }

  if (unittest) {
    if (device_brdf_unittest(0.95f))
      error_message("UNITTEST - BRDF failed.");

    return 0;
  }

  if (force_displacement) {
    log_message("DMM Usage was forced on by the user.");
    instance->optix_bvh.force_dmm_usage = 1;
  }

  if (disable_omm) {
    log_message("OMM Usage was forced off by the user.");
    instance->optix_bvh.disable_omm = 1;
  }

  instance->realtime = !offline;

  if (offline_samples)
    instance->offline_samples = offline_samples;

  instance->image_format      = img_format;
  instance->post_process_menu = post_process_menu;

  if (offline) {
    offline_output(instance);
  }
  else {
    realtime_output(instance);
  }

  free_atlases(instance);
  raytrace_free_output_buffers(instance);

  write_log();

  return 0;
}
