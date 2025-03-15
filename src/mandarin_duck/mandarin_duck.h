#ifndef MANDARIN_DUCK_INSTANCE_H
#define MANDARIN_DUCK_INSTANCE_H

#include "display.h"
#include "utils.h"

enum MandarinDuckMode { MANDARIN_DUCK_MODE_DEFAULT, MANDARIN_DUCK_MODE_BENCHMARK, MANDARIN_DUCK_MODE_COUNT } typedef MandarinDuckMode;

struct MandarinDuckCreateArgs {
  LuminaryHost* host;
  const char* output_directory;
  MandarinDuckMode mode;
  uint32_t num_benchmark_outputs;
  const char* benchmark_name;
} typedef MandarinDuckCreateArgs;

struct MandarinDuck {
  MandarinDuckMode mode;
  LuminaryHost* host;
  Display* display;
  const char* output_directory;
  LuminaryOutputPromiseHandle* benchmark_output_promises;
  const char* benchmark_name;
} typedef MandarinDuck;

void mandarin_duck_create(MandarinDuck** duck, MandarinDuckCreateArgs args);
void mandarin_duck_run(MandarinDuck* duck);
void mandarin_duck_destroy(MandarinDuck** duck);

#endif /* MANDARIN_DUCK_INSTANCE_H */
