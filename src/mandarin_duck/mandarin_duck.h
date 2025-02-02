#ifndef MANDARIN_DUCK_INSTANCE_H
#define MANDARIN_DUCK_INSTANCE_H

#include "display.h"
#include "utils.h"

struct MandarinDuckCreateArgs {
  LuminaryHost* host;
  const char* output_directory;
} typedef MandarinDuckCreateArgs;

struct MandarinDuck {
  LuminaryHost* host;
  Display* display;
  const char* output_directory;
} typedef MandarinDuck;

void mandarin_duck_create(MandarinDuck** duck, MandarinDuckCreateArgs args);
void mandarin_duck_run(MandarinDuck* duck);
void mandarin_duck_destroy(MandarinDuck** duck);

#endif /* MANDARIN_DUCK_INSTANCE_H */
