#ifndef MANDARIN_DUCK_INSTANCE_H
#define MANDARIN_DUCK_INSTANCE_H

#include "display.h"
#include "utils.h"

struct MandarinDuck {
  LuminaryHost* host;
  Display* display;
} typedef MandarinDuck;

void mandarin_duck_create(MandarinDuck** duck, LuminaryHost* host);
void mandarin_duck_run(MandarinDuck* duck);
void mandarin_duck_destroy(MandarinDuck** duck);

#endif /* MANDARIN_DUCK_INSTANCE_H */
