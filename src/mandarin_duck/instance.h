#ifndef MANDARIN_DUCK_INSTANCE_H
#define MANDARIN_DUCK_INSTANCE_H

#include "display.h"
#include "utils.h"

struct Instance {
  LuminaryHost* host;
  Display* display;
} typedef Instance;

void instance_create(Instance** instance, LuminaryHost* host);
void instance_run(Instance* instance);
void instance_destroy(Instance** instance);

#endif /* MANDARIN_DUCK_INSTANCE_H */
