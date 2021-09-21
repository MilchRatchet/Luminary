#ifndef DOUGH_H
#define DOUGH_H

#include "utils.h"

Scene load_baked(const char* filename, RaytraceInstance** instance);
void serialize_baked(RaytraceInstance* instance);

#endif /* DOUGH */
