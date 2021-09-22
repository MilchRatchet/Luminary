#ifndef DOUGH_H
#define DOUGH_H

#include "utils.h"

RaytraceInstance* load_baked(const char* filename);
void serialize_baked(RaytraceInstance* instance);

#endif /* DOUGH */
