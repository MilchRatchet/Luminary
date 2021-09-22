#ifndef OUTPUT_H
#define OUTPUT_H

#include <time.h>

#include "utils.h"

void offline_output(RaytraceInstance* instance, clock_t time);
void realtime_output(RaytraceInstance* instance);

#endif /* OUTPUT_H */
