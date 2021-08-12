#ifndef OUTPUT_H
#define OUTPUT_H

#include <time.h>

#include "utils.h"

void offline_output(Scene scene, RaytraceInstance* instance, char* output_name, clock_t time);
void realtime_output(Scene scene, RaytraceInstance* instance);

#endif /* OUTPUT_H */
