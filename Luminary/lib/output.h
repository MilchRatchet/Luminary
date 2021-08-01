#ifndef OUTPUT_H
#define OUTPUT_H

#include "utils.h"
#include <time.h>

void offline_output(
  Scene scene, RaytraceInstance* instance, char* output_name, int progress, clock_t time);
void realtime_output(Scene scene, RaytraceInstance* instance, const int filters);

#endif /* OUTPUT_H */
