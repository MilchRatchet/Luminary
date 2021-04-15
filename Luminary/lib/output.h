#ifndef OUTPUT_H
#define OUTPUT_H

#include "utils.h"

void offline_output(
  Scene scene, raytrace_instance* instance, char* output_name, int progress, clock_t time);
void realtime_output(Scene scene, raytrace_instance* instance, const int filters);

#endif /* OUTPUT_H */
