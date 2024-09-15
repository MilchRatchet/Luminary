#ifndef OUTPUT_H
#define OUTPUT_H

#include <time.h>

#include "utils.h"

void offline_exit_post_process_menu(RaytraceInstance* instance);
void offline_output(RaytraceInstance* instance);
void realtime_output(RaytraceInstance* instance);

#endif /* OUTPUT_H */
