#ifndef LUMINARY_INTERNAL_WALLTIME_H
#define LUMINARY_INTERNAL_WALLTIME_H

#include "utils.h"

LuminaryResult _thread_status_get_timestamp(uint64_t* time);
double _thread_status_get_elapsed_time(uint64_t start);

#endif /* LUMINARY_INTERNAL_WALLTIME_H */
