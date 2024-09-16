#ifndef LUMINARY_WALL_TIME_H
#define LUMINARY_WALL_TIME_H

#include <luminary/api_utils.h>
#include <luminary/error.h>

struct LuminaryWallTime;
typedef struct LuminaryWallTime LuminaryWallTime;

LUMINARY_API LuminaryResult wall_time_create(LuminaryWallTime** wall_time);
LUMINARY_API LuminaryResult wall_time_set_string(LuminaryWallTime* wall_time, const char* string);
LUMINARY_API LuminaryResult wall_time_get_string(LuminaryWallTime* wall_time, const char** string);
LUMINARY_API LuminaryResult wall_time_start(LuminaryWallTime* wall_time);
LUMINARY_API LuminaryResult wall_time_get_time(LuminaryWallTime* wall_time, double* time);
LUMINARY_API LuminaryResult wall_time_stop(LuminaryWallTime* wall_time);
LUMINARY_API LuminaryResult wall_time_destroy(LuminaryWallTime** wall_time);

#endif /* LUMINARY_WALL_TIME_H */
