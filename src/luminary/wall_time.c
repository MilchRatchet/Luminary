#include "internal_error.h"
#include "utils.h"

// TODO: Use timespec_get and timespec_getres instead.

// windows.h must be included before any other windows header
#ifdef _WIN32
#include <windows.h>
#endif

#ifdef _WIN32
#include <profileapi.h>
#else
#include <time.h>
#endif

struct LuminaryWallTime {
  const char* string;
  uint64_t time_point;
  double time;
};

static uint64_t _wall_time_get_time(void) {
#ifdef _WIN32
  LARGE_INTEGER ticks;
  if (!QueryPerformanceCounter(&ticks))
    return 0;
  return ticks.QuadPart;
#else
  return (uint64_t) clock();
#endif
}

static double _wall_time_get_time_diff(const uint64_t t0, const uint64_t t1) {
#ifdef _WIN32
  LARGE_INTEGER freq;
  if (!QueryPerformanceFrequency(&freq))
    return 0.0;

  const double t_freq = ((double) freq.QuadPart);

  return (t1 - t0) / t_freq;
#else
  return ((double) (t1 - t0)) / (CLOCKS_PER_SEC);
#endif
}

LuminaryResult wall_time_create(WallTime** _wall_time) {
  if (!_wall_time) {
    __RETURN_ERROR(LUMINARY_ERROR_ARGUMENT_NULL, "Wall time is NULL.");
  }

  WallTime* wall_time;

  __FAILURE_HANDLE(host_malloc(&wall_time, sizeof(WallTime)));

  wall_time->string     = (const char*) 0;
  wall_time->time_point = 0;
  wall_time->time       = 0.0;

  *_wall_time = wall_time;

  return LUMINARY_SUCCESS;
}

LuminaryResult wall_time_set_string(WallTime* wall_time, const char* string) {
  if (!wall_time) {
    __RETURN_ERROR(LUMINARY_ERROR_ARGUMENT_NULL, "Wall time is NULL.");
  }

  wall_time->string = string;

  return LUMINARY_SUCCESS;
}

LuminaryResult wall_time_get_string(WallTime* wall_time, const char** string) {
  if (!wall_time) {
    __RETURN_ERROR(LUMINARY_ERROR_ARGUMENT_NULL, "Wall time is NULL.");
  }

  if (!string) {
    __RETURN_ERROR(LUMINARY_ERROR_ARGUMENT_NULL, "String ptr is NULL.");
  }

  *string = wall_time->string;

  return LUMINARY_SUCCESS;
}

LuminaryResult wall_time_start(WallTime* wall_time) {
  if (!wall_time) {
    __RETURN_ERROR(LUMINARY_ERROR_ARGUMENT_NULL, "Wall time is NULL.");
  }

  wall_time->time_point = _wall_time_get_time();

  return LUMINARY_SUCCESS;
}

LuminaryResult wall_time_get_time(WallTime* wall_time, double* time) {
  if (!wall_time) {
    __RETURN_ERROR(LUMINARY_ERROR_ARGUMENT_NULL, "Wall time is NULL.");
  }

  if (wall_time->time_point != 0) {
    wall_time->time = _wall_time_get_time_diff(wall_time->time_point, _wall_time_get_time());
  }

  *time = wall_time->time;

  return LUMINARY_SUCCESS;
}

LuminaryResult wall_time_stop(WallTime* wall_time) {
  if (!wall_time) {
    __RETURN_ERROR(LUMINARY_ERROR_ARGUMENT_NULL, "Wall time is NULL.");
  }

  wall_time->time = (wall_time->time_point != 0) ? _wall_time_get_time_diff(wall_time->time_point, _wall_time_get_time()) : 0.0f;

  wall_time->time_point = 0;

  return LUMINARY_SUCCESS;
}

LuminaryResult wall_time_destroy(WallTime** wall_time) {
  if (!wall_time) {
    __RETURN_ERROR(LUMINARY_ERROR_ARGUMENT_NULL, "Wall time ptr is NULL.");
  }

  if (!(*wall_time)) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Wall time is NULL.");
  }

  __FAILURE_HANDLE(host_free(wall_time));

  return LUMINARY_SUCCESS;
}

LuminaryResult _wall_time_get_timestamp(uint64_t* time) {
  __CHECK_NULL_ARGUMENT(time);

  *time = _wall_time_get_time();

  return LUMINARY_SUCCESS;
}
