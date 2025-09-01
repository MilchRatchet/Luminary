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

struct LuminaryThreadStatus {
  const char* name;
  const char* string;
  uint64_t time_point;
  double time;
};

static uint64_t _thread_status_get_time(void) {
#ifdef _WIN32
  LARGE_INTEGER ticks;
  if (!QueryPerformanceCounter(&ticks))
    return 0;
  return ticks.QuadPart;
#else
  return (uint64_t) clock();
#endif
}

static double _thread_status_get_time_diff(const uint64_t t0, const uint64_t t1) {
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

LuminaryResult thread_status_create(ThreadStatus** thread_status) {
  __CHECK_NULL_ARGUMENT(thread_status);

  __FAILURE_HANDLE(host_malloc(thread_status, sizeof(ThreadStatus)));
  memset(*thread_status, 0, sizeof(ThreadStatus));

  return LUMINARY_SUCCESS;
}

LUMINARY_API LuminaryResult thread_status_set_worker_name(LuminaryThreadStatus* thread_status, const char* name) {
  __CHECK_NULL_ARGUMENT(thread_status);

  thread_status->name = name;

  return LUMINARY_SUCCESS;
}

LUMINARY_API LuminaryResult thread_status_get_worker_name(LuminaryThreadStatus* thread_status, const char** name) {
  __CHECK_NULL_ARGUMENT(thread_status);
  __CHECK_NULL_ARGUMENT(name);

  *name = thread_status->name;

  return LUMINARY_SUCCESS;
}

LuminaryResult thread_status_get_string(ThreadStatus* thread_status, const char** string) {
  __CHECK_NULL_ARGUMENT(thread_status);
  __CHECK_NULL_ARGUMENT(string);

  *string = thread_status->string;

  return LUMINARY_SUCCESS;
}

LuminaryResult thread_status_start(ThreadStatus* thread_status, const char* string) {
  __CHECK_NULL_ARGUMENT(thread_status);

  thread_status->time_point = _thread_status_get_time();
  thread_status->string     = string;

  return LUMINARY_SUCCESS;
}

LuminaryResult thread_status_get_time(ThreadStatus* thread_status, double* time) {
  __CHECK_NULL_ARGUMENT(thread_status);
  __CHECK_NULL_ARGUMENT(time);

  if (thread_status->time_point != 0) {
    thread_status->time = _thread_status_get_time_diff(thread_status->time_point, _thread_status_get_time());
  }

  *time = thread_status->time;

  return LUMINARY_SUCCESS;
}

LuminaryResult thread_status_stop(ThreadStatus* thread_status) {
  __CHECK_NULL_ARGUMENT(thread_status);

  thread_status->time = (thread_status->time_point != 0) ? _thread_status_get_time_diff(thread_status->time_point, _thread_status_get_time()) : 0.0f;

  thread_status->time_point = 0;
  thread_status->string     = (const char*) 0;

  return LUMINARY_SUCCESS;
}

LuminaryResult thread_status_destroy(ThreadStatus** thread_status) {
  __CHECK_NULL_ARGUMENT(thread_status);
  __CHECK_NULL_ARGUMENT(*thread_status);

  __FAILURE_HANDLE(host_free(thread_status));

  return LUMINARY_SUCCESS;
}

LuminaryResult _thread_status_get_timestamp(uint64_t* time) {
  __CHECK_NULL_ARGUMENT(time);

  *time = _thread_status_get_time();

  return LUMINARY_SUCCESS;
}
