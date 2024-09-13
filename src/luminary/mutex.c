#include "mutex.h"

#include <math.h>
#include <threads.h>
#include <time.h>

#include "internal_error.h"

typedef mtx_t Mutex;

LuminaryResult mutex_create(Mutex** mutex) {
  if (!mutex) {
    __RETURN_ERROR(LUMINARY_ERROR_ARGUMENT_NULL, "Mutex was NULL.");
  }

  __FAILURE_HANDLE(host_malloc(mutex, sizeof(Mutex)));

  const int retval = mtx_init(*mutex, mtx_plain);

  if (retval != thrd_success) {
    __FAILURE_HANDLE(host_free(mutex));
    __RETURN_ERROR(LUMINARY_ERROR_C_STD, "mtx_init returned an error.");
  }

  return LUMINARY_SUCCESS;
}

LuminaryResult mutex_lock(Mutex* mutex) {
  if (!mutex) {
    __RETURN_ERROR(LUMINARY_ERROR_ARGUMENT_NULL, "Mutex was NULL.");
  }

  const int retval = mtx_lock(mutex);

  if (retval != thrd_success) {
    __RETURN_ERROR(LUMINARY_ERROR_C_STD, "mtx_lock returned an error.");
  }

  return LUMINARY_SUCCESS;
}

LuminaryResult mutex_timed_lock(Mutex* mutex, const double timeout_time, bool* success) {
  if (!mutex) {
    __RETURN_ERROR(LUMINARY_ERROR_ARGUMENT_NULL, "Mutex was NULL.");
  }

  struct timespec ts;
  timespec_get(&ts, TIME_UTC);

  // This is not C23 compliant. C23 only guarantees that the type of tv_nsec can hold
  // values up to 999999999. In practice, this will probably never be an issue.
  ts.tv_sec += floor(timeout_time);
  ts.tv_nsec += (timeout_time - floor(timeout_time)) * 1000000000.0;

  if (ts.tv_nsec > 999999999) {
    ts.tv_nsec -= 1000000000;
    ts.tv_sec += 1;
  }

  const int retval = mtx_timedlock(mutex, &ts);

  if (retval == thrd_error) {
    __RETURN_ERROR(LUMINARY_ERROR_C_STD, "mtx_timedlock returned an error.");
  }

  *success = (retval == thrd_success);

  return LUMINARY_SUCCESS;
}

LuminaryResult mutex_try_lock(Mutex* mutex, bool* success) {
  if (!mutex) {
    __RETURN_ERROR(LUMINARY_ERROR_ARGUMENT_NULL, "Mutex was NULL.");
  }

  const int retval = mtx_trylock(mutex);

  if (retval == thrd_error) {
    __RETURN_ERROR(LUMINARY_ERROR_C_STD, "mtx_trylock returned an error.");
  }

  *success = (retval == thrd_success);

  return LUMINARY_SUCCESS;
}

LuminaryResult mutex_unlock(Mutex* mutex) {
  if (!mutex) {
    __RETURN_ERROR(LUMINARY_ERROR_ARGUMENT_NULL, "Mutex was NULL.");
  }

  const int retval = mtx_unlock(mutex);

  if (retval != thrd_success) {
    __RETURN_ERROR(LUMINARY_ERROR_C_STD, "mtx_unlock returned an error.");
  }

  return LUMINARY_SUCCESS;
}

LuminaryResult mutex_destroy(Mutex** mutex) {
  if (!mutex) {
    __RETURN_ERROR(LUMINARY_ERROR_ARGUMENT_NULL, "Mutex was NULL.");
  }

  // We do not allow for locked mutexes to be destroyed. The C standard
  // gives us undefined behaviour in this case so we need to catch this.
  bool is_not_locked;
  __FAILURE_HANDLE(mutex_try_lock(*mutex, &is_not_locked));

  if (!is_not_locked) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Mutex is still locked.");
  }

  __FAILURE_HANDLE(mutex_unlock(*mutex));

  mtx_destroy(*mutex);

  __FAILURE_HANDLE(host_free(mutex));

  return LUMINARY_SUCCESS;
}
