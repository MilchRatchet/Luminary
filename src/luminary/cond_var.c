#include "cond_var.h"

#include <threads.h>

#include "internal_error.h"

struct ConditionVariable {
  cnd_t _cond_var;
};

LuminaryResult condition_variable_create(ConditionVariable** cond_var) {
  if (!cond_var) {
    __RETURN_ERROR(LUMINARY_ERROR_ARGUMENT_NULL, "Condition variable is NULL.");
  }

  __FAILURE_HANDLE(host_malloc(cond_var, sizeof(ConditionVariable)));

  const int retval = cnd_init((cnd_t*) *cond_var);

  if (retval != thrd_success) {
    __RETURN_ERROR(LUMINARY_ERROR_C_STD, "cnd_init returned an error.");
  }

  return LUMINARY_SUCCESS;
}

LuminaryResult condition_variable_signal(ConditionVariable* cond_var) {
  if (!cond_var) {
    __RETURN_ERROR(LUMINARY_ERROR_ARGUMENT_NULL, "Condition variable is NULL.");
  }

  const int retval = cnd_signal((cnd_t*) cond_var);

  if (retval != thrd_success) {
    __RETURN_ERROR(LUMINARY_ERROR_C_STD, "cnd_signal returned an error.");
  }

  return LUMINARY_SUCCESS;
}

LuminaryResult condition_variable_broadcast(ConditionVariable* cond_var) {
  if (!cond_var) {
    __RETURN_ERROR(LUMINARY_ERROR_ARGUMENT_NULL, "Condition variable is NULL.");
  }

  const int retval = cnd_broadcast((cnd_t*) cond_var);

  if (retval != thrd_success) {
    __RETURN_ERROR(LUMINARY_ERROR_C_STD, "cnd_broadcast returned an error.");
  }

  return LUMINARY_SUCCESS;
}

LuminaryResult condition_variable_wait(ConditionVariable* cond_var, Mutex* mutex) {
  if (!cond_var) {
    __RETURN_ERROR(LUMINARY_ERROR_ARGUMENT_NULL, "Condition variable is NULL.");
  }

  if (!mutex) {
    __RETURN_ERROR(LUMINARY_ERROR_ARGUMENT_NULL, "Mutex is NULL.");
  }

  const int retval = cnd_wait((cnd_t*) cond_var, (mtx_t*) mutex);

  if (retval != thrd_success) {
    __RETURN_ERROR(LUMINARY_ERROR_C_STD, "cnd_signal returned an error.");
  }

  return LUMINARY_SUCCESS;
}

LuminaryResult condition_variable_destroy(ConditionVariable** cond_var) {
  if (!cond_var) {
    __RETURN_ERROR(LUMINARY_ERROR_ARGUMENT_NULL, "Condition variable is NULL.");
  }

  // Make sure that other threads are not blocking on this condition variable.
  __FAILURE_HANDLE(condition_variable_broadcast(*cond_var));

  cnd_destroy((cnd_t*) *cond_var);

  __FAILURE_HANDLE(host_free(cond_var));

  return LUMINARY_SUCCESS;
}
