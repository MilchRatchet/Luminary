#ifndef LUMINARY_INTERNAL_ERROR_H
#define LUMINARY_INTERNAL_ERROR_H

#include "utils.h"

#define __RETURN_ERROR(return_code, fmt, ...)      \
  {                                                \
    if ((return_code) & LUMINARY_ERROR_PROPAGATED) \
      log_message(fmt, ##__VA_ARGS__);             \
    else                                           \
      error_message(fmt, ##__VA_ARGS__);           \
    return return_code;                            \
  }

#define __CHECK_NULL_ARGUMENT(argument)                                     \
  if (!(argument)) {                                                        \
    __RETURN_ERROR(LUMINARY_ERROR_ARGUMENT_NULL, "%s is NULL.", #argument); \
  }

#define __FAILURE_HANDLE(command)                                                                             \
  {                                                                                                           \
    LuminaryResult __lum_func_err = (command);                                                                \
    if (__lum_func_err != LUMINARY_SUCCESS) {                                                                 \
      __RETURN_ERROR(                                                                                         \
        __lum_func_err | LUMINARY_ERROR_PROPAGATED, "Luminary internal function [=%s] returned %s", #command, \
        luminary_result_to_string(__lum_func_err));                                                           \
    }                                                                                                         \
  }

////////////////////////////////////////////////////////////////////
// Mutex aware error handling
////////////////////////////////////////////////////////////////////

#define __FAILURE_HANDLE_LOCK_CRITICAL() LuminaryResult __locked_section_result = LUMINARY_SUCCESS;

#define __FAILURE_HANDLE_UNLOCK_CRITICAL() \
  __UNLOCKING_CRITICAL_LABEL:

#define __FAILURE_HANDLE_CRITICAL(command)       \
  {                                              \
    LuminaryResult __lum_func_err = (command);   \
    if (__lum_func_err != LUMINARY_SUCCESS) {    \
      __locked_section_result |= __lum_func_err; \
      goto __UNLOCKING_CRITICAL_LABEL;           \
    }                                            \
  }

#define __FAILURE_HANDLE_CHECK_CRITICAL()                                                    \
  if (__locked_section_result != LUMINARY_SUCCESS) {                                         \
    __RETURN_ERROR(                                                                          \
      __locked_section_result | LUMINARY_ERROR_PROPAGATED, "Error in critical section: %s.", \
      luminary_result_to_string(__locked_section_result));                                   \
  }

#define __RETURN_ERROR_CRITICAL(return_code, fmt, ...) \
  {                                                    \
    if ((return_code) & LUMINARY_ERROR_PROPAGATED)     \
      log_message(fmt, ##__VA_ARGS__);                 \
    else                                               \
      error_message(fmt, ##__VA_ARGS__);               \
    __locked_section_result |= return_code;            \
    goto __UNLOCKING_CRITICAL_LABEL;                   \
  }

////////////////////////////////////////////////////////////////////
// Debugging
////////////////////////////////////////////////////////////////////

#ifdef LUM_DEBUG

#define __DEBUG_ASSERT(condition)                                                           \
  if ((condition) == false) {                                                               \
    __RETURN_ERROR(LUMINARY_ERROR_DEBUG_ASSERT, "Condition: " #condition " was violated."); \
  }

#else /* LUM_DEBUG */

#define __DEBUG_ASSERT(condition)

#endif /* !LUM_DEBUG */

#endif /* LUMINARY_INTERNAL_ERROR_H */
