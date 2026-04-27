#ifndef LUMINARY_INTERNAL_ERROR_H
#define LUMINARY_INTERNAL_ERROR_H

#include "error_registry.h"
#include "utils.h"

#define __RETURN_ERROR(__lum_kind, __lum_fmt, ...)                      \
  {                                                                     \
    LuminaryResult __lum_result;                                        \
    error_registry_allocate(&__lum_result);                             \
                                                                        \
    error_registry_set_kind(__lum_result, __lum_kind);                  \
    error_registry_set_message(__lum_result, __lum_fmt, ##__VA_ARGS__); \
    error_registry_add_stacktrace(__lum_result);                        \
                                                                        \
    return __lum_result;                                                \
  }

#define __CHECK_NULL_ARGUMENT(__lum_argument)                                     \
  if (!(__lum_argument)) {                                                        \
    __RETURN_ERROR(LUMINARY_ERROR_ARGUMENT_NULL, "%s is NULL.", #__lum_argument); \
  }

#define __FAILURE_HANDLE(__lum_command)            \
  {                                                \
    LuminaryResult __lum_result = (__lum_command); \
    if (__lum_result != LUMINARY_SUCCESS) {        \
      error_registry_add_stacktrace(__lum_result); \
      return __lum_result;                         \
    }                                              \
  }

////////////////////////////////////////////////////////////////////
// Mutex aware error handling
////////////////////////////////////////////////////////////////////

#define __FAILURE_HANDLE_LOCK_CRITICAL() LuminaryResult __locked_section_result = LUMINARY_SUCCESS;

#define __FAILURE_HANDLE_UNLOCK_CRITICAL() \
  __UNLOCKING_CRITICAL_LABEL:

#define __FAILURE_HANDLE_CRITICAL(__lum_command)   \
  {                                                \
    LuminaryResult __lum_result = (__lum_command); \
    if (__lum_result != LUMINARY_SUCCESS) {        \
      __locked_section_result = __lum_result;      \
      goto __UNLOCKING_CRITICAL_LABEL;             \
    }                                              \
  }

#define __FAILURE_HANDLE_CHECK_CRITICAL()            \
  {                                                  \
    if (__locked_section_result != LUMINARY_SUCCESS) \
      return __locked_section_result;                \
  }

#define __RETURN_ERROR_CRITICAL(__lum_kind, __lum_fmt, ...)             \
  {                                                                     \
    LuminaryResult __lum_result;                                        \
    error_registry_allocate(&__lum_result);                             \
                                                                        \
    error_registry_set_kind(__lum_result, __lum_kind);                  \
    error_registry_set_message(__lum_result, __lum_fmt, ##__VA_ARGS__); \
    error_registry_add_stacktrace(__lum_result);                        \
                                                                        \
    __locked_section_result = __lum_result;                             \
    goto __UNLOCKING_CRITICAL_LABEL;                                    \
  }

////////////////////////////////////////////////////////////////////
// Debugging
////////////////////////////////////////////////////////////////////

#ifdef LUM_DEBUG

#define __DEBUG_ASSERT(__lum_condition)                                                           \
  if ((__lum_condition) == false) {                                                               \
    __RETURN_ERROR(LUMINARY_ERROR_DEBUG_ASSERT, "Condition: " #__lum_condition " was violated."); \
  }

#else /* LUM_DEBUG */

#define __DEBUG_ASSERT(__lum_condition) (void) (__lum_condition)

#endif /* !LUM_DEBUG */

#endif /* LUMINARY_INTERNAL_ERROR_H */
