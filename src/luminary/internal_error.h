#ifndef LUMINARY_INTERNAL_ERROR_H
#define LUMINARY_INTERNAL_ERROR_H

#include "utils.h"

#define __RETURN_ERROR(return_code, fmt, ...)    \
  {                                              \
    if (return_code & LUMINARY_ERROR_PROPAGATED) \
      log_message(fmt, ##__VA_ARGS__);           \
    else                                         \
      error_message(fmt, ##__VA_ARGS__);         \
    return return_code;                          \
  }

#define __CHECK_NULL_ARGUMENT(argument)                                     \
  if (!(argument)) {                                                        \
    __RETURN_ERROR(LUMINARY_ERROR_ARGUMENT_NULL, "%s is NULL.", #argument); \
  }

#define __FAILURE_HANDLE(command)                                                                             \
  {                                                                                                           \
    LuminaryResult __lum_func_err = command;                                                                  \
    if (__lum_func_err != 0) {                                                                                \
      __RETURN_ERROR(                                                                                         \
        __lum_func_err | LUMINARY_ERROR_PROPAGATED, "Luminary internal function [=%s] returned %s", #command, \
        luminary_result_to_string(__lum_func_err));                                                           \
    }                                                                                                         \
  }

#endif /* LUMINARY_INTERNAL_ERROR_H */
