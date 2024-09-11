#ifndef LUMINARY_INTERNAL_ERROR_H
#define LUMINARY_INTERNAL_ERROR_H

#include "utils.h"

#define __RETURN_ERROR(return_code, fmt, ...) \
  {                                           \
    log_message(fmt, ##__VA_ARGS__);          \
    return return_code;                       \
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
