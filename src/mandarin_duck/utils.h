#ifndef MANDARIN_DUCK_UTILS_H
#define MANDARIN_DUCK_UTILS_H

#define LUMINARY_INCLUDE_EXTRA_UTILS

#include <luminary/luminary.h>

#define LUM_FAILURE_HANDLE(command)                                                           \
  {                                                                                           \
    LuminaryResult __lum_res = command;                                                       \
    if (__lum_res != LUMINARY_SUCCESS) {                                                      \
      crash_message("Luminary API returned error: %s", luminary_result_to_string(__lum_res)); \
    }                                                                                         \
  }

#define MD_CHECK_NULL_ARGUMENT(arg)     \
  if (!(arg)) {                         \
    crash_message("%s is NULL.", #arg); \
  }

#endif /* MANDARIN_DUCK_UTILS_H */
