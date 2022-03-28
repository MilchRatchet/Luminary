#ifndef LUMINARY_ERROR_H
#define LUMINARY_ERROR_H

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>

#include "log.h"

#undef assert
#define assert(ans, message, _abort)  \
  {                                   \
    if (!(ans)) {                     \
      if (_abort) {                   \
        crash_message("%s", message); \
      }                               \
      else {                          \
        error_message("%s", message); \
      }                               \
    }                                 \
  }

#define safe_realloc(ptr, size) ___s_realloc((ptr), (size));

inline void* ___s_realloc(void* ptr, const size_t size) {
  if (size == 0)
    return (void*) 0;
  void* new_ptr = realloc(ptr, size);
  assert((unsigned long long) new_ptr, "Reallocation failed!", 1);
  return new_ptr;
}

#endif /* ERROR_H */
