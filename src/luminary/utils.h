#ifndef LUMINARY_UTILS_H
#define LUMINARY_UTILS_H

#include "internal_api_resolve.h"

// API definitions must first be translated

#include <assert.h>

#include "sky_defines.h"

struct QueueEntry {
  const char* name;
  LuminaryResult (*function)(void* worker, void* args);
  void* args;
} typedef QueueEntry;

#ifndef PI
#define PI 3.141592653589f
#endif

#define LUM_STATIC_SIZE_ASSERT(struct, size) static_assert(sizeof(struct) == size, #struct " has invalid size");

#endif /* LUMINARY_UTILS_H */
