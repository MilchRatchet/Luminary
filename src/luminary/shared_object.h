#ifndef LUMINARY_SHARED_OBJECT_H
#define LUMINARY_SHARED_OBJECT_H

#define SHARED

#include "utils.h"

enum SharedObjectType {
  SHARED_OBJECT_TYPE_DEFAULT,
  SHARED_OBJECT_TYPE_MEMORY,

  SHARED_OBJECT_TYPE_COUNT
} typedef SharedObjectType;

typedef LuminaryResult (*SharedObjectDestructor)(void**);
typedef LuminaryResult (*SharedObjectMemoryDestructor)(void**, const char* buf_name, const char* func, uint32_t line);

struct SharedObjectCreateInfo {
  SharedObjectType type;
  union {
    SharedObjectDestructor destructor;
    SharedObjectMemoryDestructor memory_destructor;
  };
} typedef SharedObjectCreateInfo;

#define shared_object_create(ptr, info) _shared_object_create((void***) (ptr), info)
#define shared_object_acquire(ptr) _shared_object_create((void**) (ptr))
#define shared_object_destroy(ptr) _shared_object_destroy((void***) (ptr), (const char*) #ptr, (const char*) __func__, __LINE__)

LuminaryResult _shared_object_create(void*** ptr, SharedObjectCreateInfo info);
LuminaryResult _shared_object_acquire(void** ptr);
LuminaryResult _shared_object_destroy(void*** ptr, const char* buf_name, const char* func, uint32_t line);

#endif /* LUMINARY_SHARED_OBJECT_H */
