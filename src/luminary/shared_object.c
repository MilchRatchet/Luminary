#include "shared_object.h"

#include <string.h>

#include "internal_error.h"
#include "utils.h"

struct SharedObjectHeader {
  uint64_t magic;
  uint64_t reference_count;
  SharedObjectDestructor destructor;
  SharedObjectMemoryDestructor memory_destructor;
  uint64_t padding[4];
};
LUM_STATIC_SIZE_ASSERT(struct SharedObjectHeader, 64);

// LUMSHAOB
#define SHARED_OBJECT_HEADER_MAGIC (0x424F4148534D554Cull)
#define SHARED_OBJECT_HEADER_FREED_MAGIC (1337ull)

LuminaryResult _shared_object_create(void*** ptr, SharedObjectCreateInfo info) {
  __CHECK_NULL_ARGUMENT(ptr);

  struct SharedObjectHeader* header;
  __FAILURE_HANDLE(host_malloc(&header, sizeof(struct SharedObjectHeader) + sizeof(void**)));

  memset(header, 0, sizeof(struct SharedObjectHeader) + sizeof(void**));

  header->magic           = SHARED_OBJECT_HEADER_MAGIC;
  header->reference_count = 1;

  if (info.type == SHARED_OBJECT_TYPE_DEFAULT) {
    header->destructor = info.destructor;
  }
  else {
    header->memory_destructor = info.memory_destructor;
  }

  *ptr = (void**) (header + 1);

  return LUMINARY_SUCCESS;
}

LuminaryResult _shared_object_acquire(void** ptr) {
  __CHECK_NULL_ARGUMENT(ptr);

  struct SharedObjectHeader* header = ((struct SharedObjectHeader*) ptr) - 1;

  if (header->magic != SHARED_OBJECT_HEADER_MAGIC) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Given object is not a valid shared object.");
  }

  header->reference_count++;

  return LUMINARY_SUCCESS;
}

LuminaryResult _shared_object_destroy(void*** ptr, const char* buf_name, const char* func, uint32_t line) {
  __CHECK_NULL_ARGUMENT(ptr);
  __CHECK_NULL_ARGUMENT(*ptr);

  struct SharedObjectHeader* header = ((struct SharedObjectHeader*) *ptr) - 1;

  if (header->magic != SHARED_OBJECT_HEADER_MAGIC) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Given object is not a valid shared object.");
  }

  if (header->reference_count == 0) {
    __RETURN_ERROR(
      LUMINARY_ERROR_API_EXCEPTION,
      "Shared object does not hold any references, the destroy call is invalid and the object was not correctly destroyed earlier.")
  }

  header->reference_count--;

  if (header->reference_count == 0) {
    if (header->destructor != (SharedObjectDestructor) 0) {
      __FAILURE_HANDLE(header->destructor(*ptr));
    }

    if (header->memory_destructor != (SharedObjectMemoryDestructor) 0) {
      __FAILURE_HANDLE(header->memory_destructor(*ptr, buf_name, func, line));
    }

    memset(header, 0, sizeof(struct SharedObjectHeader) + sizeof(void**));

    // Write a magic into memory to possibly later identify use after free situations.
    header->magic = SHARED_OBJECT_HEADER_FREED_MAGIC;

    __FAILURE_HANDLE(host_free(&header));
  }

  *ptr = (void**) 0;

  return LUMINARY_SUCCESS;
}
