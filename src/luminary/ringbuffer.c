#include <string.h>

#include "internal_error.h"
#include "utils.h"

struct LuminaryRingBuffer {
  void* memory;
  size_t size;
  size_t total_allocated_memory;
  size_t ptr;
  size_t last_entry_size;
};

LuminaryResult _ringbuffer_create(RingBuffer** _buffer, size_t size, const char* buf_name, const char* func, uint32_t line) {
  __CHECK_NULL_ARGUMENT(_buffer);

  if (size == 0) {
    __RETURN_ERROR(LUMINARY_ERROR_INVALID_API_ARGUMENT, "Size was 0.");
  }

  RingBuffer* buffer;
  __FAILURE_HANDLE(_host_malloc((void**) &buffer, sizeof(RingBuffer), buf_name, func, line));

  memset(buffer, 0, sizeof(RingBuffer));

  __FAILURE_HANDLE(_host_malloc((void**) &buffer->memory, size, buf_name, func, line));
  buffer->size                   = size;
  buffer->total_allocated_memory = 0;
  buffer->ptr                    = 0;

  *_buffer = buffer;

  return LUMINARY_SUCCESS;
}

LuminaryResult ringbuffer_allocate_entry(RingBuffer* buffer, size_t entry_size, void** entry) {
  __CHECK_NULL_ARGUMENT(buffer);
  __CHECK_NULL_ARGUMENT(entry);

  if (entry_size == 0) {
    __RETURN_ERROR(LUMINARY_ERROR_INVALID_API_ARGUMENT, "Requested entry size was 0.");
  }

  if (buffer->total_allocated_memory + entry_size > buffer->size) {
    __RETURN_ERROR(LUMINARY_ERROR_OUT_OF_MEMORY, "Insufficient ring buffer size to allocate this entry. Requested size: %zu.", entry_size);
  }

  if (buffer->ptr + entry_size > buffer->size) {
    const size_t total_size_this_allocation = (buffer->size - buffer->ptr) + entry_size;

    if (buffer->total_allocated_memory + total_size_this_allocation > buffer->size) {
      __RETURN_ERROR(
        LUMINARY_ERROR_OUT_OF_MEMORY, "Insufficient ring buffer size to allocate this entry due to fragmentation. Requested size: %zu.",
        entry_size);
    }

    *entry = buffer->memory;

    buffer->ptr = entry_size;
  }
  else {
    *entry = (void*) (((uint8_t*) (buffer->memory)) + entry_size);

    buffer->ptr += entry_size;
  }

  buffer->total_allocated_memory += entry_size;

  return LUMINARY_SUCCESS;
}

LuminaryResult ringbuffer_release_entry(RingBuffer* buffer, size_t entry_size) {
  __CHECK_NULL_ARGUMENT(buffer);

  if (entry_size == 0) {
    __RETURN_ERROR(LUMINARY_ERROR_INVALID_API_ARGUMENT, "Tried to release an entry of size 0.");
  }

  if (buffer->total_allocated_memory < entry_size) {
    __RETURN_ERROR(
      LUMINARY_ERROR_INVALID_API_ARGUMENT, "Tried to release an entry of size %zu when only %zu bytes are currently allocated.", entry_size,
      buffer->total_allocated_memory);
  }

  buffer->total_allocated_memory -= entry_size;

  return LUMINARY_SUCCESS;
}

LuminaryResult _ringbuffer_destroy(RingBuffer** buffer, const char* buf_name, const char* func, uint32_t line) {
  __CHECK_NULL_ARGUMENT(buffer);
  __CHECK_NULL_ARGUMENT(*buffer);

  if ((*buffer)->total_allocated_memory > 0) {
    __RETURN_ERROR(LUMINARY_ERROR_MEMORY_LEAK, "Attempted to destroy ringbuffer that has entries that have not been released.");
  }

  __FAILURE_HANDLE(_host_free((void**) &(*buffer)->memory, buf_name, func, line));
  __FAILURE_HANDLE(_host_free((void**) buffer, buf_name, func, line));

  return LUMINARY_SUCCESS;
}
