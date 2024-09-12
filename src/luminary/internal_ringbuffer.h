#ifndef LUMINARY_INTERNAL_RINGBUFFER_H
#define LUMINARY_INTERNAL_RINGBUFFER_H

#include "utils.h"

struct RingBuffer {
  void* memory;
  size_t size;
  size_t total_allocated_memory;
  size_t ptr;
  size_t last_entry_size;
} typedef RingBuffer;

#endif /* LUMINARY_INTERNAL_RINGBUFFER_H */
