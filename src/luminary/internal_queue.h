#ifndef LUMINARY_INTERNAL_QUEUE_H
#define LUMINARY_INTERNAL_QUEUE_H

#include "utils.h"

struct QueueEntry {
  const char* name;
  LuminaryResult (*function)(Host* host, void* args);
  void* args;
} typedef QueueEntry;

struct LuminaryQueue {
  QueueEntry* buffer;
  size_t buffer_size;
  size_t read_ptr;
  size_t write_ptr;
} typedef LuminaryQueue;

#endif /* LUMINARY_INTERNAL_QUEUE_H */
