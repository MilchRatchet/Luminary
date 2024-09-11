#ifndef LUMINARY_INTERNAL_QUEUE_H
#define LUMINARY_INTERNAL_QUEUE_H

#include <luminary/queue.h>

#include "host.h"

struct QueueEntry {
  LuminaryResult (*function)(Host* host, void* args);
  void* args;
} typedef QueueEntry;

struct LuminaryQueue {
  QueueEntry* buffer;
  size_t buffer_size;
  size_t read_ptr;
  size_t write_ptr;
} typedef LuminaryQueue;

typedef LuminaryQueue Queue;

#endif /* LUMINARY_INTERNAL_QUEUE_H */
