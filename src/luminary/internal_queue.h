#ifndef LUMINARY_INTERNAL_QUEUE_H
#define LUMINARY_INTERNAL_QUEUE_H

#include "cond_var.h"
#include "mutex.h"
#include "utils.h"

struct QueueEntry {
  const char* name;
  LuminaryResult (*function)(Host* host, void* args);
  void* args;
} typedef QueueEntry;

struct LuminaryQueue {
  void* buffer;
  size_t element_count;
  size_t element_size;
  size_t read_ptr;
  size_t write_ptr;
  size_t elements_in_queue;
  Mutex* mutex;
  ConditionVariable* cond_var;
} typedef LuminaryQueue;

#endif /* LUMINARY_INTERNAL_QUEUE_H */
