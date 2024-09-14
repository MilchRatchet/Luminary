#ifndef LUMINARY_INTERNAL_HOST_H
#define LUMINARY_INTERNAL_HOST_H

#include "device.h"
#include "internal_queue.h"
#include "utils.h"

struct LuminaryHost {
  ARRAY Device* devices;
  Queue* work_queue;
  RingBuffer* ring_buffer;
  LuminaryCamera camera;
  LuminaryCamera camera_external;
  const char* current_work_string;
  bool enable_output;
  bool exit_requested;
} typedef LuminaryHost;

#endif /* LUMINARY_INTERNAL_HOST_H */
