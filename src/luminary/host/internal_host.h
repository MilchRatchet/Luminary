#ifndef LUMINARY_INTERNAL_HOST_H
#define LUMINARY_INTERNAL_HOST_H

#include "device/device.h"
#include "utils.h"

struct LuminaryHost {
  ARRAY Device* devices;
  Queue* work_queue;
  RingBuffer* ring_buffer;
  Camera camera;
  Camera camera_external;
  WallTime* queue_wall_time;
  const char* current_work_string;
  bool enable_output;
  bool exit_requested;
} typedef LuminaryHost;

#endif /* LUMINARY_INTERNAL_HOST_H */
