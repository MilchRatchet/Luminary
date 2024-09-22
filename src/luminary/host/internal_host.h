#ifndef LUMINARY_INTERNAL_HOST_H
#define LUMINARY_INTERNAL_HOST_H

#include "device/device.h"
#include "mesh.h"
#include "thread.h"
#include "utils.h"

struct LuminaryHost {
  ARRAY Device* devices;
  Thread* work_thread;
  Queue* work_queue;
  RingBuffer* ring_buffer;
  Camera camera;
  Camera camera_external;
  WallTime* queue_wall_time;
  const char* current_work_string;
  ARRAY Mesh** meshes;
  ARRAY Material** materials;
  bool enable_output;
} typedef LuminaryHost;

#endif /* LUMINARY_INTERNAL_HOST_H */
