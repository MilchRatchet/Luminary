#ifndef LUMINARY_INTERNAL_HOST_H
#define LUMINARY_INTERNAL_HOST_H

#include "device/device_manager.h"
#include "mesh.h"
#include "thread.h"
#include "utils.h"

struct LuminaryHost {
  DeviceManager* device_manager;
  Queue* work_queue;
  RingBuffer* ring_buffer;
  WallTime* queue_wall_time;
  Thread* work_thread;
  ARRAY Mesh** meshes;
  ARRAY Material** materials;
  bool enable_output;
  Camera camera;
  Camera camera_external;
} typedef LuminaryHost;

#endif /* LUMINARY_INTERNAL_HOST_H */
