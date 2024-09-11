#ifndef LUMINARY_INTERNAL_HOST_H
#define LUMINARY_INTERNAL_HOST_H

#include "device.h"
#include "internal_queue.h"
#include "utils.h"

LUMINARY_API struct LuminaryHost {
  ARRAY Device* devices;
  Queue* work_queue;
  LuminaryCamera camera;
  LuminaryCamera camera_external;
  bool enable_output;
} typedef LuminaryHost;

typedef LuminaryHost Host;

#endif /* LUMINARY_INTERNAL_HOST_H */
