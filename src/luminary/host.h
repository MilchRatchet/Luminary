#ifndef LUMINARY_HOST_H
#define LUMINARY_HOST_H

#include <luminary/array.h>
#include <luminary/host.h>

#include "device.h"
#include "utils.h"

LUMINARY_API struct LuminaryHost {
  ARRAY Device* devices;
  LuminaryCamera camera;
  bool enable_output;
} typedef LuminaryHost;

typedef LuminaryHost Host;

#endif /* LUMINARY_HOST_H */
