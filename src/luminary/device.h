#ifndef LUMINARY_DEVICE_H
#define LUMINARY_DEVICE_H

#include <luminary/queue.h>
#include <stdint.h>

// Set of architectures supported by Luminary
enum DeviceArch {
  DEVICE_ARCH_UNKNOWN   = 0,
  DEVICE_ARCH_PASCAL    = 1,
  DEVICE_ARCH_VOLTA     = 11,
  DEVICE_ARCH_TURING    = 2,
  DEVICE_ARCH_AMPERE    = 3,
  DEVICE_ARCH_ADA       = 4,
  DEVICE_ARCH_HOPPER    = 41,
  DEVICE_ARCH_BLACKWELL = 5
} typedef DeviceArch;

struct DeviceProperties {
  const char* name;
  DeviceArch arch;
  size_t memory_size;
} typedef DeviceProperties;

struct Device {
  uint32_t index;
  DeviceProperties properties;
  Queue* work_queue;
} typedef Device;

#endif /* LUMINARY_DEVICE_H */
