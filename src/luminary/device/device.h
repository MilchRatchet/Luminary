#ifndef LUMINARY_DEVICE_H
#define LUMINARY_DEVICE_H

#include "device_utils.h"
#include "optixrt.h"

// Set of architectures supported by Luminary
enum DeviceArch {
  DEVICE_ARCH_UNKNOWN   = 0,
  DEVICE_ARCH_PASCAL    = 1,
  DEVICE_ARCH_VOLTA     = 2,
  DEVICE_ARCH_TURING    = 3,
  DEVICE_ARCH_AMPERE    = 4,
  DEVICE_ARCH_ADA       = 5,
  DEVICE_ARCH_HOPPER    = 6,
  DEVICE_ARCH_BLACKWELL = 7
} typedef DeviceArch;

struct DeviceProperties {
  const char name[256];
  DeviceArch arch;
  uint32_t rt_core_version;
  size_t memory_size;
} typedef DeviceProperties;

struct Device {
  uint32_t index;
  DeviceProperties properties;
  uint32_t accumulated_sample_count;
  bool exit_requested;
  bool optix_callback_error;
  OptixDeviceContext optix_ctx;
  OptixKernel* optix_kernels[OPTIX_KERNEL_TYPE_COUNT];
  DevicePointers buffers;
} typedef Device;

void _device_init(void);
void _device_shutdown(void);

LuminaryResult device_create(Device** device, uint32_t index);
LuminaryResult device_compile_kernels(Device* device);
LuminaryResult device_destroy(Device** device);

#endif /* LUMINARY_DEVICE_H */
