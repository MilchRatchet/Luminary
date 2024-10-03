#ifndef LUMINARY_DEVICE_H
#define LUMINARY_DEVICE_H

#include "device_utils.h"

struct OptixKernel;

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
  struct OptixKernel* optix_kernel;
  struct OptixKernel* optix_kernel_geometry;
  struct OptixKernel* optix_kernel_volume;
  struct OptixKernel* optix_kernel_particle;
  struct OptixKernel* optix_kernel_volume_bridges;
} typedef Device;

void _device_init(void);

LuminaryResult device_create(Device** device, uint32_t index);
LuminaryResult device_destroy(Device** device);

#endif /* LUMINARY_DEVICE_H */
