#ifndef LUMINARY_DEVICE_H
#define LUMINARY_DEVICE_H

#include "device_utils.h"
#include "kernel.h"
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
  char name[256];
  DeviceArch arch;
  uint32_t rt_core_version;
  size_t memory_size;
} typedef DeviceProperties;

struct DeviceConstantMemoryDirtyProperties {
  bool is_dirty;
  bool update_everything;
  DeviceConstantMemoryMember member;
} typedef DeviceConstantMemoryDirtyProperties;

struct Device {
  uint32_t index;
  DeviceProperties properties;
  SampleCountSlice sample_count;
  bool exit_requested;
  bool optix_callback_error;
  CUdevice cuda_device;
  CUcontext cuda_ctx;
  CUDAKernel* cuda_kernels[CUDA_KERNEL_TYPE_COUNT];
  CUdeviceptr cuda_device_const_memory;
  OptixDeviceContext optix_ctx;
  OptixKernel* optix_kernels[OPTIX_KERNEL_TYPE_COUNT];
  CUstream stream_main;
  CUstream stream_secondary;
  DevicePointers buffers;
  STAGING DeviceConstantMemory* constant_memory;
  DeviceConstantMemoryDirtyProperties constant_memory_dirty;
  DeviceTexture* moon_albedo_tex;
  DeviceTexture* moon_normal_tex;
} typedef Device;

void _device_init(void);
void _device_shutdown(void);

LuminaryResult device_create(Device** device, uint32_t index);
LuminaryResult device_compile_kernels(Device* device, CUlibrary library);
LuminaryResult device_load_embedded_data(Device* device);
LuminaryResult device_update_scene_entity(Device* device, const void* object, SceneEntity entity);
LuminaryResult device_allocate_work_buffers(Device* device);
LuminaryResult device_destroy(Device** device);

#endif /* LUMINARY_DEVICE_H */
