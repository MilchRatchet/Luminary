#ifndef LUMINARY_KERNEL_H
#define LUMINARY_KERNEL_H

#include "device_utils.h"

struct Device typedef Device;

enum CUDAKernelType {
  CUDA_KERNEL_TYPE_SKY_SHADING,

  CUDA_KERNEL_TYPE_COUNT
} typedef CUDAKernelType;

struct CUDAKernel {
  CUkernel cuda_kernel;
  uint32_t shared_memory_size;
} typedef CUDAKernel;

LuminaryResult kernel_create(CUDAKernel** kernel, Device* device, CUlibrary library, CUDAKernelType type);
LuminaryResult kernel_execute(CUDAKernel* kernel);
LuminaryResult kernel_execute_custom(
  CUDAKernel* kernel, uint32_t threads_per_block, uint32_t blocks_per_grid, void* arg_struct, size_t arg_struct_size);
LuminaryResult kernel_destroy(CUDAKernel** kernel);

#endif /* LUMINARY_KERNEL_H */
