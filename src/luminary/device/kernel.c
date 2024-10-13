#include "kernel.h"

#include "internal_error.h"

struct CUDAKernelConfig {
  const char* name;
  uint32_t shared_memory_size;
} typedef CUDAKernelConfig;

static const CUDAKernelConfig cuda_kernel_configs[CUDA_KERNEL_TYPE_COUNT] = {
  [CUDA_KERNEL_TYPE_SKY_SHADING] = {.name = "sky_process_tasks", .shared_memory_size = 0}};

LuminaryResult kernel_create(CUDAKernel** kernel, Device* device, CUlibrary library, CUDAKernelType type) {
  __CHECK_NULL_ARGUMENT(kernel);
  __CHECK_NULL_ARGUMENT(device);
  __CHECK_NULL_ARGUMENT(library);

  __FAILURE_HANDLE(host_malloc(kernel, sizeof(CUDAKernel)));

  const CUDAKernelConfig* config = cuda_kernel_configs + type;

  CUDA_DRIVER_FAILURE_HANDLE(cuLibraryGetKernel(&(*kernel)->cuda_kernel, library, config->name));

  (*kernel)->shared_memory_size = config->shared_memory_size;

  return LUMINARY_SUCCESS;
}

LuminaryResult kernel_execute(CUDAKernel* kernel) {
  __CHECK_NULL_ARGUMENT(kernel);

  CUlaunchConfig launch_config;

  launch_config.blockDimX      = THREADS_PER_BLOCK;
  launch_config.blockDimY      = 1;
  launch_config.blockDimZ      = 1;
  launch_config.gridDimX       = BLOCKS_PER_GRID;
  launch_config.gridDimY       = 1;
  launch_config.gridDimZ       = 1;
  launch_config.sharedMemBytes = kernel->shared_memory_size;
  launch_config.hStream        = (CUstream) 0;
  launch_config.attrs          = (CUlaunchAttribute*) 0;
  launch_config.numAttrs       = 0;

  CUDA_DRIVER_FAILURE_HANDLE(cuLaunchKernelEx(&launch_config, (CUfunction) kernel->cuda_kernel, (void**) 0, (void**) 0));

  return LUMINARY_SUCCESS;
}

LuminaryResult kernel_execute_custom(
  CUDAKernel* kernel, uint32_t threads_per_block, uint32_t blocks_per_grid, void* arg_struct, size_t arg_struct_size) {
  __CHECK_NULL_ARGUMENT(kernel);

  CUlaunchConfig launch_config;

  launch_config.blockDimX      = threads_per_block;
  launch_config.blockDimY      = 1;
  launch_config.blockDimZ      = 1;
  launch_config.gridDimX       = blocks_per_grid;
  launch_config.gridDimY       = 1;
  launch_config.gridDimZ       = 1;
  launch_config.sharedMemBytes = kernel->shared_memory_size;
  launch_config.hStream        = (CUstream) 0;
  launch_config.attrs          = (CUlaunchAttribute*) 0;
  launch_config.numAttrs       = 0;

  void* kernel_args[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, arg_struct, CU_LAUNCH_PARAM_BUFFER_SIZE, &arg_struct_size, CU_LAUNCH_PARAM_END};

  CUDA_DRIVER_FAILURE_HANDLE(cuLaunchKernelEx(&launch_config, (CUfunction) kernel->cuda_kernel, (void**) 0, kernel_args));

  return LUMINARY_SUCCESS;
}

LuminaryResult kernel_destroy(CUDAKernel** kernel) {
  __CHECK_NULL_ARGUMENT(kernel);
  __CHECK_NULL_ARGUMENT(*kernel);

  __FAILURE_HANDLE(host_free(kernel));

  return LUMINARY_SUCCESS;
}
