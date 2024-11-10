#include "kernel.h"

#include "internal_error.h"
#include "kernel_args.h"

struct CUDAKernelConfig {
  const char* name;
  uint32_t shared_memory_size;
  size_t arg_size;
} typedef CUDAKernelConfig;

static const CUDAKernelConfig cuda_kernel_configs[CUDA_KERNEL_TYPE_COUNT] = {
  [CUDA_KERNEL_TYPE_GENERATE_TRACE_TASKS]            = {.name = "generate_trace_tasks", .shared_memory_size = 0, .arg_size = 0},
  [CUDA_KERNEL_TYPE_BALANCE_TRACE_TASKS]             = {.name = "balance_trace_tasks", .shared_memory_size = 0, .arg_size = 0},
  [CUDA_KERNEL_TYPE_POSTPROCESS_TRACE_TASKS]         = {.name = "postprocess_trace_tasks", .shared_memory_size = 0, .arg_size = 0},
  [CUDA_KERNEL_TYPE_GEOMETRY_PROCESS_TASKS_DEBUG]    = {.name = "geometry_process_tasks_debug", .shared_memory_size = 0, .arg_size = 0},
  [CUDA_KERNEL_TYPE_SKY_PROCESS_TASKS]               = {.name = "sky_process_tasks", .shared_memory_size = 0, .arg_size = 0},
  [CUDA_KERNEL_TYPE_SKY_PROCESS_TASKS_DEBUG]         = {.name = "sky_process_tasks_debug", .shared_memory_size = 0, .arg_size = 0},
  [CUDA_KERNEL_TYPE_SKY_PROCESS_INSCATTERING_EVENTS] = {.name = "sky_process_inscattering_events", .shared_memory_size = 0, .arg_size = 0},
  [CUDA_KERNEL_TYPE_SKY_COMPUTE_HDRI]                = {.name = "sky_compute_hdri", .shared_memory_size = 0, .arg_size = 0},
  [CUDA_KERNEL_TYPE_SKY_COMPUTE_TRANSMITTANCE_LUT]   = {.name = "sky_compute_transmittance_lut", .shared_memory_size = 0, .arg_size = 0},
  [CUDA_KERNEL_TYPE_SKY_COMPUTE_MULTISCATTERING_LUT] = {.name = "sky_compute_multiscattering_lut", .shared_memory_size = 0, .arg_size = 0},
  [CUDA_KERNEL_TYPE_CLOUD_COMPUTE_SHAPE_NOISE] =
    {.name = "cloud_compute_shape_noise", .shared_memory_size = 0, .arg_size = sizeof(KernelArgsCloudComputeShapeNoise)},
  [CUDA_KERNEL_TYPE_CLOUD_COMPUTE_DETAIL_NOISE] =
    {.name = "cloud_compute_detail_noise", .shared_memory_size = 0, .arg_size = sizeof(KernelArgsCloudComputeDetailNoise)},
  [CUDA_KERNEL_TYPE_CLOUD_COMPUTE_WEATHER_NOISE] =
    {.name = "cloud_compute_weather_noise", .shared_memory_size = 0, .arg_size = sizeof(KernelArgsCloudComputeWeatherNoise)},
  [CUDA_KERNEL_TYPE_VOLUME_PROCESS_EVENTS]        = {.name = "volume_process_events", .shared_memory_size = 0, .arg_size = 0},
  [CUDA_KERNEL_TYPE_PARTICLE_PROCESS_TASKS_DEBUG] = {.name = "particle_process_tasks_debug", .shared_memory_size = 0, .arg_size = 0},
  [CUDA_KERNEL_TYPE_PARTICLE_GENERATE]            = {.name = "particle_generate", .shared_memory_size = 0, .arg_size = 0},
  [CUDA_KERNEL_TYPE_LIGHT_COMPUTE_POWER]          = {.name = "light_compute_power", .shared_memory_size = 0, .arg_size = 0},
  [CUDA_KERNEL_TYPE_TEMPORAL_ACCUMULATION]        = {.name = "temporal_accumulation", .shared_memory_size = 0, .arg_size = 0},
  [CUDA_KERNEL_TYPE_TEMPORAL_ACCUMULATION_AOV]    = {.name = "temporal_accumulation_aov", .shared_memory_size = 0, .arg_size = 0},
  [CUDA_KERNEL_TYPE_GENERATE_FINAL_IMAGE]         = {.name = "generate_final_image", .shared_memory_size = 0, .arg_size = 0},
  [CUDA_KERNEL_TYPE_CONVERT_RGBF_TO_XRGB8]        = {.name = "convert_RGBF_to_XRGB8", .shared_memory_size = 0, .arg_size = 0},
  [CUDA_KERNEL_TYPE_CAMERA_POST_IMAGE_DOWNSAMPLE] = {.name = "camera_post_image_downsample", .shared_memory_size = 0, .arg_size = 0},
  [CUDA_KERNEL_TYPE_CAMERA_POST_IMAGE_DOWNSAMPLE_THRESHOLD] =
    {.name = "camera_post_image_downsample_threshold", .shared_memory_size = 0, .arg_size = 0},
  [CUDA_KERNEL_TYPE_CAMERA_POST_IMAGE_UPSAMPLE]    = {.name = "camera_post_image_upsample", .shared_memory_size = 0, .arg_size = 0},
  [CUDA_KERNEL_TYPE_CAMERA_POST_LENS_FLARE_GHOSTS] = {.name = "camera_post_lens_flare_ghosts", .shared_memory_size = 0, .arg_size = 0},
  [CUDA_KERNEL_TYPE_CAMERA_POST_LENS_FLARE_HALO]   = {.name = "camera_post_lens_flare_halo", .shared_memory_size = 0, .arg_size = 0}};
LUM_STATIC_SIZE_ASSERT(cuda_kernel_configs, sizeof(CUDAKernelConfig) * CUDA_KERNEL_TYPE_COUNT);

LuminaryResult kernel_create(CUDAKernel** kernel, Device* device, CUlibrary library, CUDAKernelType type) {
  __CHECK_NULL_ARGUMENT(kernel);
  __CHECK_NULL_ARGUMENT(device);
  __CHECK_NULL_ARGUMENT(library);

  __FAILURE_HANDLE(host_malloc(kernel, sizeof(CUDAKernel)));

  const CUDAKernelConfig* config = cuda_kernel_configs + type;

  CUDA_FAILURE_HANDLE(cuLibraryGetKernel(&(*kernel)->cuda_kernel, library, config->name));

  (*kernel)->shared_memory_size = config->shared_memory_size;
  (*kernel)->arg_size           = config->arg_size;

  return LUMINARY_SUCCESS;
}

LuminaryResult kernel_execute(CUDAKernel* kernel, CUstream stream) {
  __CHECK_NULL_ARGUMENT(kernel);

  CUlaunchConfig launch_config;

  launch_config.blockDimX      = THREADS_PER_BLOCK;
  launch_config.blockDimY      = 1;
  launch_config.blockDimZ      = 1;
  launch_config.gridDimX       = BLOCKS_PER_GRID;
  launch_config.gridDimY       = 1;
  launch_config.gridDimZ       = 1;
  launch_config.sharedMemBytes = kernel->shared_memory_size;
  launch_config.hStream        = stream;
  launch_config.attrs          = (CUlaunchAttribute*) 0;
  launch_config.numAttrs       = 0;

  CUDA_FAILURE_HANDLE(cuLaunchKernelEx(&launch_config, (CUfunction) kernel->cuda_kernel, (void**) 0, (void**) 0));

  return LUMINARY_SUCCESS;
}

LuminaryResult kernel_execute_with_args(CUDAKernel* kernel, void* arg_struct, CUstream stream) {
  __CHECK_NULL_ARGUMENT(kernel);

  CUlaunchConfig launch_config;

  launch_config.blockDimX      = THREADS_PER_BLOCK;
  launch_config.blockDimY      = 1;
  launch_config.blockDimZ      = 1;
  launch_config.gridDimX       = BLOCKS_PER_GRID;
  launch_config.gridDimY       = 1;
  launch_config.gridDimZ       = 1;
  launch_config.sharedMemBytes = kernel->shared_memory_size;
  launch_config.hStream        = stream;
  launch_config.attrs          = (CUlaunchAttribute*) 0;
  launch_config.numAttrs       = 0;

  void* kernel_args[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, arg_struct, CU_LAUNCH_PARAM_BUFFER_SIZE, &kernel->arg_size, CU_LAUNCH_PARAM_END};

  CUDA_FAILURE_HANDLE(cuLaunchKernelEx(&launch_config, (CUfunction) kernel->cuda_kernel, (void**) 0, kernel_args));

  return LUMINARY_SUCCESS;
}

LuminaryResult kernel_execute_custom(
  CUDAKernel* kernel, uint32_t threads_per_block, uint32_t blocks_per_grid, void* arg_struct, size_t arg_struct_size, CUstream stream) {
  __CHECK_NULL_ARGUMENT(kernel);

  CUlaunchConfig launch_config;

  launch_config.blockDimX      = threads_per_block;
  launch_config.blockDimY      = 1;
  launch_config.blockDimZ      = 1;
  launch_config.gridDimX       = blocks_per_grid;
  launch_config.gridDimY       = 1;
  launch_config.gridDimZ       = 1;
  launch_config.sharedMemBytes = kernel->shared_memory_size;
  launch_config.hStream        = stream;
  launch_config.attrs          = (CUlaunchAttribute*) 0;
  launch_config.numAttrs       = 0;

  void* kernel_args[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, arg_struct, CU_LAUNCH_PARAM_BUFFER_SIZE, &arg_struct_size, CU_LAUNCH_PARAM_END};

  CUDA_FAILURE_HANDLE(cuLaunchKernelEx(&launch_config, (CUfunction) kernel->cuda_kernel, (void**) 0, kernel_args));

  return LUMINARY_SUCCESS;
}

LuminaryResult kernel_destroy(CUDAKernel** kernel) {
  __CHECK_NULL_ARGUMENT(kernel);
  __CHECK_NULL_ARGUMENT(*kernel);

  __FAILURE_HANDLE(host_free(kernel));

  return LUMINARY_SUCCESS;
}
