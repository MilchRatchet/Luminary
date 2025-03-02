#ifndef LUMINARY_KERNEL_H
#define LUMINARY_KERNEL_H

#include "device_utils.h"

struct Device typedef Device;

enum CUDAKernelType {
  CUDA_KERNEL_TYPE_GENERATE_TRACE_TASKS,
  CUDA_KERNEL_TYPE_BALANCE_TRACE_TASKS,
  CUDA_KERNEL_TYPE_POSTPROCESS_TRACE_TASKS,
  CUDA_KERNEL_TYPE_GEOMETRY_PROCESS_TASKS_DEBUG,
  CUDA_KERNEL_TYPE_BSDF_GENERATE_SS_LUT,
  CUDA_KERNEL_TYPE_BSDF_GENERATE_GLOSSY_LUT,
  CUDA_KERNEL_TYPE_BSDF_GENERATE_DIELECTRIC_LUT,
  CUDA_KERNEL_TYPE_OCEAN_PROCESS_TASKS,
  CUDA_KERNEL_TYPE_OCEAN_PROCESS_TASKS_DEBUG,
  CUDA_KERNEL_TYPE_SKY_PROCESS_TASKS,
  CUDA_KERNEL_TYPE_SKY_PROCESS_TASKS_DEBUG,
  CUDA_KERNEL_TYPE_SKY_PROCESS_INSCATTERING_EVENTS,
  CUDA_KERNEL_TYPE_SKY_COMPUTE_HDRI,
  CUDA_KERNEL_TYPE_SKY_COMPUTE_TRANSMITTANCE_LUT,
  CUDA_KERNEL_TYPE_SKY_COMPUTE_MULTISCATTERING_LUT,
  CUDA_KERNEL_TYPE_CLOUD_COMPUTE_SHAPE_NOISE,
  CUDA_KERNEL_TYPE_CLOUD_COMPUTE_DETAIL_NOISE,
  CUDA_KERNEL_TYPE_CLOUD_COMPUTE_WEATHER_NOISE,
  CUDA_KERNEL_TYPE_CLOUD_PROCESS_TASKS,
  CUDA_KERNEL_TYPE_VOLUME_PROCESS_EVENTS,
  CUDA_KERNEL_TYPE_PARTICLE_PROCESS_TASKS_DEBUG,
  CUDA_KERNEL_TYPE_PARTICLE_GENERATE,
  CUDA_KERNEL_TYPE_LIGHT_COMPUTE_POWER,
  CUDA_KERNEL_TYPE_TEMPORAL_ACCUMULATION_FIRST_SAMPLE,
  CUDA_KERNEL_TYPE_TEMPORAL_ACCUMULATION_UPDATE,
  CUDA_KERNEL_TYPE_TEMPORAL_ACCUMULATION_OUTPUT,
  CUDA_KERNEL_TYPE_TEMPORAL_ACCUMULATION_OUTPUT_RAW,
  CUDA_KERNEL_TYPE_TEMPORAL_ACCUMULATION_AOV,
  CUDA_KERNEL_TYPE_GENERATE_FINAL_IMAGE,
  CUDA_KERNEL_TYPE_CONVERT_RGBF_TO_ARGB8,
  CUDA_KERNEL_TYPE_CAMERA_POST_IMAGE_DOWNSAMPLE,
  CUDA_KERNEL_TYPE_CAMERA_POST_IMAGE_DOWNSAMPLE_THRESHOLD,
  CUDA_KERNEL_TYPE_CAMERA_POST_IMAGE_UPSAMPLE,
  CUDA_KERNEL_TYPE_CAMERA_POST_LENS_FLARE_GHOSTS,
  CUDA_KERNEL_TYPE_CAMERA_POST_LENS_FLARE_HALO,
  CUDA_KERNEL_TYPE_OMM_LEVEL_0_FORMAT_4,
  CUDA_KERNEL_TYPE_OMM_REFINE_FORMAT_4,
  CUDA_KERNEL_TYPE_OMM_GATHER_ARRAY_FORMAT_4,

  CUDA_KERNEL_TYPE_COUNT
} typedef CUDAKernelType;

struct CUDAKernel {
  CUkernel cuda_kernel;
  uint32_t shared_memory_size;
  size_t param_size;
} typedef CUDAKernel;

DEVICE_CTX_FUNC LuminaryResult kernel_create(CUDAKernel** kernel, Device* device, CUlibrary library, CUDAKernelType type);
DEVICE_CTX_FUNC LuminaryResult kernel_execute(CUDAKernel* kernel, CUstream stream);
DEVICE_CTX_FUNC LuminaryResult kernel_execute_with_args(CUDAKernel* kernel, void* arg_struct, CUstream stream);
DEVICE_CTX_FUNC LuminaryResult kernel_execute_custom(
  CUDAKernel* kernel, uint32_t block_dim_x, uint32_t block_dim_y, uint32_t block_dim_z, uint32_t grid_dim_x, uint32_t grid_dim_y,
  uint32_t grid_dim_z, void* arg_struct, CUstream stream);
DEVICE_CTX_FUNC LuminaryResult kernel_destroy(CUDAKernel** kernel);

#endif /* LUMINARY_KERNEL_H */
