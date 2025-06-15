#include "kernel.h"

#include "device.h"
#include "internal_error.h"
#include "kernel_args.h"

struct CUDAKernelConfig {
  const char* name;
  size_t param_size;
} typedef CUDAKernelConfig;

static const CUDAKernelConfig cuda_kernel_configs[CUDA_KERNEL_TYPE_COUNT] = {
  [CUDA_KERNEL_TYPE_TASKS_CREATE]                 = {.name = "tasks_create", .param_size = 0},
  [CUDA_KERNEL_TYPE_TASKS_SORT]                   = {.name = "tasks_sort", .param_size = 0},
  [CUDA_KERNEL_TYPE_GEOMETRY_PROCESS_TASKS]       = {.name = "geometry_process_tasks", .param_size = 0},
  [CUDA_KERNEL_TYPE_GEOMETRY_PROCESS_TASKS_DEBUG] = {.name = "geometry_process_tasks_debug", .param_size = 0},
  [CUDA_KERNEL_TYPE_BSDF_GENERATE_SS_LUT]         = {.name = "bsdf_generate_ss_lut", .param_size = sizeof(KernelArgsBSDFGenerateSSLUT)},
  [CUDA_KERNEL_TYPE_BSDF_GENERATE_GLOSSY_LUT] = {.name = "bsdf_generate_glossy_lut", .param_size = sizeof(KernelArgsBSDFGenerateGlossyLUT)},
  [CUDA_KERNEL_TYPE_BSDF_GENERATE_DIELECTRIC_LUT] =
    {.name = "bsdf_generate_dielectric_lut", .param_size = sizeof(KernelArgsBSDFGenerateDielectricLUT)},
  [CUDA_KERNEL_TYPE_OCEAN_PROCESS_TASKS]             = {.name = "ocean_process_tasks", .param_size = 0},
  [CUDA_KERNEL_TYPE_OCEAN_PROCESS_TASKS_DEBUG]       = {.name = "ocean_process_tasks_debug", .param_size = 0},
  [CUDA_KERNEL_TYPE_SKY_PROCESS_TASKS]               = {.name = "sky_process_tasks", .param_size = 0},
  [CUDA_KERNEL_TYPE_SKY_PROCESS_TASKS_DEBUG]         = {.name = "sky_process_tasks_debug", .param_size = 0},
  [CUDA_KERNEL_TYPE_SKY_PROCESS_INSCATTERING_EVENTS] = {.name = "sky_process_inscattering_events", .param_size = 0},
  [CUDA_KERNEL_TYPE_SKY_COMPUTE_HDRI]                = {.name = "sky_compute_hdri", .param_size = sizeof(KernelArgsSkyComputeHDRI)},
  [CUDA_KERNEL_TYPE_SKY_COMPUTE_TRANSMITTANCE_LUT] =
    {.name = "sky_compute_transmittance_lut", .param_size = sizeof(KernelArgsSkyComputeTransmittanceLUT)},
  [CUDA_KERNEL_TYPE_SKY_COMPUTE_MULTISCATTERING_LUT] =
    {.name = "sky_compute_multiscattering_lut", .param_size = sizeof(KernelArgsSkyComputeMultiscatteringLUT)},
  [CUDA_KERNEL_TYPE_CLOUD_COMPUTE_SHAPE_NOISE] =
    {.name = "cloud_compute_shape_noise", .param_size = sizeof(KernelArgsCloudComputeShapeNoise)},
  [CUDA_KERNEL_TYPE_CLOUD_COMPUTE_DETAIL_NOISE] =
    {.name = "cloud_compute_detail_noise", .param_size = sizeof(KernelArgsCloudComputeDetailNoise)},
  [CUDA_KERNEL_TYPE_CLOUD_COMPUTE_WEATHER_NOISE] =
    {.name = "cloud_compute_weather_noise", .param_size = sizeof(KernelArgsCloudComputeWeatherNoise)},
  [CUDA_KERNEL_TYPE_CLOUD_PROCESS_TASKS]                = {.name = "cloud_process_tasks", .param_size = 0},
  [CUDA_KERNEL_TYPE_VOLUME_PROCESS_EVENTS]              = {.name = "volume_process_events", .param_size = 0},
  [CUDA_KERNEL_TYPE_VOLUME_PROCESS_TASKS]               = {.name = "volume_process_tasks", .param_size = 0},
  [CUDA_KERNEL_TYPE_PARTICLE_PROCESS_TASKS]             = {.name = "particle_process_tasks", .param_size = 0},
  [CUDA_KERNEL_TYPE_PARTICLE_PROCESS_TASKS_DEBUG]       = {.name = "particle_process_tasks_debug", .param_size = 0},
  [CUDA_KERNEL_TYPE_PARTICLE_GENERATE]                  = {.name = "particle_generate", .param_size = 0},
  [CUDA_KERNEL_TYPE_LIGHT_COMPUTE_INTENSITY]            = {.name = "light_compute_intensity", .param_size = 0},
  [CUDA_KERNEL_TYPE_TEMPORAL_ACCUMULATION_FIRST_SAMPLE] = {.name = "temporal_accumulation_first_sample", .param_size = 0},
  [CUDA_KERNEL_TYPE_TEMPORAL_ACCUMULATION_UPDATE]       = {.name = "temporal_accumulation_update", .param_size = 0},
  [CUDA_KERNEL_TYPE_TEMPORAL_ACCUMULATION_OUTPUT]       = {.name = "temporal_accumulation_output", .param_size = 0},
  [CUDA_KERNEL_TYPE_TEMPORAL_ACCUMULATION_OUTPUT_RAW]   = {.name = "temporal_accumulation_output_raw", .param_size = 0},
  [CUDA_KERNEL_TYPE_TEMPORAL_ACCUMULATION_AOV]          = {.name = "temporal_accumulation_aov", .param_size = 0},
  [CUDA_KERNEL_TYPE_GENERATE_FINAL_IMAGE]  = {.name = "generate_final_image", .param_size = sizeof(KernelArgsGenerateFinalImage)},
  [CUDA_KERNEL_TYPE_CONVERT_RGBF_TO_ARGB8] = {.name = "convert_RGBF_to_ARGB8", .param_size = sizeof(KernelArgsConvertRGBFToARGB8)},
  [CUDA_KERNEL_TYPE_CAMERA_POST_IMAGE_DOWNSAMPLE] =
    {.name = "camera_post_image_downsample", .param_size = sizeof(KernelArgsCameraPostImageDownsample)},
  [CUDA_KERNEL_TYPE_CAMERA_POST_IMAGE_DOWNSAMPLE_THRESHOLD] =
    {.name = "camera_post_image_downsample_threshold", .param_size = sizeof(KernelArgsCameraPostImageDownsampleThreshold)},
  [CUDA_KERNEL_TYPE_CAMERA_POST_IMAGE_UPSAMPLE] =
    {.name = "camera_post_image_upsample", .param_size = sizeof(KernelArgsCameraPostImageUpsample)},
  [CUDA_KERNEL_TYPE_CAMERA_POST_LENS_FLARE_GHOSTS] =
    {.name = "camera_post_lens_flare_ghosts", .param_size = sizeof(KernelArgsCameraPostLensFlareGhosts)},
  [CUDA_KERNEL_TYPE_CAMERA_POST_LENS_FLARE_HALO] =
    {.name = "camera_post_lens_flare_halo", .param_size = sizeof(KernelArgsCameraPostLensFlareHalo)},
  [CUDA_KERNEL_TYPE_OMM_LEVEL_0_FORMAT_4] = {.name = "omm_level_0_format_4", .param_size = sizeof(KernelArgsOMMLevel0Format4)},
  [CUDA_KERNEL_TYPE_OMM_REFINE_FORMAT_4]  = {.name = "omm_refine_format_4", .param_size = sizeof(KernelArgsOMMRefineFormat4)},
  [CUDA_KERNEL_TYPE_OMM_GATHER_ARRAY_FORMAT_4] =
    {.name = "omm_gather_array_format_4", .param_size = sizeof(KernelArgsOMMGatherArrayFormat4)},
  [CUDA_KERNEL_TYPE_BUFFER_ADD] = {.name = "buffer_add", .param_size = sizeof(KernelArgsBufferAdd)}};
LUM_STATIC_SIZE_ASSERT(cuda_kernel_configs, sizeof(CUDAKernelConfig) * CUDA_KERNEL_TYPE_COUNT);

LuminaryResult kernel_create(CUDAKernel** kernel, Device* device, CUlibrary library, CUDAKernelType type) {
  __CHECK_NULL_ARGUMENT(kernel);
  __CHECK_NULL_ARGUMENT(device);
  __CHECK_NULL_ARGUMENT(library);

  __FAILURE_HANDLE(host_malloc(kernel, sizeof(CUDAKernel)));
  memset(*kernel, 0, sizeof(CUDAKernel));

  (*kernel)->type = type;

  const CUDAKernelConfig* config = cuda_kernel_configs + type;

  CUresult result = cuLibraryGetKernel(&(*kernel)->cuda_kernel, library, config->name);

  if (result != CUDA_SUCCESS) {
    __RETURN_ERROR(LUMINARY_ERROR_CUDA, "Kernel %s failed to load.", config->name);
  }

  (*kernel)->param_size = config->param_size;

  uint32_t shared_memory_size;
  CUDA_FAILURE_HANDLE(
    cuKernelGetAttribute((int*) &shared_memory_size, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, (*kernel)->cuda_kernel, device->cuda_device));

  (*kernel)->shared_memory_size  = shared_memory_size;
  (*kernel)->default_block_count = device->properties.optimal_block_count;

  return LUMINARY_SUCCESS;
}

LuminaryResult kernel_execute(CUDAKernel* kernel, CUstream stream) {
  __CHECK_NULL_ARGUMENT(kernel);

  CUlaunchConfig launch_config;

  launch_config.blockDimX      = THREADS_PER_BLOCK;
  launch_config.blockDimY      = 1;
  launch_config.blockDimZ      = 1;
  launch_config.gridDimX       = kernel->default_block_count;
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
  launch_config.gridDimX       = kernel->default_block_count;
  launch_config.gridDimY       = 1;
  launch_config.gridDimZ       = 1;
  launch_config.sharedMemBytes = kernel->shared_memory_size;
  launch_config.hStream        = stream;
  launch_config.attrs          = (CUlaunchAttribute*) 0;
  launch_config.numAttrs       = 0;

  CUDA_FAILURE_HANDLE(cuLaunchKernelEx(&launch_config, (CUfunction) kernel->cuda_kernel, &arg_struct, (void**) 0));

  return LUMINARY_SUCCESS;
}

LuminaryResult kernel_execute_custom(
  CUDAKernel* kernel, uint32_t block_dim_x, uint32_t block_dim_y, uint32_t block_dim_z, uint32_t grid_dim_x, uint32_t grid_dim_y,
  uint32_t grid_dim_z, void* arg_struct, CUstream stream) {
  __CHECK_NULL_ARGUMENT(kernel);

  CUlaunchConfig launch_config;

  launch_config.blockDimX      = block_dim_x;
  launch_config.blockDimY      = block_dim_y;
  launch_config.blockDimZ      = block_dim_z;
  launch_config.gridDimX       = grid_dim_x;
  launch_config.gridDimY       = grid_dim_y;
  launch_config.gridDimZ       = grid_dim_z;
  launch_config.sharedMemBytes = kernel->shared_memory_size;
  launch_config.hStream        = stream;
  launch_config.attrs          = (CUlaunchAttribute*) 0;
  launch_config.numAttrs       = 0;

  CUDA_FAILURE_HANDLE(cuLaunchKernelEx(&launch_config, (CUfunction) kernel->cuda_kernel, &arg_struct, (void**) 0));

  return LUMINARY_SUCCESS;
}

LuminaryResult kernel_get_name(CUDAKernel* kernel, const char** name) {
  __CHECK_NULL_ARGUMENT(kernel);
  __CHECK_NULL_ARGUMENT(name);

  const CUDAKernelConfig* config = cuda_kernel_configs + kernel->type;

  *name = config->name;

  return LUMINARY_SUCCESS;
}

LuminaryResult kernel_destroy(CUDAKernel** kernel) {
  __CHECK_NULL_ARGUMENT(kernel);
  __CHECK_NULL_ARGUMENT(*kernel);

  __FAILURE_HANDLE(host_free(kernel));

  return LUMINARY_SUCCESS;
}
