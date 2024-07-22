#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>

#include "buffer.h"
#include "config.h"
#include "cuda/brdf_unittest.cuh"
#include "cuda/bsdf_lut.cuh"
#include "cuda/bvh.cuh"
#include "cuda/camera_post.cuh"
#include "cuda/cloud_noise.cuh"
#include "cuda/directives.cuh"
#include "cuda/geometry.cuh"
#include "cuda/kernels.cuh"
#include "cuda/math.cuh"
#include "cuda/micromap.cuh"
#include "cuda/mipmap.cuh"
#include "cuda/particle.cuh"
#include "cuda/random.cuh"
#include "cuda/random_unittest.cuh"
#include "cuda/ris.cuh"
#include "cuda/sky.cuh"
#include "cuda/sky_hdri.cuh"
#include "cuda/utils.cuh"
#include "cuda/volume.cuh"
#include "device.h"
#include "log.h"
#include "optixrt.h"
#include "structs.h"
#include "utils.h"

extern "C" void device_init() {
  gpuErrchk(cudaSetDeviceFlags(cudaDeviceMapHost));

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);

  device_set_memory_limit(prop.totalGlobalMem);

  print_info("Luminary - %s", prop.name);
  print_info("[%s] %s", LUMINARY_BRANCH_NAME, LUMINARY_VERSION_DATE);
  print_info("Compiled using %s on %s", LUMINARY_COMPILER, LUMINARY_OS);
  print_info("CUDA Version %s OptiX Version %s", LUMINARY_CUDA_VERSION, LUMINARY_OPTIX_VERSION);
  print_info("Copyright (c) 2024 MilchRatchet");
}

void device_handle_accumulation(RaytraceInstance* instance) {
  switch (instance->accum_mode) {
    case NO_ACCUMULATION:
    case TEMPORAL_ACCUMULATION:
      temporal_accumulation<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
      break;
    case TEMPORAL_REPROJECTION:
    default:
      error_message("Invalid accumulation mode %d specified", instance->accum_mode);
      break;
  }
}

extern "C" void device_execute_main_kernels(RaytraceInstance* instance, int depth) {
  device_update_symbol(depth, depth);

  if (depth > 0)
    balance_trace_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();

  if (instance->bvh_type == BVH_LUMINARY) {
    process_trace_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
  }
  else {
    optixrt_execute(instance->optix_kernel);
  }

  if (instance->scene.particles.active) {
    optixrt_execute(instance->particles_instance.kernel);
  }

  if (instance->scene.fog.active || instance->scene.ocean.active) {
    volume_process_events<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
  }

  if (instance->scene.sky.cloud.active && !instance->scene.sky.hdri_active) {
    clouds_render_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
  }

  if (instance->scene.sky.aerial_perspective) {
    process_sky_inscattering_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
  }

  postprocess_trace_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();

  ris_presample_lights<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();

  optixrt_execute(instance->optix_kernel_geometry);

  if (instance->scene.fog.active || instance->scene.ocean.active || instance->scene.particles.active) {
    optixrt_execute(instance->optix_kernel_volume);
  }

  process_sky_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
}

extern "C" void device_execute_debug_kernels(RaytraceInstance* instance) {
  const int depth = 0;
  device_update_symbol(depth, depth);

  if (instance->bvh_type == BVH_LUMINARY) {
    process_trace_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
  }
  else {
    optixrt_execute(instance->optix_kernel);
  }

  if (instance->scene.particles.active) {
    optixrt_execute(instance->particles_instance.kernel);
  }

  if (instance->scene.fog.active || instance->scene.ocean.active) {
    volume_process_events<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
  }

  postprocess_trace_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();

  process_debug_geometry_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
  if (instance->scene.particles.active) {
    particle_process_debug_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
  }
  process_debug_sky_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
}

extern "C" void device_generate_tasks() {
  generate_trace_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
}

extern "C" void _device_update_symbol(const size_t offset, const void* src, const size_t size) {
  gpuErrchk(cudaMemcpyToSymbol(device, src, size, offset, cudaMemcpyHostToDevice));
}

extern "C" void _device_gather_symbol(void* dst, const size_t offset, size_t size) {
  gpuErrchk(cudaMemcpyFromSymbol(dst, device, size, offset, cudaMemcpyDeviceToHost));
}

extern "C" void device_gather_device_table(void* dst, cudaMemcpyKind kind) {
  gpuErrchk(cudaMemcpyFromSymbol(dst, device, sizeof(DeviceConstantMemory), 0, kind));
}

extern "C" unsigned int device_get_thread_count() {
  return THREADS_PER_BLOCK * BLOCKS_PER_GRID;
}

extern "C" void device_copy_framebuffer_to_8bit(
  RGBF* gpu_source, XRGB8* gpu_scratch, XRGB8* cpu_dest, const int width, const int height, const int ld,
  const OutputVariable output_variable) {
  convert_RGBF_to_XRGB8<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(gpu_source, gpu_scratch, width, height, ld, output_variable);
  gpuErrchk(cudaMemcpy(cpu_dest, gpu_scratch, sizeof(XRGB8) * ld * height, cudaMemcpyDeviceToHost));
}
