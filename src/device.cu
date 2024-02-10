#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>

#include "buffer.h"
#include "config.h"
#include "cuda/brdf_unittest.cuh"
#include "cuda/bvh.cuh"
#include "cuda/camera_post.cuh"
#include "cuda/cloud_noise.cuh"
#include "cuda/directives.cuh"
#include "cuda/kernels.cuh"
#include "cuda/math.cuh"
#include "cuda/micromap.cuh"
#include "cuda/mipmap.cuh"
#include "cuda/ocean.cuh"
#include "cuda/particle.cuh"
#include "cuda/random.cuh"
#include "cuda/random_unittest.cuh"
#include "cuda/restir.cuh"
#include "cuda/sky.cuh"
#include "cuda/sky_hdri.cuh"
#include "cuda/toy.cuh"
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
  print_info("Copyright (c) 2023 MilchRatchet");
}

void device_handle_accumulation(RaytraceInstance* instance) {
  switch (instance->accum_mode) {
    case NO_ACCUMULATION:
      device_buffer_copy(instance->frame_buffer, instance->frame_accumulate);
      break;
    case TEMPORAL_ACCUMULATION:
      temporal_accumulation<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
      break;
    case TEMPORAL_REPROJECTION:
      device_buffer_copy(instance->frame_accumulate, instance->frame_temporal);
      temporal_reprojection<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
      break;
    default:
      error_message("Invalid accumulation mode %d specified", instance->accum_mode);
      break;
  }

  if (instance->aov_mode) {
    switch (instance->accum_mode) {
      case NO_ACCUMULATION:
      case TEMPORAL_REPROJECTION:
        device_buffer_copy(instance->frame_direct_buffer, instance->frame_direct_accumulate);
        device_buffer_copy(instance->frame_indirect_buffer, instance->frame_indirect_accumulate);
        break;
      case TEMPORAL_ACCUMULATION:
        temporal_accumulation_aov<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(
          (const RGBF*) device_buffer_get_pointer(instance->frame_direct_buffer),
          (RGBF*) device_buffer_get_pointer(instance->frame_direct_accumulate));
        temporal_accumulation_aov<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(
          (const RGBF*) device_buffer_get_pointer(instance->frame_indirect_buffer),
          (RGBF*) device_buffer_get_pointer(instance->frame_indirect_accumulate));
        break;
      default:
        error_message("Invalid accumulation mode %d specified", instance->accum_mode);
        break;
    }
  }
}

static void bind_type(RaytraceInstance* instance, int type, int depth) {
  device_update_symbol(iteration_type, type);
  device_update_symbol(depth, depth);

  switch (type) {
    case TYPE_CAMERA:
    case TYPE_BOUNCE:
      device_update_symbol(trace_tasks, instance->bounce_trace->device_pointer);
      device_update_symbol(trace_count, instance->bounce_trace_count->device_pointer);
      device_update_symbol(records, instance->bounce_records->device_pointer);
      break;
    case TYPE_LIGHT:
      device_update_symbol(trace_tasks, instance->light_trace->device_pointer);
      device_update_symbol(trace_count, instance->light_trace_count->device_pointer);
      device_update_symbol(records, instance->light_records->device_pointer);
      break;
  }
}

extern "C" void device_execute_main_kernels(RaytraceInstance* instance, int type, int depth) {
  bind_type(instance, type, depth);

  if (type != TYPE_CAMERA)
    balance_trace_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();

  preprocess_trace_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();

  if (instance->scene.particles.active && type != TYPE_LIGHT) {
    optixrt_execute(instance->particles_instance.kernel);
  }

  if (instance->bvh_type == BVH_LUMINARY) {
    process_trace_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
  }
  else {
    optixrt_execute(instance->optix_kernel);
  }

  if (instance->scene.fog.active || instance->scene.ocean.active) {
    volume_process_events<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
  }

  if (instance->scene.sky.cloud.active && !instance->scene.sky.hdri_active) {
    clouds_render_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
  }

  if (type != TYPE_LIGHT && instance->scene.sky.aerial_perspective) {
    process_sky_inscattering_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
  }

  postprocess_trace_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();

  if (type != TYPE_LIGHT) {
    restir_candidates_pool_generation<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
  }

  process_geometry_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();

  if (instance->scene.particles.active && type != TYPE_LIGHT) {
    particle_process_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
  }

  if (instance->scene.ocean.active) {
    process_ocean_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
  }

  process_sky_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();

  if (instance->scene.toy.active) {
    process_toy_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
  }

  if (type != TYPE_LIGHT && (instance->scene.fog.active || instance->scene.ocean.active)) {
    volume_process_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
  }
}

extern "C" void device_execute_debug_kernels(RaytraceInstance* instance, int type) {
  bind_type(instance, type, 0);

  preprocess_trace_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();

  if (instance->scene.particles.active && type != TYPE_LIGHT) {
    optixrt_execute(instance->particles_instance.kernel);
  }

  if (instance->bvh_type == BVH_LUMINARY) {
    process_trace_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
  }
  else {
    optixrt_execute(instance->optix_kernel);
  }

  if (instance->scene.fog.active || instance->scene.ocean.active) {
    volume_process_events<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
  }

  postprocess_trace_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();

  process_debug_geometry_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
  if (instance->scene.particles.active && type != TYPE_LIGHT) {
    particle_process_debug_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
  }
  if (instance->scene.ocean.active) {
    process_debug_ocean_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
  }
  process_debug_sky_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
  if (instance->scene.toy.active) {
    process_debug_toy_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
  }
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

extern "C" void device_gather_device_table(void* dst, enum cudaMemcpyKind kind) {
  gpuErrchk(cudaMemcpyFromSymbol(dst, device, sizeof(DeviceConstantMemory), 0, kind));
}

extern "C" void device_initialize_random_generators() {
  initialize_randoms<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
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
