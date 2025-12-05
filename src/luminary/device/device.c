#include "device.h"

#include <optix_function_table_definition.h>

#include "camera.h"
#include "cloud.h"
#include "config.h"
#include "device_light.h"
#include "device_memory.h"
#include "device_texture.h"
#include "device_utils.h"
#include "fog.h"
#include "internal_error.h"
#include "particles.h"
#include "sky.h"

#ifdef CUDA_STALL_VALIDATION
ThreadStatus* __cuda_stall_validation_macro_walltime;
#endif

#define DEVICE_ASSERT_AVAILABLE                    \
  {                                                \
    if (device->state == DEVICE_STATE_UNAVAILABLE) \
      return LUMINARY_SUCCESS;                     \
  }

static const DeviceConstantMemoryMember device_scene_entity_to_const_memory_member[SCENE_ENTITY_GLOBAL_COUNT] = {
  DEVICE_CONSTANT_MEMORY_MEMBER_SETTINGS,   // SCENE_ENTITY_SETTINGS
  DEVICE_CONSTANT_MEMORY_MEMBER_CAMERA,     // SCENE_ENTITY_CAMERA
  DEVICE_CONSTANT_MEMORY_MEMBER_OCEAN,      // SCENE_ENTITY_OCEAN
  DEVICE_CONSTANT_MEMORY_MEMBER_SKY,        // SCENE_ENTITY_SKY
  DEVICE_CONSTANT_MEMORY_MEMBER_CLOUD,      // SCENE_ENTITY_CLOUD
  DEVICE_CONSTANT_MEMORY_MEMBER_FOG,        // SCENE_ENTITY_FOG
  DEVICE_CONSTANT_MEMORY_MEMBER_PARTICLES,  // SCENE_ENTITY_PARTICLES
};

static const size_t device_cuda_const_memory_offsets[DEVICE_CONSTANT_MEMORY_MEMBER_COUNT + 1] = {
  offsetof(DeviceConstantMemory, ptrs),                          // DEVICE_CONSTANT_MEMORY_MEMBER_PTRS
  offsetof(DeviceConstantMemory, settings),                      // DEVICE_CONSTANT_MEMORY_MEMBER_SETTINGS
  offsetof(DeviceConstantMemory, camera),                        // DEVICE_CONSTANT_MEMORY_MEMBER_CAMERA
  offsetof(DeviceConstantMemory, ocean),                         // DEVICE_CONSTANT_MEMORY_MEMBER_OCEAN
  offsetof(DeviceConstantMemory, sky),                           // DEVICE_CONSTANT_MEMORY_MEMBER_SKY
  offsetof(DeviceConstantMemory, cloud),                         // DEVICE_CONSTANT_MEMORY_MEMBER_CLOUD
  offsetof(DeviceConstantMemory, fog),                           // DEVICE_CONSTANT_MEMORY_MEMBER_FOG
  offsetof(DeviceConstantMemory, particles),                     // DEVICE_CONSTANT_MEMORY_MEMBER_PARTICLES
  offsetof(DeviceConstantMemory, optix_bvh),                     // DEVICE_CONSTANT_MEMORY_MEMBER_OPTIX_BVH
  offsetof(DeviceConstantMemory, moon_albedo_tex),               // DEVICE_CONSTANT_MEMORY_MEMBER_MOON_TEX
  offsetof(DeviceConstantMemory, sky_lut_transmission_low_tex),  // DEVICE_CONSTANT_MEMORY_MEMBER_SKY_LUT_TEX
  offsetof(DeviceConstantMemory, sky_hdri_color_tex),            // DEVICE_CONSTANT_MEMORY_MEMBER_SKY_HDRI_TEX
  offsetof(DeviceConstantMemory, bsdf_lut_conductor),            // DEVICE_CONSTANT_MEMORY_MEMBER_BSDF_LUT_TEX
  offsetof(DeviceConstantMemory, cloud_noise_shape_tex),         // DEVICE_CONSTANT_MEMORY_MEMBER_CLOUD_NOISE_TEX
  offsetof(DeviceConstantMemory, spectral_xy_lut_tex),           // DEVICE_CONSTANT_MEMORY_MEMBER_SPECTRAL_LUT_TEX
  offsetof(DeviceConstantMemory, config),                        // DEVICE_CONSTANT_MEMORY_MEMBER_CONFIG
  offsetof(DeviceConstantMemory, state),                         // DEVICE_CONSTANT_MEMORY_MEMBER_STATE
  sizeof(DeviceConstantMemory)                                   // DEVICE_CONSTANT_MEMORY_MEMBER_COUNT
};

static const size_t device_cuda_const_memory_sizes[DEVICE_CONSTANT_MEMORY_MEMBER_COUNT] = {
  sizeof(DevicePointers),                // DEVICE_CONSTANT_MEMORY_MEMBER_PTRS
  sizeof(DeviceRendererSettings),        // DEVICE_CONSTANT_MEMORY_MEMBER_SETTINGS
  sizeof(DeviceCamera),                  // DEVICE_CONSTANT_MEMORY_MEMBER_CAMERA
  sizeof(DeviceOcean),                   // DEVICE_CONSTANT_MEMORY_MEMBER_OCEAN
  sizeof(DeviceSky),                     // DEVICE_CONSTANT_MEMORY_MEMBER_SKY
  sizeof(DeviceCloud),                   // DEVICE_CONSTANT_MEMORY_MEMBER_CLOUD
  sizeof(DeviceFog),                     // DEVICE_CONSTANT_MEMORY_MEMBER_FOG
  sizeof(DeviceParticles),               // DEVICE_CONSTANT_MEMORY_MEMBER_PARTICLES
  sizeof(OptixTraversableHandle) * 4,    // DEVICE_CONSTANT_MEMORY_MEMBER_OPTIX_BVH
  sizeof(DeviceTextureObject) * 2,       // DEVICE_CONSTANT_MEMORY_MEMBER_MOON_TEX
  sizeof(DeviceTextureObject) * 4,       // DEVICE_CONSTANT_MEMORY_MEMBER_SKY_LUT_TEX
  sizeof(DeviceTextureObject) * 2,       // DEVICE_CONSTANT_MEMORY_MEMBER_SKY_HDRI_TEX
  sizeof(DeviceTextureObject) * 4,       // DEVICE_CONSTANT_MEMORY_MEMBER_BSDF_LUT_TEX
  sizeof(DeviceTextureObject) * 3,       // DEVICE_CONSTANT_MEMORY_MEMBER_CLOUD_NOISE_TEX
  sizeof(DeviceTextureObject) * 2,       // DEVICE_CONSTANT_MEMORY_MEMBER_SPECTRAL_LUT_TEX
  sizeof(DeviceExecutionConfiguration),  // DEVICE_CONSTANT_MEMORY_MEMBER_CONFIG
  sizeof(DeviceExecutionState)           // DEVICE_CONSTANT_MEMORY_MEMBER_STATE
};

#define DEVICE_UPDATE_CONSTANT_MEMORY(member, value)                                              \
  {                                                                                               \
    device->constant_memory->member      = (value);                                               \
    const size_t __macro_offset          = offsetof(DeviceConstantMemory, member);                \
    uint32_t __macro_const_memory_member = 0;                                                     \
    while (__macro_offset >= device_cuda_const_memory_offsets[__macro_const_memory_member + 1]) { \
      __macro_const_memory_member++;                                                              \
    }                                                                                             \
    __FAILURE_HANDLE(_device_set_constant_memory_dirty(device, __macro_const_memory_member));     \
  }

void _device_init(void) {
  CUresult cuda_result = cuInit(0);

  if (cuda_result != CUDA_SUCCESS) {
    crash_message("Failed to initialize CUDA.");
  }

  info_message("Initialized CUDA %s", LUMINARY_CUDA_VERSION);

  OptixResult optix_result = optixInit();

  if (optix_result != OPTIX_SUCCESS) {
    crash_message("Failed to initialize OptiX.");
  }

  info_message("Initialized OptiX %s", LUMINARY_OPTIX_VERSION);

  _device_memory_init();

#ifdef CUDA_STALL_VALIDATION
  thread_status_create(&__cuda_stall_validation_macro_walltime);
#endif
}

void _device_shutdown(void) {
#ifdef CUDA_STALL_VALIDATION
  thread_status_destroy(&__cuda_stall_validation_macro_walltime);
#endif

  _device_memory_shutdown();
}

#define OPTIX_CHECK_CALLBACK_ERROR(device)                                        \
  if (device->optix_callback_error) {                                             \
    __RETURN_ERROR(LUMINARY_ERROR_OPTIX, "OptiX callback logged a fatal error."); \
  }

////////////////////////////////////////////////////////////////////
// OptiX Log Callback
////////////////////////////////////////////////////////////////////

#ifdef OPTIX_VALIDATION
static void _device_optix_log_callback(unsigned int level, const char* tag, const char* message, void* cbdata) {
  Device* device = (Device*) cbdata;

  switch (level) {
    case 1:
      device->optix_callback_error = true;
      luminary_print_error("[OptiX Log Message][%s] %s", tag, message);
      break;
    case 2:
      luminary_print_error("[OptiX Log Message][%s] %s", tag, message);
      break;
    case 3:
      luminary_print_warn("[OptiX Log Message][%s] %s", tag, message);
      break;
    default:
      luminary_print_info("[OptiX Log Message][%s] %s", tag, message);
      break;
  }
}
#endif

////////////////////////////////////////////////////////////////////
// Utility functions
////////////////////////////////////////////////////////////////////

static LuminaryResult _device_get_properties(DeviceProperties* props, Device* device) {
  __CHECK_NULL_ARGUMENT(props);

  int major;
  CUDA_FAILURE_HANDLE(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device->cuda_device));

  int minor;
  CUDA_FAILURE_HANDLE(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device->cuda_device));

  int sm_count;
  CUDA_FAILURE_HANDLE(cuDeviceGetAttribute(&sm_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device->cuda_device));

  int l2_cache_size;
  CUDA_FAILURE_HANDLE(cuDeviceGetAttribute(&l2_cache_size, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, device->cuda_device));

  int max_block_count;
  CUDA_FAILURE_HANDLE(cuDeviceGetAttribute(&max_block_count, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, device->cuda_device));

  int max_blocks_per_sm;
  CUDA_FAILURE_HANDLE(cuDeviceGetAttribute(&max_blocks_per_sm, CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR, device->cuda_device));

  int max_threads_per_sm;
  CUDA_FAILURE_HANDLE(cuDeviceGetAttribute(&max_threads_per_sm, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, device->cuda_device));

  props->major              = (uint32_t) major;
  props->minor              = (uint32_t) minor;
  props->sm_count           = (uint32_t) sm_count;
  props->l2_cache_size      = (size_t) l2_cache_size;
  props->max_block_count    = (uint32_t) max_block_count;
  props->max_blocks_per_sm  = (uint32_t) max_blocks_per_sm;
  props->max_threads_per_sm = (uint32_t) max_threads_per_sm;

  const uint32_t max_actual_blocks_per_sm = min(props->max_blocks_per_sm, props->max_threads_per_sm / THREADS_PER_BLOCK);
  props->optimal_block_count              = max_actual_blocks_per_sm * props->sm_count;

  CUDA_FAILURE_HANDLE(cuDeviceGetName(props->name, 256, device->cuda_device));

  CUDA_FAILURE_HANDLE(cuDeviceTotalMem(&props->memory_size, device->cuda_device));

  int warp_size;
  CUDA_FAILURE_HANDLE(cuDeviceGetAttribute(&warp_size, CU_DEVICE_ATTRIBUTE_WARP_SIZE, device->cuda_device));

  if (warp_size != 32) {
    warn_message(
      "CUDA reported that %s has a warp size of %d. This is not supported and unexpected. Deactivating this GPU.", props->name, warp_size);
    device->state = DEVICE_STATE_UNAVAILABLE;
    return LUMINARY_SUCCESS;
  }

  props->arch            = DEVICE_ARCH_UNKNOWN;
  props->rt_core_version = 0;

  switch (major) {
    case 5: {
      if (minor == 0 || minor == 2 || minor == 3) {
        props->arch            = DEVICE_ARCH_MAXWELL;
        props->rt_core_version = 0;
      }
    } break;
    case 6: {
      if (minor == 0 || minor == 1 || minor == 2) {
        props->arch            = DEVICE_ARCH_PASCAL;
        props->rt_core_version = 0;
      }
    } break;
    case 7: {
      if (minor == 0 || minor == 2) {
        props->arch            = DEVICE_ARCH_VOLTA;
        props->rt_core_version = 0;
      }
      else if (minor == 5) {
        props->arch            = DEVICE_ARCH_TURING;
        props->rt_core_version = 1;

        // TU116 and TU117 do not have RT cores, these can be detected by searching for GTX in the name
        for (int i = 0; i < 250; i++) {
          if (props->name[i] == 'G' && props->name[i + 1] == 'T' && props->name[i + 2] == 'X') {
            props->rt_core_version = 0;
            break;
          }
        }
      }
    } break;
    case 8: {
      if (minor == 0) {
        // GA100 has no RT cores
        props->arch            = DEVICE_ARCH_AMPERE_HPC;
        props->rt_core_version = 0;
      }
      else if (minor == 6 || minor == 7) {
        props->arch            = DEVICE_ARCH_AMPERE_CONSUMER;
        props->rt_core_version = 2;
      }
      else if (minor == 9) {
        props->arch            = DEVICE_ARCH_ADA;
        props->rt_core_version = 3;
      }
    } break;
    case 9: {
      if (minor == 0) {
        props->arch            = DEVICE_ARCH_HOPPER;
        props->rt_core_version = 0;
      }
    } break;
    case 10: {
      if (minor == 0) {
        props->arch            = DEVICE_ARCH_BLACKWELL_HPC;
        props->rt_core_version = 0;
      }
    } break;
    case 12: {
      if (minor == 0) {
        props->arch            = DEVICE_ARCH_BLACKWELL_CONSUMER;
        props->rt_core_version = 4;
      }
    } break;
    default:
      break;
  }

  if (props->arch == DEVICE_ARCH_UNKNOWN) {
    warn_message("Luminary failed to identify architecture of CUDA compute capability %d.%d. Deactivated %s.", major, minor, props->name);
    device->state = DEVICE_STATE_UNAVAILABLE;
  }

  return LUMINARY_SUCCESS;
}

static LuminaryResult _device_get_optix_properties(Device* device) {
  __CHECK_NULL_ARGUMENT(device);

  OPTIX_FAILURE_HANDLE(optixDeviceContextGetProperty(
    device->optix_ctx, OPTIX_DEVICE_PROPERTY_LIMIT_MAX_TRACE_DEPTH, &device->optix_properties.max_trace_depth,
    sizeof(device->optix_properties.max_trace_depth)));
  OPTIX_FAILURE_HANDLE(optixDeviceContextGetProperty(
    device->optix_ctx, OPTIX_DEVICE_PROPERTY_LIMIT_MAX_TRAVERSABLE_GRAPH_DEPTH, &device->optix_properties.max_traversable_graph_depth,
    sizeof(device->optix_properties.max_traversable_graph_depth)));
  OPTIX_FAILURE_HANDLE(optixDeviceContextGetProperty(
    device->optix_ctx, OPTIX_DEVICE_PROPERTY_LIMIT_MAX_PRIMITIVES_PER_GAS, &device->optix_properties.max_primitives_per_gas,
    sizeof(device->optix_properties.max_primitives_per_gas)));
  OPTIX_FAILURE_HANDLE(optixDeviceContextGetProperty(
    device->optix_ctx, OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCES_PER_IAS, &device->optix_properties.max_instances_per_ias,
    sizeof(device->optix_properties.max_instances_per_ias)));
  OPTIX_FAILURE_HANDLE(optixDeviceContextGetProperty(
    device->optix_ctx, OPTIX_DEVICE_PROPERTY_RTCORE_VERSION, &device->optix_properties.rtcore_version,
    sizeof(device->optix_properties.rtcore_version)));
  OPTIX_FAILURE_HANDLE(optixDeviceContextGetProperty(
    device->optix_ctx, OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCE_ID, &device->optix_properties.max_instance_id,
    sizeof(device->optix_properties.max_instance_id)));
  OPTIX_FAILURE_HANDLE(optixDeviceContextGetProperty(
    device->optix_ctx, OPTIX_DEVICE_PROPERTY_LIMIT_NUM_BITS_INSTANCE_VISIBILITY_MASK,
    &device->optix_properties.num_bits_instance_visibility_mask, sizeof(device->optix_properties.num_bits_instance_visibility_mask)));
  OPTIX_FAILURE_HANDLE(optixDeviceContextGetProperty(
    device->optix_ctx, OPTIX_DEVICE_PROPERTY_LIMIT_MAX_SBT_RECORDS_PER_GAS, &device->optix_properties.max_sbt_records_per_gas,
    sizeof(device->optix_properties.max_sbt_records_per_gas)));
  OPTIX_FAILURE_HANDLE(optixDeviceContextGetProperty(
    device->optix_ctx, OPTIX_DEVICE_PROPERTY_LIMIT_MAX_SBT_OFFSET, &device->optix_properties.max_sbt_offset,
    sizeof(device->optix_properties.max_sbt_offset)));
  OPTIX_FAILURE_HANDLE(optixDeviceContextGetProperty(
    device->optix_ctx, OPTIX_DEVICE_PROPERTY_SHADER_EXECUTION_REORDERING, &device->optix_properties.shader_execution_reordering,
    sizeof(device->optix_properties.shader_execution_reordering)));

  return LUMINARY_SUCCESS;
}

static LuminaryResult _device_print_info(Device* device) {
  __CHECK_NULL_ARGUMENT(device);

  info_message("[Device %u] %s", device->index, device->properties.name);
  info_message(
    "           sm_%u%u | %u SMs | %.1f GB VRAM | %.1f MB L2", device->properties.major, device->properties.minor,
    device->properties.sm_count, device->properties.memory_size / (1024.0f * 1024.0f * 1024.0f),
    device->properties.l2_cache_size / (1024.0f * 1024.0f));

  return LUMINARY_SUCCESS;
}

static LuminaryResult _device_set_constant_memory_dirty(Device* device, DeviceConstantMemoryMember member) {
  __CHECK_NULL_ARGUMENT(device);

  if (device->constant_memory_dirty.is_dirty) {
    device->constant_memory_dirty.update_everything |= device->constant_memory_dirty.member != member;
  }
  else {
    device->constant_memory_dirty.is_dirty = true;
    device->constant_memory_dirty.member   = member;
  }

  return LUMINARY_SUCCESS;
}

static LuminaryResult _device_reset_constant_memory_dirty(Device* device) {
  __CHECK_NULL_ARGUMENT(device);

  device->constant_memory_dirty.is_dirty          = false;
  device->constant_memory_dirty.update_everything = false;

  return LUMINARY_SUCCESS;
}

static LuminaryResult _device_update_constant_memory(Device* device) {
  __CHECK_NULL_ARGUMENT(device);

  if (!device->constant_memory_dirty.is_dirty)
    return LUMINARY_SUCCESS;

  size_t offset;
  size_t size;
  if (device->constant_memory_dirty.update_everything) {
    offset = 0;
    size   = sizeof(DeviceConstantMemory) - sizeof(DeviceExecutionState);  // Exec state is updated separately by the renderer.
  }
  else {
    offset = device_cuda_const_memory_offsets[device->constant_memory_dirty.member];
    size   = device_cuda_const_memory_sizes[device->constant_memory_dirty.member];
  }

  CUDA_FAILURE_HANDLE(
    cuMemcpyHtoDAsync_v2(device->cuda_device_const_memory + offset, device->constant_memory + offset, size, device->stream_main));

  return LUMINARY_SUCCESS;
}

static LuminaryResult _device_setup_execution_config(Device* device) {
  __CHECK_NULL_ARGUMENT(device);

  DeviceExecutionConfiguration config;

  config.num_blocks           = device->properties.optimal_block_count;
  config.num_tasks_per_thread = RECOMMENDED_TASKS_PER_THREAD;

  DEVICE_UPDATE_CONSTANT_MEMORY(config, config);

  return LUMINARY_SUCCESS;
}

static LuminaryResult _device_update_get_next_undersampling_state(Device* device, uint32_t* new_undersampling_state) {
  __CHECK_NULL_ARGUMENT(device);

  uint32_t undersampling_state = device->undersampling_state;

  if (undersampling_state) {
    // Remove first sample flag
    undersampling_state &= ~UNDERSAMPLING_FIRST_SAMPLE_MASK;

    if (undersampling_state) {
      if ((undersampling_state & UNDERSAMPLING_ITERATION_MASK) == 0) {
        // Decrement stage, since iteration is 0 we can just subtract first and then apply the mask.
        uint32_t stage      = (undersampling_state - 1) & UNDERSAMPLING_STAGE_MASK;
        undersampling_state = (undersampling_state & ~UNDERSAMPLING_STAGE_MASK) | (stage & UNDERSAMPLING_STAGE_MASK);

        undersampling_state |= (stage > 0) ? 0b10 & UNDERSAMPLING_ITERATION_MASK : 0;
      }
      else {
        // Decrement iteration
        uint32_t iteration  = (undersampling_state & UNDERSAMPLING_ITERATION_MASK) - 1;
        undersampling_state = (undersampling_state & ~UNDERSAMPLING_ITERATION_MASK) | (iteration & UNDERSAMPLING_ITERATION_MASK);
      }
    }
  }

  *new_undersampling_state = undersampling_state;

  return LUMINARY_SUCCESS;
}

////////////////////////////////////////////////////////////////////
// Buffer handling
////////////////////////////////////////////////////////////////////

#define __DEVICE_BUFFER_FREE(buffer)                          \
  if (device->buffers.buffer) {                               \
    __FAILURE_HANDLE(device_free(&(device->buffers.buffer))); \
    DEVICE_UPDATE_CONSTANT_MEMORY(ptrs.buffer, (void*) 0);    \
  }

#define __DEVICE_BUFFER_ALLOCATE(buffer, size)                                                  \
  {                                                                                             \
    bool __macro_buffer_requires_allocation = true;                                             \
    if (device->buffers.buffer) {                                                               \
      size_t __macro_previous_size;                                                             \
      __FAILURE_HANDLE(device_memory_get_size(device->buffers.buffer, &__macro_previous_size)); \
      if (__macro_previous_size != size) {                                                      \
        __DEVICE_BUFFER_FREE(buffer);                                                           \
      }                                                                                         \
      else {                                                                                    \
        __macro_buffer_requires_allocation = false;                                             \
      }                                                                                         \
    }                                                                                           \
    if (__macro_buffer_requires_allocation) {                                                   \
      __FAILURE_HANDLE(device_malloc(&device->buffers.buffer, size));                           \
      DEVICE_UPDATE_CONSTANT_MEMORY(ptrs.buffer, DEVICE_PTR(device->buffers.buffer));           \
    }                                                                                           \
  }

#define __DEVICE_BUFFER_REALLOC(buffer, size)                                                                                             \
  {                                                                                                                                       \
    if (device->buffers.buffer) {                                                                                                         \
      size_t __macro_previous_size;                                                                                                       \
      __FAILURE_HANDLE(device_memory_get_size(device->buffers.buffer, &__macro_previous_size));                                           \
      if (size > __macro_previous_size) {                                                                                                 \
        DEVICE void* __macro_new_device_buffer;                                                                                           \
        __FAILURE_HANDLE(device_malloc(&__macro_new_device_buffer, size));                                                                \
        void* __macro_staging_buffer;                                                                                                     \
        __FAILURE_HANDLE(device_staging_manager_register_direct_access(                                                                   \
          device->staging_manager, __macro_new_device_buffer, 0, size, &__macro_staging_buffer));                                         \
        __FAILURE_HANDLE(device_download(__macro_staging_buffer, device->buffers.buffer, 0, __macro_previous_size, device->stream_main)); \
        CUDA_FAILURE_HANDLE(cuStreamSynchronize(device->stream_main));                                                                    \
        __DEVICE_BUFFER_FREE(buffer);                                                                                                     \
        device->buffers.buffer = __macro_new_device_buffer;                                                                               \
        DEVICE_UPDATE_CONSTANT_MEMORY(ptrs.buffer, DEVICE_PTR(device->buffers.buffer));                                                   \
      }                                                                                                                                   \
    }                                                                                                                                     \
    else {                                                                                                                                \
      __DEVICE_BUFFER_ALLOCATE(buffer, size);                                                                                             \
    }                                                                                                                                     \
  }

static LuminaryResult _device_allocate_work_buffers(Device* device) {
  __CHECK_NULL_ARGUMENT(device);

  const uint32_t internal_pixel_count     = device->constant_memory->settings.width * device->constant_memory->settings.height;
  const uint32_t external_pixel_count     = (internal_pixel_count >> (device->constant_memory->settings.supersampling * 2));
  const uint32_t gbuffer_meta_pixel_count = external_pixel_count >> 2;

  const uint32_t thread_count = THREADS_PER_BLOCK * device->properties.optimal_block_count;

  // Start by computing how well this pixel count fits to the recommended tasks per thread.
  uint32_t tasks_per_thread = RECOMMENDED_TASKS_PER_THREAD;
  uint32_t total_task_count;

  while (tasks_per_thread < MAXIMUM_TASKS_PER_THREAD) {
    total_task_count = thread_count * tasks_per_thread;

    const uint32_t tile_count       = (internal_pixel_count + total_task_count - 1) / total_task_count;
    const uint32_t stale_tail_tasks = tile_count * total_task_count - internal_pixel_count;

    // If the number of resident tasks in the last tile is above a threshold, then accept this tasks per thread.
    if (total_task_count - stale_tail_tasks > thread_count * MINIMUM_TASKS_PER_THREAD)
      break;

    tasks_per_thread++;
  }

  DeviceWorkBuffersAllocationProperties properties;
  properties.external_pixel_count = external_pixel_count;
  properties.internal_pixel_count = internal_pixel_count;
  properties.gbuffer_pixel_count  = gbuffer_meta_pixel_count;
  properties.thread_count         = thread_count;
  properties.task_count           = total_task_count;

  bool buffers_have_changed;
  __FAILURE_HANDLE(device_work_buffers_update(device->work_buffers, &properties, &buffers_have_changed));

  if (buffers_have_changed) {
    DeviceWorkBuffersPtrs ptrs;
    __FAILURE_HANDLE(device_work_buffers_get_ptrs(device->work_buffers, &ptrs));

    DEVICE_UPDATE_CONSTANT_MEMORY(ptrs.task_states, (void*) ptrs.task_states);
    DEVICE_UPDATE_CONSTANT_MEMORY(ptrs.task_direct_light, (void*) ptrs.task_direct_light);
    DEVICE_UPDATE_CONSTANT_MEMORY(ptrs.task_results, (void*) ptrs.task_results);
    DEVICE_UPDATE_CONSTANT_MEMORY(ptrs.results_counts, (void*) ptrs.results_counts);
    DEVICE_UPDATE_CONSTANT_MEMORY(ptrs.trace_counts, (void*) ptrs.trace_counts);
    DEVICE_UPDATE_CONSTANT_MEMORY(ptrs.task_counts, (void*) ptrs.task_counts);
    DEVICE_UPDATE_CONSTANT_MEMORY(ptrs.task_offsets, (void*) ptrs.task_offsets);
    DEVICE_UPDATE_CONSTANT_MEMORY(ptrs.frame_swap, (void*) ptrs.frame_swap);
    DEVICE_UPDATE_CONSTANT_MEMORY(ptrs.gbuffer_meta, (void*) ptrs.gbuffer_meta);

    for (uint32_t channel_id = 0; channel_id < FRAME_CHANNEL_COUNT; channel_id++) {
      DEVICE_UPDATE_CONSTANT_MEMORY(ptrs.frame_first_moment[channel_id], (void*) ptrs.frame_first_moment[channel_id]);
      DEVICE_UPDATE_CONSTANT_MEMORY(ptrs.frame_second_moment[channel_id], (void*) ptrs.frame_second_moment[channel_id]);
      DEVICE_UPDATE_CONSTANT_MEMORY(ptrs.frame_result[channel_id], (void*) ptrs.frame_result[channel_id]);
      DEVICE_UPDATE_CONSTANT_MEMORY(ptrs.frame_output[channel_id], (void*) ptrs.frame_output[channel_id]);
    }

    const size_t gbuffer_size = sizeof(GBufferMetaData) * gbuffer_meta_pixel_count;

    __FAILURE_HANDLE(device_malloc_staging(&device->gbuffer_meta_dst, gbuffer_size, DEVICE_MEMORY_STAGING_FLAG_NONE));
    memset(device->gbuffer_meta_dst, 0, gbuffer_size);

    DEVICE_UPDATE_CONSTANT_MEMORY(config.num_tasks_per_thread, tasks_per_thread);
  }

  return LUMINARY_SUCCESS;
}

static LuminaryResult _device_free_buffers(Device* device) {
  __CHECK_NULL_ARGUMENT(device);

  __DEVICE_BUFFER_FREE(textures);
  __DEVICE_BUFFER_FREE(materials);
  __DEVICE_BUFFER_FREE(triangles);
  __DEVICE_BUFFER_FREE(triangle_counts);
  __DEVICE_BUFFER_FREE(instance_mesh_id);
  __DEVICE_BUFFER_FREE(instance_transforms);
  __DEVICE_BUFFER_FREE(light_tree_root);
  __DEVICE_BUFFER_FREE(light_tree_nodes);
  __DEVICE_BUFFER_FREE(light_tree_tri_handle_map);

  return LUMINARY_SUCCESS;
}

////////////////////////////////////////////////////////////////////
// External API implementation
////////////////////////////////////////////////////////////////////

LuminaryResult device_create(Device** _device, uint32_t index) {
  __CHECK_NULL_ARGUMENT(_device);

  Device* device;
  __FAILURE_HANDLE(host_malloc(&device, sizeof(Device)));
  memset(device, 0, sizeof(Device));

  device->index                = index;
  device->optix_callback_error = false;
  device->exit_requested       = false;
  device->is_main_device       = false;
  device->state                = DEVICE_STATE_ENABLED;

  __FAILURE_HANDLE(_device_reset_constant_memory_dirty(device));

  CUDA_FAILURE_HANDLE(cuDeviceGet(&device->cuda_device, device->index));

  __FAILURE_HANDLE(_device_get_properties(&device->properties, device));

  ////////////////////////////////////////////////////////////////////
  // CUDA context creation
  ////////////////////////////////////////////////////////////////////

  CUDA_FAILURE_HANDLE(cuCtxCreate(&device->cuda_ctx, (CUctxCreateParams*) 0, 0, device->cuda_device));

  CUDA_FAILURE_HANDLE(cuCtxPushCurrent(device->cuda_ctx));

  ////////////////////////////////////////////////////////////////////
  // OptiX context creation
  ////////////////////////////////////////////////////////////////////

  OptixDeviceContextOptions optix_device_context_options;
  memset(&optix_device_context_options, 0, sizeof(OptixDeviceContextOptions));

#ifdef OPTIX_VALIDATION
  // https://forums.developer.nvidia.com/t/avoid-synchronization-in-optixlaunch/221458
  warn_message("OptiX validation is turned on. This will serialize all OptiX calls!");
  optix_device_context_options.logCallbackData     = (void*) 0;
  optix_device_context_options.logCallbackFunction = _device_optix_log_callback;
  optix_device_context_options.logCallbackLevel    = 3;
  optix_device_context_options.validationMode      = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
#endif

  OPTIX_FAILURE_HANDLE(optixDeviceContextCreate(device->cuda_ctx, &optix_device_context_options, &device->optix_ctx));

  __FAILURE_HANDLE(_device_get_optix_properties(device));

  ////////////////////////////////////////////////////////////////////
  // Stream creation
  ////////////////////////////////////////////////////////////////////

  CUDA_FAILURE_HANDLE(cuStreamCreateWithPriority(&device->stream_main, CU_STREAM_NON_BLOCKING, 2));
  CUDA_FAILURE_HANDLE(cuStreamCreateWithPriority(&device->stream_output, CU_STREAM_NON_BLOCKING, 3));
  CUDA_FAILURE_HANDLE(cuStreamCreateWithPriority(&device->stream_abort, CU_STREAM_NON_BLOCKING, 0));
  CUDA_FAILURE_HANDLE(cuStreamCreateWithPriority(&device->stream_callbacks, CU_STREAM_NON_BLOCKING, 1));

  ////////////////////////////////////////////////////////////////////
  // Event creation
  ////////////////////////////////////////////////////////////////////

  CUDA_FAILURE_HANDLE(cuEventCreate(&device->event_queue_render, CU_EVENT_DISABLE_TIMING));
  CUDA_FAILURE_HANDLE(cuEventCreate(&device->event_queue_gbuffer_meta, CU_EVENT_DISABLE_TIMING));

  ////////////////////////////////////////////////////////////////////
  // Constant memory initialization
  ////////////////////////////////////////////////////////////////////

  __FAILURE_HANDLE(
    device_malloc_staging(&device->constant_memory, sizeof(DeviceConstantMemory), DEVICE_MEMORY_STAGING_FLAG_PCIE_TRANSFER_ONLY));
  memset(device->constant_memory, 0, sizeof(DeviceConstantMemory));

  __FAILURE_HANDLE(device_staging_manager_create(&device->staging_manager, device));
  __FAILURE_HANDLE(_device_setup_execution_config(device));

  __FAILURE_HANDLE(array_create(&device->textures, sizeof(DeviceTexture*), 16));

  ////////////////////////////////////////////////////////////////////
  // Optix data
  ////////////////////////////////////////////////////////////////////

  __FAILURE_HANDLE(array_create(&device->meshes, sizeof(DeviceMesh*), 4));
  __FAILURE_HANDLE(array_create(&device->omms, sizeof(OpacityMicromap*), 4));
  __FAILURE_HANDLE(optix_bvh_instance_cache_create(&device->optix_instance_cache, device));
  __FAILURE_HANDLE(optix_bvh_create(&device->optix_bvh_ias));
  __FAILURE_HANDLE(optix_bvh_create(&device->optix_bvh_light));

  ////////////////////////////////////////////////////////////////////
  // Initialize processing objects
  ////////////////////////////////////////////////////////////////////

  __FAILURE_HANDLE(device_embedded_data_create(&device->embedded_data));
  __FAILURE_HANDLE(device_work_buffers_create(&device->work_buffers));
  __FAILURE_HANDLE(device_renderer_create(&device->renderer));
  __FAILURE_HANDLE(device_output_create(&device->output));
  __FAILURE_HANDLE(device_post_create(&device->post));
  __FAILURE_HANDLE(device_abort_create(&device->abort));

  ////////////////////////////////////////////////////////////////////
  // Initialize scene entity LUT objects
  ////////////////////////////////////////////////////////////////////

  __FAILURE_HANDLE(device_sky_lut_create(&device->sky_lut));
  __FAILURE_HANDLE(device_sky_hdri_create(&device->sky_hdri));
  __FAILURE_HANDLE(device_sky_stars_create(&device->sky_stars));
  __FAILURE_HANDLE(device_bsdf_lut_create(&device->bsdf_lut));
  __FAILURE_HANDLE(device_physical_camera_create(&device->physical_camera));
  __FAILURE_HANDLE(device_cloud_noise_create(&device->cloud_noise, device));
  __FAILURE_HANDLE(device_particles_handle_create(&device->particles_handle));
  __FAILURE_HANDLE(device_adaptive_sampler_create(&device->adaptive_sampler));

  ////////////////////////////////////////////////////////////////////
  // Initialize abort flag
  ////////////////////////////////////////////////////////////////////

  DeviceAbortDeviceBufferPtrs abort_buffer_ptrs;
  __FAILURE_HANDLE(device_abort_get_ptrs(device->abort, &abort_buffer_ptrs));

  DEVICE_UPDATE_CONSTANT_MEMORY(ptrs.abort_flag, (void*) abort_buffer_ptrs.abort_flag);

  __FAILURE_HANDLE(device_abort_set(device->abort, device, false));

  CUDA_FAILURE_HANDLE(cuCtxPopCurrent(&device->cuda_ctx));

  __FAILURE_HANDLE(_device_print_info(device));

  *_device = device;

  return LUMINARY_SUCCESS;
}

LuminaryResult device_register_as_main(Device* device) {
  __CHECK_NULL_ARGUMENT(device);

  if (device->state == DEVICE_STATE_UNAVAILABLE) {
    __RETURN_ERROR(
      LUMINARY_ERROR_API_EXCEPTION, "Device %s was registered as the main device but it is not available.", device->properties.name);
  }

  if (device->state == DEVICE_STATE_DISABLED) {
    __RETURN_ERROR(
      LUMINARY_ERROR_API_EXCEPTION, "Device %s was registered as the main device but it is not enabled.", device->properties.name);
  }

  device->is_main_device = true;

  return LUMINARY_SUCCESS;
}

LuminaryResult device_unregister_as_main(Device* device) {
  __CHECK_NULL_ARGUMENT(device);

  device->is_main_device = false;

  return LUMINARY_SUCCESS;
}

LuminaryResult device_set_enable(Device* device, bool enable) {
  __CHECK_NULL_ARGUMENT(device);

  if (device->state == DEVICE_STATE_UNAVAILABLE && enable) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Tried to enable device %s but is is not available.", device->properties.name);
  }

  device->state = (enable) ? DEVICE_STATE_ENABLED : DEVICE_STATE_DISABLED;

  return LUMINARY_SUCCESS;
}

LuminaryResult device_compile_kernels(Device* device, CUlibrary library) {
  __CHECK_NULL_ARGUMENT(device);

  if (library == (CUlibrary) 0) {
    warn_message("Deactivating %s because no CUBIN is present for it.", device->properties.name);
    device->state = DEVICE_STATE_UNAVAILABLE;
    return LUMINARY_SUCCESS;
  }

  CUDA_FAILURE_HANDLE(cuCtxPushCurrent(device->cuda_ctx));

  for (uint32_t kernel_id = 0; kernel_id < CUDA_KERNEL_TYPE_COUNT; kernel_id++) {
    __FAILURE_HANDLE(kernel_create(&device->cuda_kernels[kernel_id], device, library, kernel_id));
  }

  for (uint32_t kernel_id = 0; kernel_id < OPTIX_KERNEL_TYPE_COUNT; kernel_id++) {
    __FAILURE_HANDLE(optix_kernel_create(&device->optix_kernels[kernel_id], device, kernel_id));

    if (device->optix_kernels[kernel_id]->available == false) {
      warn_message("Deactivating %s because an OptiX kernel is missing.", device->properties.name);
      device->state = DEVICE_STATE_UNAVAILABLE;
      return LUMINARY_SUCCESS;
    }
  }

  size_t const_memory_size;
  CUDA_FAILURE_HANDLE(cuLibraryGetGlobal(&device->cuda_device_const_memory, &const_memory_size, library, "device"));

  if (const_memory_size != sizeof(DeviceConstantMemory)) {
    __RETURN_ERROR(
      LUMINARY_ERROR_API_EXCEPTION, "Const memory is expected to be %llu bytes in size but is %llu.", sizeof(DeviceConstantMemory),
      const_memory_size);
  }

  OPTIX_CHECK_CALLBACK_ERROR(device);

  CUDA_FAILURE_HANDLE(cuCtxPopCurrent(&device->cuda_ctx));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_load_embedded_data(Device* device) {
  __CHECK_NULL_ARGUMENT(device);

  DEVICE_ASSERT_AVAILABLE

  CUDA_FAILURE_HANDLE(cuCtxPushCurrent(device->cuda_ctx));

  bool buffers_have_changed;
  __FAILURE_HANDLE(device_embedded_data_update(device->embedded_data, device, &buffers_have_changed));

  if (buffers_have_changed) {
    DeviceEmbeddedDataPtrs ptrs;
    __FAILURE_HANDLE(device_embedded_data_get_ptrs(device->embedded_data, &ptrs));

    DEVICE_UPDATE_CONSTANT_MEMORY(ptrs.bluenoise_1D, (void*) ptrs.bluenoise_1D);
    DEVICE_UPDATE_CONSTANT_MEMORY(ptrs.bluenoise_2D, (void*) ptrs.bluenoise_2D);
    DEVICE_UPDATE_CONSTANT_MEMORY(ptrs.bridge_lut, (void*) ptrs.bridge_lut);
    DEVICE_UPDATE_CONSTANT_MEMORY(ptrs.spectral_cdf, (void*) ptrs.spectral_cdf);

    DEVICE_UPDATE_CONSTANT_MEMORY(moon_albedo_tex, ptrs.moon_albedo_tex);
    DEVICE_UPDATE_CONSTANT_MEMORY(moon_normal_tex, ptrs.moon_normal_tex);
    DEVICE_UPDATE_CONSTANT_MEMORY(spectral_xy_lut_tex, ptrs.spectral_xy_tex);
    DEVICE_UPDATE_CONSTANT_MEMORY(spectral_z_lut_tex, ptrs.spectral_z_tex);
  }

  CUDA_FAILURE_HANDLE(cuCtxPopCurrent(&device->cuda_ctx));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_get_internal_resolution(Device* device, uint32_t* width, uint32_t* height) {
  __CHECK_NULL_ARGUMENT(device);
  __CHECK_NULL_ARGUMENT(width);
  __CHECK_NULL_ARGUMENT(height);

  *width  = device->constant_memory->settings.width;
  *height = device->constant_memory->settings.height;

  return LUMINARY_SUCCESS;
}

LuminaryResult device_get_internal_render_resolution(Device* device, uint32_t* width, uint32_t* height) {
  __CHECK_NULL_ARGUMENT(device);
  __CHECK_NULL_ARGUMENT(width);
  __CHECK_NULL_ARGUMENT(height);

  *width  = device->constant_memory->settings.window_width;
  *height = device->constant_memory->settings.window_height;

  return LUMINARY_SUCCESS;
}

LuminaryResult device_get_allocated_task_count(Device* device, uint32_t* task_count) {
  __CHECK_NULL_ARGUMENT(device);
  __CHECK_NULL_ARGUMENT(task_count);

  const uint32_t num_threads = device->constant_memory->config.num_blocks * THREADS_PER_BLOCK;

  *task_count = num_threads * device->constant_memory->config.num_tasks_per_thread;

  return LUMINARY_SUCCESS;
}

LuminaryResult device_get_current_pixels_per_thread(Device* device, uint32_t* pixels_per_thread) {
  __CHECK_NULL_ARGUMENT(device);
  __CHECK_NULL_ARGUMENT(pixels_per_thread);

  const uint32_t num_threads = device->constant_memory->config.num_blocks * THREADS_PER_BLOCK;
  const uint32_t num_pixels  = device->constant_memory->settings.width * device->constant_memory->settings.height;

  const uint32_t num_current_pixels = num_pixels >> ((device->undersampling_state & UNDERSAMPLING_STAGE_MASK) >> UNDERSAMPLING_STAGE_SHIFT);

  *pixels_per_thread = (num_current_pixels + num_threads - 1) / num_threads;

  return LUMINARY_SUCCESS;
}

LuminaryResult device_update_scene_entity(Device* device, const void* object, SceneEntity entity) {
  __CHECK_NULL_ARGUMENT(device);
  __CHECK_NULL_ARGUMENT(object);

  DEVICE_ASSERT_AVAILABLE

  CUDA_FAILURE_HANDLE(cuCtxPushCurrent(device->cuda_ctx));

  const DeviceConstantMemoryMember member = device_scene_entity_to_const_memory_member[entity];
  const size_t member_offset              = device_cuda_const_memory_offsets[member];
  const size_t member_size                = device_cuda_const_memory_sizes[member];

  memcpy(((uint8_t*) device->constant_memory) + member_offset, object, member_size);

  __FAILURE_HANDLE(_device_set_constant_memory_dirty(device, member));

  CUDA_FAILURE_HANDLE(cuCtxPopCurrent(&device->cuda_ctx));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_update_dynamic_const_mem(Device* device, DeviceSampleAllocation sample_allocation) {
  __CHECK_NULL_ARGUMENT(device);

  DEVICE_ASSERT_AVAILABLE

  CUDA_FAILURE_HANDLE(cuCtxPushCurrent(device->cuda_ctx));

  // These have to be done through a memset. If a memcpy would be used, the abort flag memcpy would stall until these here
  // have completed. If the abort flag would use a memset, then that one would stall until all kernels have completed. Hence
  // it is mandatory that the renderer NEVER queues any host to device memcpys.

  CUDA_FAILURE_HANDLE(
    cuMemsetD8Async(device->cuda_device_const_memory + offsetof(DeviceConstantMemory, state.depth), 0, 1, device->stream_main));
  CUDA_FAILURE_HANDLE(cuMemsetD8Async(
    device->cuda_device_const_memory + offsetof(DeviceConstantMemory, state.undersampling), device->undersampling_state, 1,
    device->stream_main));

  const CUdeviceptr stage_sample_offsets_ptr =
    device->cuda_device_const_memory + offsetof(DeviceConstantMemory, state.sample_allocation.stage_sample_offsets);
  for (uint32_t stage_id = 0; stage_id < ADAPTIVE_SAMPLER_NUM_STAGES + 1; stage_id++) {
    const CUdeviceptr dst = stage_sample_offsets_ptr + sizeof(uint32_t) * stage_id;
    CUDA_FAILURE_HANDLE(cuMemsetD32Async(dst, sample_allocation.stage_sample_offsets[stage_id], 1, device->stream_main));
  }

  CUDA_FAILURE_HANDLE(cuMemsetD32Async(
    device->cuda_device_const_memory + offsetof(DeviceConstantMemory, state.sample_allocation.global_sample_id),
    sample_allocation.global_sample_id, 1, device->stream_main));
  CUDA_FAILURE_HANDLE(cuMemsetD32Async(
    device->cuda_device_const_memory + offsetof(DeviceConstantMemory, state.sample_allocation.upper_bound_tasks_per_sample),
    sample_allocation.upper_bound_tasks_per_sample, 1, device->stream_main));
  CUDA_FAILURE_HANDLE(cuMemsetD8Async(
    device->cuda_device_const_memory + offsetof(DeviceConstantMemory, state.sample_allocation.stage_id), sample_allocation.stage_id, 1,
    device->stream_main));
  CUDA_FAILURE_HANDLE(cuMemsetD8Async(
    device->cuda_device_const_memory + offsetof(DeviceConstantMemory, state.sample_allocation.num_samples), sample_allocation.num_samples,
    1, device->stream_main));

  CUDA_FAILURE_HANDLE(cuCtxPopCurrent(&device->cuda_ctx));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_update_tile_id_const_mem(Device* device, uint32_t tile_id) {
  __CHECK_NULL_ARGUMENT(device);

  DEVICE_ASSERT_AVAILABLE

  CUDA_FAILURE_HANDLE(cuCtxPushCurrent(device->cuda_ctx));

  CUDA_FAILURE_HANDLE(
    cuMemsetD32Async(device->cuda_device_const_memory + offsetof(DeviceConstantMemory, state.tile_id), tile_id, 1, device->stream_main));

  CUDA_FAILURE_HANDLE(cuCtxPopCurrent(&device->cuda_ctx));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_update_depth_const_mem(Device* device, uint8_t depth) {
  __CHECK_NULL_ARGUMENT(device);

  DEVICE_ASSERT_AVAILABLE

  CUDA_FAILURE_HANDLE(cuCtxPushCurrent(device->cuda_ctx));

  CUDA_FAILURE_HANDLE(
    cuMemsetD8Async(device->cuda_device_const_memory + offsetof(DeviceConstantMemory, state.depth), depth, 1, device->stream_main));

  CUDA_FAILURE_HANDLE(cuCtxPopCurrent(&device->cuda_ctx));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_sync_constant_memory(Device* device) {
  __CHECK_NULL_ARGUMENT(device);

  DEVICE_ASSERT_AVAILABLE

  CUDA_FAILURE_HANDLE(cuCtxPushCurrent(device->cuda_ctx));

  __FAILURE_HANDLE(_device_update_constant_memory(device));

  CUDA_FAILURE_HANDLE(cuCtxPopCurrent(&device->cuda_ctx));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_allocate_work_buffers(Device* device) {
  __CHECK_NULL_ARGUMENT(device);

  DEVICE_ASSERT_AVAILABLE

  CUDA_FAILURE_HANDLE(cuCtxPushCurrent(device->cuda_ctx));

  __FAILURE_HANDLE(_device_allocate_work_buffers(device));
  __FAILURE_HANDLE(device_post_allocate(device->post, device->constant_memory->settings.width, device->constant_memory->settings.height));

  CUDA_FAILURE_HANDLE(cuCtxPopCurrent(&device->cuda_ctx));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_update_mesh(Device* device, const Mesh* mesh) {
  __CHECK_NULL_ARGUMENT(device);
  __CHECK_NULL_ARGUMENT(mesh);

  DEVICE_ASSERT_AVAILABLE

  CUDA_FAILURE_HANDLE(cuCtxPushCurrent(device->cuda_ctx));

  ////////////////////////////////////////////////////////////////////
  // Compute new device mesh
  ////////////////////////////////////////////////////////////////////

  uint32_t num_meshes;
  __FAILURE_HANDLE(array_get_num_elements(device->meshes, &num_meshes));

  if (mesh->id > num_meshes) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Meshes were not added in sequence.");
  }

  if (mesh->id < num_meshes) {
    __FAILURE_HANDLE(device_mesh_destroy(device->meshes + mesh->id));
  }

  DeviceMesh* device_mesh;
  OpacityMicromap* omm;

  if (mesh->id == num_meshes) {
    __FAILURE_HANDLE(device_mesh_create(&device_mesh, device, mesh));
    __FAILURE_HANDLE(array_push(&device->meshes, &device_mesh));

    __FAILURE_HANDLE(omm_create(&omm));
    __FAILURE_HANDLE(array_push(&device->omms, &omm));

    num_meshes++;
    __DEVICE_BUFFER_REALLOC(triangles, sizeof(DeviceTriangle*) * num_meshes);
    __DEVICE_BUFFER_REALLOC(triangle_counts, sizeof(uint32_t) * num_meshes);
  }
  else {
    device_mesh = device->meshes[mesh->id];
    omm         = device->omms[mesh->id];
  }

  void** direct_access_buffer;
  __FAILURE_HANDLE(device_staging_manager_register_direct_access(
    device->staging_manager, (void*) device->buffers.triangles, sizeof(DeviceTriangle*) * mesh->id, sizeof(DeviceTriangle*),
    (void**) &direct_access_buffer));

  *direct_access_buffer = DEVICE_PTR(device->meshes[mesh->id]->triangles);

  __FAILURE_HANDLE(device_staging_manager_register_direct_access(
    device->staging_manager, (void*) device->buffers.triangle_counts, sizeof(uint32_t) * mesh->id, sizeof(uint32_t),
    (void**) &direct_access_buffer));

  *(uint32_t*) direct_access_buffer = mesh->data.triangle_count;

  device->meshes_need_building = true;

  CUDA_FAILURE_HANDLE(cuCtxPopCurrent(&device->cuda_ctx));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_add_textures(Device* device, const Texture** textures, uint32_t num_textures) {
  __CHECK_NULL_ARGUMENT(device);
  __CHECK_NULL_ARGUMENT(textures);

  DEVICE_ASSERT_AVAILABLE

  CUDA_FAILURE_HANDLE(cuCtxPushCurrent(device->cuda_ctx));

  for (uint32_t texture_id = 0; texture_id < num_textures; texture_id++) {
    DeviceTexture* device_texture;
    __FAILURE_HANDLE(device_texture_create(&device_texture, textures[texture_id], device, device->stream_main));

    __FAILURE_HANDLE(array_push(&device->textures, &device_texture));
  }

  uint32_t texture_object_count;
  __FAILURE_HANDLE(array_get_num_elements(device->textures, &texture_object_count));

  const size_t total_texture_object_size = sizeof(DeviceTextureObject) * texture_object_count;

  __DEVICE_BUFFER_ALLOCATE(textures, total_texture_object_size);

  DeviceTextureObject* buffer;
  __FAILURE_HANDLE(device_staging_manager_register_direct_access(
    device->staging_manager, (void*) device->buffers.textures, 0, total_texture_object_size, (void**) &buffer));

  for (uint32_t texture_object_id = 0; texture_object_id < texture_object_count; texture_object_id++) {
    __FAILURE_HANDLE(device_struct_texture_object_convert(device->textures[texture_object_id], buffer + texture_object_id));
  }

  CUDA_FAILURE_HANDLE(cuCtxPopCurrent(&device->cuda_ctx));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_apply_instance_updates(Device* device, const ARRAY MeshInstanceUpdate* instance_updates) {
  __CHECK_NULL_ARGUMENT(device);
  __CHECK_NULL_ARGUMENT(instance_updates);

  DEVICE_ASSERT_AVAILABLE

  CUDA_FAILURE_HANDLE(cuCtxPushCurrent(device->cuda_ctx));

  uint32_t num_updates;
  __FAILURE_HANDLE(array_get_num_elements(instance_updates, &num_updates));

  bool new_instances_are_added = false;

  for (uint32_t update_id = 0; update_id < num_updates; update_id++) {
    const uint32_t instance_id = instance_updates[update_id].instance_id;

    if (instance_id >= device->num_instances) {
      device->num_instances   = instance_id + 1;
      new_instances_are_added = true;
    }
  }

  if (new_instances_are_added) {
    __DEVICE_BUFFER_REALLOC(instance_mesh_id, sizeof(uint32_t) * device->num_instances);
    __DEVICE_BUFFER_REALLOC(instance_transforms, sizeof(DeviceTransform) * device->num_instances);
  }

  for (uint32_t update_id = 0; update_id < num_updates; update_id++) {
    const uint32_t instance_id = instance_updates[update_id].instance_id;
    const uint32_t mesh_id     = instance_updates[update_id].instance.mesh_id;

    __FAILURE_HANDLE(device_staging_manager_register(
      device->staging_manager, &mesh_id, (DEVICE void*) device->buffers.instance_mesh_id, sizeof(uint32_t) * instance_id,
      sizeof(uint32_t)));

    DeviceTransform* transform;

    __FAILURE_HANDLE(device_staging_manager_register_direct_access(
      device->staging_manager, (DEVICE void*) device->buffers.instance_transforms, sizeof(DeviceTransform) * instance_id,
      sizeof(DeviceTransform), (void**) &transform));

    __FAILURE_HANDLE(device_struct_instance_transform_convert(&instance_updates[update_id].instance, transform));
  }

  if (device->meshes_need_building) {
    uint32_t num_meshes;
    __FAILURE_HANDLE(array_get_num_elements(device->meshes, &num_meshes));

    for (uint32_t mesh_id = 0; mesh_id < num_meshes; mesh_id++) {
      __FAILURE_HANDLE(device_mesh_build_structures(device->meshes[mesh_id], device->omms[mesh_id], device));
    }

    device->meshes_need_building = false;
  }

  __FAILURE_HANDLE(optix_bvh_instance_cache_update(device->optix_instance_cache, instance_updates));
  __FAILURE_HANDLE(optix_bvh_ias_build(device->optix_bvh_ias, device));

  DEVICE_UPDATE_CONSTANT_MEMORY(optix_bvh, device->optix_bvh_ias->traversable[OPTIX_BVH_TYPE_DEFAULT]);
  DEVICE_UPDATE_CONSTANT_MEMORY(optix_bvh_shadow, device->optix_bvh_ias->traversable[OPTIX_BVH_TYPE_SHADOW]);

  CUDA_FAILURE_HANDLE(cuCtxPopCurrent(&device->cuda_ctx));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_apply_material_updates(
  Device* device, const ARRAY MaterialUpdate* updates, const ARRAY DeviceMaterialCompressed* materials) {
  __CHECK_NULL_ARGUMENT(device);
  __CHECK_NULL_ARGUMENT(updates);
  __CHECK_NULL_ARGUMENT(materials);

  DEVICE_ASSERT_AVAILABLE

  CUDA_FAILURE_HANDLE(cuCtxPushCurrent(device->cuda_ctx));

  uint32_t num_updates;
  __FAILURE_HANDLE(array_get_num_elements(updates, &num_updates));

  bool new_materials_are_added = false;

  for (uint32_t update_id = 0; update_id < num_updates; update_id++) {
    const uint32_t material_id = updates[update_id].material_id;

    if (material_id >= device->num_materials) {
      device->num_materials   = material_id + 1;
      new_materials_are_added = true;
    }
  }

  if (new_materials_are_added) {
    __DEVICE_BUFFER_REALLOC(materials, sizeof(DeviceMaterialCompressed) * device->num_materials);
  }

  for (uint32_t update_id = 0; update_id < num_updates; update_id++) {
    const uint32_t material_id = updates[update_id].material_id;

    __FAILURE_HANDLE(device_staging_manager_register(
      device->staging_manager, materials + update_id, (DEVICE void*) device->buffers.materials,
      sizeof(DeviceMaterialCompressed) * material_id, sizeof(DeviceMaterialCompressed)));
  }

  CUDA_FAILURE_HANDLE(cuCtxPopCurrent(&device->cuda_ctx));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_build_light_tree(Device* device, LightTree* tree) {
  __CHECK_NULL_ARGUMENT(device);
  __CHECK_NULL_ARGUMENT(tree);

  DEVICE_ASSERT_AVAILABLE

  CUDA_FAILURE_HANDLE(cuCtxPushCurrent(device->cuda_ctx));

  __FAILURE_HANDLE(light_tree_build(tree, device));

  CUDA_FAILURE_HANDLE(cuCtxPopCurrent(&device->cuda_ctx));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_update_light_tree_data(Device* device, LightTree* tree) {
  __CHECK_NULL_ARGUMENT(device);
  __CHECK_NULL_ARGUMENT(tree);

  DEVICE_ASSERT_AVAILABLE

  CUDA_FAILURE_HANDLE(cuCtxPushCurrent(device->cuda_ctx));

  __DEVICE_BUFFER_FREE(light_tree_root);
  __DEVICE_BUFFER_FREE(light_tree_nodes);
  __DEVICE_BUFFER_FREE(light_tree_tri_handle_map);

  __DEVICE_BUFFER_ALLOCATE(light_tree_root, tree->root_size);
  __DEVICE_BUFFER_ALLOCATE(light_tree_nodes, tree->nodes_size);
  __DEVICE_BUFFER_ALLOCATE(light_tree_tri_handle_map, tree->tri_handle_map_size);

  __FAILURE_HANDLE(device_staging_manager_register(
    device->staging_manager, tree->root_data, (DEVICE void*) device->buffers.light_tree_root, 0, tree->root_size));
  __FAILURE_HANDLE(device_staging_manager_register(
    device->staging_manager, tree->nodes_data, (DEVICE void*) device->buffers.light_tree_nodes, 0, tree->nodes_size));
  __FAILURE_HANDLE(device_staging_manager_register(
    device->staging_manager, tree->tri_handle_map_data, (DEVICE void*) device->buffers.light_tree_tri_handle_map, 0,
    tree->tri_handle_map_size));

  __FAILURE_HANDLE(optix_bvh_light_build(device->optix_bvh_light, device, tree));

  DEVICE_UPDATE_CONSTANT_MEMORY(optix_bvh_light, device->optix_bvh_light->traversable[OPTIX_BVH_TYPE_DEFAULT]);

  CUDA_FAILURE_HANDLE(cuCtxPopCurrent(&device->cuda_ctx));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_unload_light_tree(Device* device, LightTree* tree) {
  __CHECK_NULL_ARGUMENT(device);
  __CHECK_NULL_ARGUMENT(tree);

  DEVICE_ASSERT_AVAILABLE

  CUDA_FAILURE_HANDLE(cuCtxPushCurrent(device->cuda_ctx));

  __FAILURE_HANDLE(light_tree_unload_integrator(tree));

  CUDA_FAILURE_HANDLE(cuCtxPopCurrent(&device->cuda_ctx));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_build_sky_lut(Device* device, SkyLUT* sky_lut) {
  __CHECK_NULL_ARGUMENT(device);
  __CHECK_NULL_ARGUMENT(sky_lut);

  DEVICE_ASSERT_AVAILABLE

  CUDA_FAILURE_HANDLE(cuCtxPushCurrent(device->cuda_ctx));

  __FAILURE_HANDLE(sky_lut_generate(sky_lut, device));

  CUDA_FAILURE_HANDLE(cuCtxPopCurrent(&device->cuda_ctx));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_update_sky_lut(Device* device, const SkyLUT* sky_lut) {
  __CHECK_NULL_ARGUMENT(device);
  __CHECK_NULL_ARGUMENT(sky_lut);

  DEVICE_ASSERT_AVAILABLE

  CUDA_FAILURE_HANDLE(cuCtxPushCurrent(device->cuda_ctx));

  bool luts_have_changed = false;
  __FAILURE_HANDLE(device_sky_lut_update(device->sky_lut, device, sky_lut, &luts_have_changed));

  if (luts_have_changed) {
    __FAILURE_HANDLE(
      device_struct_texture_object_convert(device->sky_lut->transmittance_low, &device->constant_memory->sky_lut_transmission_low_tex));
    __FAILURE_HANDLE(
      device_struct_texture_object_convert(device->sky_lut->transmittance_high, &device->constant_memory->sky_lut_transmission_high_tex));
    __FAILURE_HANDLE(device_struct_texture_object_convert(
      device->sky_lut->multiscattering_low, &device->constant_memory->sky_lut_multiscattering_low_tex));
    __FAILURE_HANDLE(device_struct_texture_object_convert(
      device->sky_lut->multiscattering_high, &device->constant_memory->sky_lut_multiscattering_high_tex));

    __FAILURE_HANDLE(_device_set_constant_memory_dirty(device, DEVICE_CONSTANT_MEMORY_MEMBER_SKY_LUT_TEX));
  }

  CUDA_FAILURE_HANDLE(cuCtxPopCurrent(&device->cuda_ctx));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_build_sky_hdri(Device* device, SkyHDRI* sky_hdri) {
  __CHECK_NULL_ARGUMENT(device);
  __CHECK_NULL_ARGUMENT(sky_hdri);

  DEVICE_ASSERT_AVAILABLE

  CUDA_FAILURE_HANDLE(cuCtxPushCurrent(device->cuda_ctx));

  __FAILURE_HANDLE(sky_hdri_generate(sky_hdri, device));

  CUDA_FAILURE_HANDLE(cuCtxPopCurrent(&device->cuda_ctx));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_update_sky_hdri(Device* device, const SkyHDRI* sky_hdri) {
  __CHECK_NULL_ARGUMENT(device);
  __CHECK_NULL_ARGUMENT(sky_hdri);

  DEVICE_ASSERT_AVAILABLE

  CUDA_FAILURE_HANDLE(cuCtxPushCurrent(device->cuda_ctx));

  bool buffers_have_changed;
  __FAILURE_HANDLE(device_sky_hdri_update(device->sky_hdri, device, sky_hdri, &buffers_have_changed));

  if (buffers_have_changed) {
    DeviceSkyHDRIPtrs ptrs;
    __FAILURE_HANDLE(device_sky_hdri_get_ptrs(device->sky_hdri, &ptrs));

    DEVICE_UPDATE_CONSTANT_MEMORY(sky_hdri_color_tex, ptrs.color_tex_obj);
    DEVICE_UPDATE_CONSTANT_MEMORY(sky_hdri_shadow_tex, ptrs.shadow_tex_obj);
  }

  CUDA_FAILURE_HANDLE(cuCtxPopCurrent(&device->cuda_ctx));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_update_sky_stars(Device* device, const SkyStars* sky_stars) {
  __CHECK_NULL_ARGUMENT(device);
  __CHECK_NULL_ARGUMENT(sky_stars);

  DEVICE_ASSERT_AVAILABLE

  CUDA_FAILURE_HANDLE(cuCtxPushCurrent(device->cuda_ctx));

  bool buffers_have_changed;
  __FAILURE_HANDLE(device_sky_stars_update(device->sky_stars, device, sky_stars, &buffers_have_changed));

  if (buffers_have_changed) {
    DeviceSkyStarsPtrs ptrs;
    __FAILURE_HANDLE(device_sky_stars_get_ptrs(device->sky_stars, &ptrs));

    DEVICE_UPDATE_CONSTANT_MEMORY(ptrs.stars, (void*) ptrs.data);
    DEVICE_UPDATE_CONSTANT_MEMORY(ptrs.stars_offsets, (void*) ptrs.offsets);
  }

  CUDA_FAILURE_HANDLE(cuCtxPopCurrent(&device->cuda_ctx));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_build_bsdf_lut(Device* device, BSDFLUT* bsdf_lut) {
  __CHECK_NULL_ARGUMENT(device);
  __CHECK_NULL_ARGUMENT(bsdf_lut);

  DEVICE_ASSERT_AVAILABLE

  CUDA_FAILURE_HANDLE(cuCtxPushCurrent(device->cuda_ctx));

  __FAILURE_HANDLE(bsdf_lut_generate(bsdf_lut, device));

  CUDA_FAILURE_HANDLE(cuCtxPopCurrent(&device->cuda_ctx));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_update_bsdf_lut(Device* device, const BSDFLUT* bsdf_lut) {
  __CHECK_NULL_ARGUMENT(device);
  __CHECK_NULL_ARGUMENT(bsdf_lut);

  DEVICE_ASSERT_AVAILABLE

  CUDA_FAILURE_HANDLE(cuCtxPushCurrent(device->cuda_ctx));

  __FAILURE_HANDLE(device_bsdf_lut_update(device->bsdf_lut, device, bsdf_lut));

  __FAILURE_HANDLE(device_struct_texture_object_convert(device->bsdf_lut->conductor, &device->constant_memory->bsdf_lut_conductor));
  __FAILURE_HANDLE(device_struct_texture_object_convert(device->bsdf_lut->specular, &device->constant_memory->bsdf_lut_glossy));
  __FAILURE_HANDLE(device_struct_texture_object_convert(device->bsdf_lut->dielectric, &device->constant_memory->bsdf_lut_dielectric));
  __FAILURE_HANDLE(
    device_struct_texture_object_convert(device->bsdf_lut->dielectric_inv, &device->constant_memory->bsdf_lut_dielectric_inv));

  __FAILURE_HANDLE(_device_set_constant_memory_dirty(device, DEVICE_CONSTANT_MEMORY_MEMBER_BSDF_LUT_TEX));

  CUDA_FAILURE_HANDLE(cuCtxPopCurrent(&device->cuda_ctx));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_update_cloud_noise(Device* device, const Cloud* cloud) {
  __CHECK_NULL_ARGUMENT(device);
  __CHECK_NULL_ARGUMENT(cloud);

  DEVICE_ASSERT_AVAILABLE

  CUDA_FAILURE_HANDLE(cuCtxPushCurrent(device->cuda_ctx));

  __FAILURE_HANDLE(device_cloud_noise_generate(device->cloud_noise, cloud, device));

  __FAILURE_HANDLE(device_struct_texture_object_convert(device->cloud_noise->shape_tex, &device->constant_memory->cloud_noise_shape_tex));
  __FAILURE_HANDLE(device_struct_texture_object_convert(device->cloud_noise->detail_tex, &device->constant_memory->cloud_noise_detail_tex));
  __FAILURE_HANDLE(
    device_struct_texture_object_convert(device->cloud_noise->weather_tex, &device->constant_memory->cloud_noise_weather_tex));

  __FAILURE_HANDLE(_device_set_constant_memory_dirty(device, DEVICE_CONSTANT_MEMORY_MEMBER_CLOUD_NOISE_TEX));

  CUDA_FAILURE_HANDLE(cuCtxPopCurrent(&device->cuda_ctx));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_update_particles(Device* device, const Particles* particles) {
  __CHECK_NULL_ARGUMENT(device);
  __CHECK_NULL_ARGUMENT(particles);

  DEVICE_ASSERT_AVAILABLE

  CUDA_FAILURE_HANDLE(cuCtxPushCurrent(device->cuda_ctx));

  __FAILURE_HANDLE(device_particles_handle_generate(device->particles_handle, particles, device));

  if (particles->active) {
    DEVICE_UPDATE_CONSTANT_MEMORY(optix_bvh_particles, device->particles_handle->instance_bvh->traversable[OPTIX_BVH_TYPE_DEFAULT]);
    DEVICE_UPDATE_CONSTANT_MEMORY(ptrs.particle_quads, DEVICE_PTR(device->particles_handle->quad_buffer));
  }

  CUDA_FAILURE_HANDLE(cuCtxPopCurrent(&device->cuda_ctx));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_update_physical_camera(Device* device, const PhysicalCamera* physical_camera) {
  __CHECK_NULL_ARGUMENT(device);
  __CHECK_NULL_ARGUMENT(physical_camera);

  DEVICE_ASSERT_AVAILABLE

  CUDA_FAILURE_HANDLE(cuCtxPushCurrent(device->cuda_ctx));

  bool buffers_have_changed;
  __FAILURE_HANDLE(device_physical_camera_update(device->physical_camera, device, physical_camera, &buffers_have_changed));

  if (buffers_have_changed) {
    DevicePhysicalCameraPtrs ptrs;
    __FAILURE_HANDLE(device_physical_camera_get_ptrs(device->physical_camera, &ptrs));

    DEVICE_UPDATE_CONSTANT_MEMORY(ptrs.camera_interfaces, (void*) ptrs.camera_interfaces);
    DEVICE_UPDATE_CONSTANT_MEMORY(ptrs.camera_media, (void*) ptrs.camera_media);
  }

  CUDA_FAILURE_HANDLE(cuCtxPopCurrent(&device->cuda_ctx));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_update_post(Device* device, const Camera* camera) {
  __CHECK_NULL_ARGUMENT(device);
  __CHECK_NULL_ARGUMENT(camera);

  DEVICE_ASSERT_AVAILABLE

  CUDA_FAILURE_HANDLE(cuCtxPushCurrent(device->cuda_ctx));

  __FAILURE_HANDLE(device_post_update(device->post, camera));

  CUDA_FAILURE_HANDLE(cuCtxPopCurrent(&device->cuda_ctx));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_setup_undersampling(Device* device, const RendererSettings* settings) {
  __CHECK_NULL_ARGUMENT(device);

  DEVICE_ASSERT_AVAILABLE

  bool recurring_enabled;
  __FAILURE_HANDLE(device_output_get_recurring_enabled(device->output, &recurring_enabled));

  uint32_t undersampling;
  if (settings->region_width >= 1.0f && settings->region_height >= 1.0f) {
    undersampling = settings->undersampling;
  }
  else {
    // No undersampling when using render regions.
    undersampling = 0;
  }

  device->undersampling_state = 0;

  device->undersampling_state |= UNDERSAMPLING_FIRST_SAMPLE_MASK;

  // No need to undersample if we don't have recurring outputs.
  // Output requests can never request undersampled outputs.
  if (undersampling > 0 && recurring_enabled) {
    device->undersampling_state |= 0b11 & UNDERSAMPLING_ITERATION_MASK;
    device->undersampling_state |= (undersampling << 2) & UNDERSAMPLING_STAGE_MASK;
  }

  return LUMINARY_SUCCESS;
}

LuminaryResult device_build_adaptive_sampling_stage(Device* device, AdaptiveSampler* sampler, uint8_t stage_id) {
  __CHECK_NULL_ARGUMENT(device);
  __CHECK_NULL_ARGUMENT(sampler);

  DEVICE_ASSERT_AVAILABLE

  CUDA_FAILURE_HANDLE(cuCtxPushCurrent(device->cuda_ctx));

  __DEBUG_ASSERT(device->is_main_device);

  __FAILURE_HANDLE(adaptive_sampler_compute_next_stage(sampler, device, stage_id));

  CUDA_FAILURE_HANDLE(cuCtxPopCurrent(&device->cuda_ctx));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_update_adaptive_sampling(Device* device, AdaptiveSampler* sampler) {
  __CHECK_NULL_ARGUMENT(device);

  DEVICE_ASSERT_AVAILABLE

  CUDA_FAILURE_HANDLE(cuCtxPushCurrent(device->cuda_ctx));

  __DEBUG_ASSERT(device->is_main_device);

  bool buffers_have_changed;
  __FAILURE_HANDLE(device_adaptive_sampler_update(device->adaptive_sampler, device, sampler, &buffers_have_changed));

  if (buffers_have_changed) {
    DeviceAdaptiveSamplerDeviceBufferPtrs ptrs;
    __FAILURE_HANDLE(device_adaptive_sampler_get_device_buffer_ptrs(device->adaptive_sampler, &ptrs));

    DEVICE_UPDATE_CONSTANT_MEMORY(ptrs.stage_sample_counts, DEVICE_PTR(ptrs.stage_sample_counts));
    DEVICE_UPDATE_CONSTANT_MEMORY(ptrs.stage_total_task_counts, DEVICE_PTR(ptrs.stage_total_task_counts));
    DEVICE_UPDATE_CONSTANT_MEMORY(ptrs.adaptive_sampling_block_task_offsets, DEVICE_PTR(ptrs.adaptive_sampling_block_task_offsets));
    DEVICE_UPDATE_CONSTANT_MEMORY(ptrs.tile_last_adaptive_sampling_block_index, DEVICE_PTR(ptrs.tile_last_adaptive_sampling_block_index));
  }

  CUDA_FAILURE_HANDLE(cuCtxPopCurrent(&device->cuda_ctx));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_ensure_adaptive_sampling_stage(Device* device) {
  __CHECK_NULL_ARGUMENT(device);

  DEVICE_ASSERT_AVAILABLE

  CUDA_FAILURE_HANDLE(cuCtxPushCurrent(device->cuda_ctx));

  __DEBUG_ASSERT(device->is_main_device);

  bool buffers_have_changed;
  __FAILURE_HANDLE(device_adaptive_sampler_ensure_stage(device->adaptive_sampler, device, &buffers_have_changed));

  if (buffers_have_changed) {
    DEVICE_UPDATE_CONSTANT_MEMORY(ptrs.tile_last_adaptive_sampling_block_index, DEVICE_PTR(device->adaptive_sampler->subtile_last_blocks));

    __FAILURE_HANDLE(_device_update_constant_memory(device));
  }

  CUDA_FAILURE_HANDLE(cuCtxPopCurrent(&device->cuda_ctx));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_unload_adaptive_sampling(Device* device, AdaptiveSampler* sampler) {
  __CHECK_NULL_ARGUMENT(device);
  __CHECK_NULL_ARGUMENT(sampler);

  DEVICE_ASSERT_AVAILABLE

  CUDA_FAILURE_HANDLE(cuCtxPushCurrent(device->cuda_ctx));

  __DEBUG_ASSERT(device->is_main_device);

  __FAILURE_HANDLE(adaptive_sampler_unload(sampler));

  CUDA_FAILURE_HANDLE(cuCtxPopCurrent(&device->cuda_ctx));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_register_callbacks(Device* device, DeviceRegisterCallbackFuncs funcs, DeviceCommonCallbackData callback_data) {
  __CHECK_NULL_ARGUMENT(device);

  DEVICE_ASSERT_AVAILABLE

  CUDA_FAILURE_HANDLE(cuCtxPushCurrent(device->cuda_ctx));

  __FAILURE_HANDLE(device_output_register_callback(device->output, funcs.output_callback_func, callback_data));
  __FAILURE_HANDLE(device_renderer_register_callback(
    device->renderer, funcs.render_continue_callback_func, funcs.render_finished_callback_func, callback_data));

  CUDA_FAILURE_HANDLE(cuCtxPopCurrent(&device->cuda_ctx));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_set_output_dirty(Device* device) {
  __CHECK_NULL_ARGUMENT(device);

  DEVICE_ASSERT_AVAILABLE

  __FAILURE_HANDLE(device_output_set_output_dirty(device->output));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_update_output_properties(Device* device, LuminaryOutputProperties properties) {
  __CHECK_NULL_ARGUMENT(device);

  DEVICE_ASSERT_AVAILABLE

  CUDA_FAILURE_HANDLE(cuCtxPushCurrent(device->cuda_ctx));

  __FAILURE_HANDLE(device_output_set_properties(device->output, properties));

  CUDA_FAILURE_HANDLE(cuCtxPopCurrent(&device->cuda_ctx));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_update_output_camera_params(Device* device, const Camera* camera) {
  __CHECK_NULL_ARGUMENT(device);

  DEVICE_ASSERT_AVAILABLE

  CUDA_FAILURE_HANDLE(cuCtxPushCurrent(device->cuda_ctx));

  __FAILURE_HANDLE(device_output_set_camera_params(device->output, camera));

  CUDA_FAILURE_HANDLE(cuCtxPopCurrent(&device->cuda_ctx));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_add_output_request(Device* device, OutputRequestProperties properties) {
  __CHECK_NULL_ARGUMENT(device);

  DEVICE_ASSERT_AVAILABLE

  CUDA_FAILURE_HANDLE(cuCtxPushCurrent(device->cuda_ctx));

  __FAILURE_HANDLE(device_output_add_request(device->output, properties));

  CUDA_FAILURE_HANDLE(cuCtxPopCurrent(&device->cuda_ctx));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_start_render(Device* device, DeviceRendererQueueArgs* args) {
  __CHECK_NULL_ARGUMENT(device);

  DEVICE_ASSERT_AVAILABLE

  CUDA_FAILURE_HANDLE(cuCtxPushCurrent(device->cuda_ctx));

  __FAILURE_HANDLE(device_staging_manager_execute(device->staging_manager));

  device->gbuffer_meta_state = GBUFFER_META_STATE_NOT_READY;

  __FAILURE_HANDLE(device_sync_constant_memory(device));
  __FAILURE_HANDLE(device_renderer_init_new_render(device->renderer, args));
  __FAILURE_HANDLE(device_renderer_continue(device->renderer, device));

  CUDA_FAILURE_HANDLE(cuCtxPopCurrent(&device->cuda_ctx));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_validate_render_callback(Device* device, DeviceRenderCallbackData* callback_data, bool* is_valid) {
  __CHECK_NULL_ARGUMENT(device);
  __CHECK_NULL_ARGUMENT(callback_data);

  DEVICE_ASSERT_AVAILABLE

  *is_valid = false;

  if (device->state != DEVICE_STATE_ENABLED)
    return LUMINARY_SUCCESS;

  bool continuation_is_valid;
  __FAILURE_HANDLE(device_renderer_handle_callback(device->renderer, callback_data, &continuation_is_valid));

  if (!continuation_is_valid)
    return LUMINARY_SUCCESS;

  *is_valid = true;

  return LUMINARY_SUCCESS;
}

LuminaryResult device_finish_render_iteration(Device* device, AdaptiveSampler* sampler, DeviceRenderCallbackData* callback_data) {
  __CHECK_NULL_ARGUMENT(device);
  __CHECK_NULL_ARGUMENT(callback_data);

  DEVICE_ASSERT_AVAILABLE

  uint32_t renderer_status;
  __FAILURE_HANDLE(device_renderer_get_status(device->renderer, &renderer_status));

  // We can only finish the render iteration if we are next up starting a new sample.
  if ((renderer_status & DEVICE_RENDERER_STATUS_FLAG_FINISHED) == 0)
    return LUMINARY_SUCCESS;

  CUDA_FAILURE_HANDLE(cuCtxPushCurrent(device->cuda_ctx));

  uint32_t new_undersampling_state;
  __FAILURE_HANDLE(_device_update_get_next_undersampling_state(device, &new_undersampling_state));

  __FAILURE_HANDLE(device_renderer_finish_iteration(device->renderer, new_undersampling_state != 0));

  if (device->is_main_device) {
    bool does_output;
    __FAILURE_HANDLE(device_output_will_output(device->output, device->renderer, &does_output));

    if (does_output) {
      __FAILURE_HANDLE(device_post_apply(device->post, device));
      __FAILURE_HANDLE(device_output_generate_output(device->output, device, callback_data->render_event_id));
    }
  }

  __FAILURE_HANDLE(device_renderer_allocate_sample(device->renderer, sampler));

  device->undersampling_state = new_undersampling_state;

  CUDA_FAILURE_HANDLE(cuCtxPopCurrent(&device->cuda_ctx));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_continue_render(Device* device) {
  __CHECK_NULL_ARGUMENT(device);

  DEVICE_ASSERT_AVAILABLE

  CUDA_FAILURE_HANDLE(cuCtxPushCurrent(device->cuda_ctx));

  __FAILURE_HANDLE(device_renderer_continue(device->renderer, device));

  CUDA_FAILURE_HANDLE(cuCtxPopCurrent(&device->cuda_ctx));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_update_render_time(Device* device, DeviceRenderCallbackData* callback_data) {
  __CHECK_NULL_ARGUMENT(device);
  __CHECK_NULL_ARGUMENT(callback_data);

  DEVICE_ASSERT_AVAILABLE

  CUDA_FAILURE_HANDLE(cuCtxPushCurrent(device->cuda_ctx));

  __FAILURE_HANDLE(device_renderer_update_render_time(device->renderer, callback_data->render_event_id));

  CUDA_FAILURE_HANDLE(cuCtxPopCurrent(&device->cuda_ctx));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_handle_result_sharing(Device* device, DeviceResultInterface* interface) {
  __CHECK_NULL_ARGUMENT(device);
  __CHECK_NULL_ARGUMENT(interface);

  DEVICE_ASSERT_AVAILABLE

  uint32_t renderer_status;
  __FAILURE_HANDLE(device_renderer_get_status(device->renderer, &renderer_status));

  // Only gather results between samples.
  if ((renderer_status & DEVICE_RENDERER_STATUS_FLAG_IN_PROGRESS) != 0)
    return LUMINARY_SUCCESS;

  CUDA_FAILURE_HANDLE(cuCtxPushCurrent(device->cuda_ctx));

  if (device->is_main_device) {
    __FAILURE_HANDLE(device_result_interface_gather_results(interface, device));
  }
  else {
    __FAILURE_HANDLE(device_result_interface_queue_result(interface, device));
  }

  CUDA_FAILURE_HANDLE(cuCtxPopCurrent(&device->cuda_ctx));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_get_recommended_sample_queue_counts(Device* device, uint32_t* recommended_count) {
  __CHECK_NULL_ARGUMENT(device);
  __CHECK_NULL_ARGUMENT(recommended_count);

  uint32_t tile_count;
  __FAILURE_HANDLE(device_renderer_get_tile_count(device->renderer, device, 0, &tile_count));

  *recommended_count = (2048 + tile_count - 1) / tile_count;

  return LUMINARY_SUCCESS;
}

LuminaryResult device_set_abort(Device* device) {
  __CHECK_NULL_ARGUMENT(device);

  DEVICE_ASSERT_AVAILABLE

  CUDA_FAILURE_HANDLE(cuCtxPushCurrent(device->cuda_ctx));

  __FAILURE_HANDLE(device_abort_set(device->abort, device, true));

  device->state_abort = true;

  CUDA_FAILURE_HANDLE(cuCtxPopCurrent(&device->cuda_ctx));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_unset_abort(Device* device) {
  __CHECK_NULL_ARGUMENT(device);

  DEVICE_ASSERT_AVAILABLE

  CUDA_FAILURE_HANDLE(cuCtxPushCurrent(device->cuda_ctx));

  __FAILURE_HANDLE(device_abort_set(device->abort, device, false));

  device->state_abort = false;

  CUDA_FAILURE_HANDLE(cuCtxPopCurrent(&device->cuda_ctx));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_query_gbuffer_meta(Device* device) {
  __CHECK_NULL_ARGUMENT(device);

  DEVICE_ASSERT_AVAILABLE

  bool recurring_outputs_enabled;
  __FAILURE_HANDLE(device_output_get_recurring_enabled(device->output, &recurring_outputs_enabled));

  // GBuffer meta data is only available for recurring outputs (i.e. interactive rendering)
  if (recurring_outputs_enabled == false)
    return LUMINARY_SUCCESS;

  CUDA_FAILURE_HANDLE(cuCtxPushCurrent(device->cuda_ctx));

  const uint16_t width  = device->constant_memory->settings.width >> (device->constant_memory->settings.supersampling + 1);
  const uint16_t height = device->constant_memory->settings.height >> (device->constant_memory->settings.supersampling + 1);

  __FAILURE_HANDLE(device_download(
    device->gbuffer_meta_dst, device->work_buffers->gbuffer_meta, 0, sizeof(GBufferMetaData) * width * height, device->stream_main));

  CUDA_FAILURE_HANDLE(cuEventRecord(device->event_queue_gbuffer_meta, device->stream_main));
  device->gbuffer_meta_state = GBUFFER_META_STATE_QUEUED;

  CUDA_FAILURE_HANDLE(cuCtxPopCurrent(&device->cuda_ctx));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_get_gbuffer_meta(Device* device, uint16_t x, uint16_t y, GBufferMetaData* data) {
  __CHECK_NULL_ARGUMENT(device);

  DEVICE_ASSERT_AVAILABLE

  x = x >> 1;
  y = y >> 1;

  const uint16_t width  = device->constant_memory->settings.width >> (device->constant_memory->settings.supersampling + 1);
  const uint16_t height = device->constant_memory->settings.height >> (device->constant_memory->settings.supersampling + 1);

  bool data_available = x < width && y < height && device->gbuffer_meta_state != GBUFFER_META_STATE_NOT_READY;

  if (data_available) {
    if (device->gbuffer_meta_state == GBUFFER_META_STATE_QUEUED) {
      CUresult result = cuEventQuery(device->event_queue_gbuffer_meta);

      if (result == CUDA_ERROR_NOT_READY) {
        data_available = false;
      }
      else {
        CUDA_FAILURE_HANDLE(result);
        device->gbuffer_meta_state = GBUFFER_META_STATE_READY;
      }
    }
  }

  if (data_available == false) {
    data->depth       = DEPTH_INVALID;
    data->instance_id = 0xFFFFFFFF;
    data->material_id = MATERIAL_ID_INVALID;

    return LUMINARY_SUCCESS;
  }

  memcpy(data, device->gbuffer_meta_dst + x + y * width, sizeof(GBufferMetaData));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_destroy(Device** device) {
  __CHECK_NULL_ARGUMENT(device);
  __CHECK_NULL_ARGUMENT(*device);

  CUDA_FAILURE_HANDLE(cuCtxPushCurrent((*device)->cuda_ctx));
  CUDA_FAILURE_HANDLE(cuCtxSynchronize());

  __FAILURE_HANDLE(_device_free_buffers(*device));

  uint32_t num_meshes;
  __FAILURE_HANDLE(array_get_num_elements((*device)->meshes, &num_meshes));

  for (uint32_t mesh_id = 0; mesh_id < num_meshes; mesh_id++) {
    __FAILURE_HANDLE(device_mesh_destroy(&(*device)->meshes[mesh_id]));
    __FAILURE_HANDLE(omm_destroy(&(*device)->omms[mesh_id]));
  }

  __FAILURE_HANDLE(array_destroy(&(*device)->meshes));
  __FAILURE_HANDLE(array_destroy(&(*device)->omms));

  __FAILURE_HANDLE(device_abort_destroy(&(*device)->abort));
  __FAILURE_HANDLE(device_output_destroy(&(*device)->output));
  __FAILURE_HANDLE(device_renderer_destroy(&(*device)->renderer));
  __FAILURE_HANDLE(device_work_buffers_destroy(&(*device)->work_buffers));
  __FAILURE_HANDLE(device_embedded_data_destroy(&(*device)->embedded_data));

  __FAILURE_HANDLE(device_sky_lut_destroy(&(*device)->sky_lut));
  __FAILURE_HANDLE(device_sky_hdri_destroy(&(*device)->sky_hdri));
  __FAILURE_HANDLE(device_sky_stars_destroy(&(*device)->sky_stars));
  __FAILURE_HANDLE(device_bsdf_lut_destroy(&(*device)->bsdf_lut));
  __FAILURE_HANDLE(device_physical_camera_destroy(&(*device)->physical_camera));
  __FAILURE_HANDLE(device_cloud_noise_destroy(&(*device)->cloud_noise));
  __FAILURE_HANDLE(device_particles_handle_destroy(&(*device)->particles_handle));
  __FAILURE_HANDLE(device_adaptive_sampler_destroy(&(*device)->adaptive_sampler));

  __FAILURE_HANDLE(optix_bvh_instance_cache_destroy(&(*device)->optix_instance_cache));
  __FAILURE_HANDLE(optix_bvh_destroy(&(*device)->optix_bvh_ias));
  __FAILURE_HANDLE(optix_bvh_destroy(&(*device)->optix_bvh_light));

  __FAILURE_HANDLE(device_post_destroy(&(*device)->post));

  uint32_t num_textures;
  __FAILURE_HANDLE(array_get_num_elements((*device)->textures, &num_textures));

  for (uint32_t texture_id = 0; texture_id < num_textures; texture_id++) {
    __FAILURE_HANDLE(device_texture_destroy(&(*device)->textures[texture_id]));
  }

  __FAILURE_HANDLE(array_destroy(&(*device)->textures));

  for (uint32_t kernel_id = 0; kernel_id < CUDA_KERNEL_TYPE_COUNT; kernel_id++) {
    __FAILURE_HANDLE(kernel_destroy(&(*device)->cuda_kernels[kernel_id]));
  }

  for (uint32_t kernel_id = 0; kernel_id < OPTIX_KERNEL_TYPE_COUNT; kernel_id++) {
    __FAILURE_HANDLE(optix_kernel_destroy(&(*device)->optix_kernels[kernel_id]));
  }

  __FAILURE_HANDLE(device_free_staging(&(*device)->constant_memory));

  if ((*device)->gbuffer_meta_dst != (GBufferMetaData*) 0) {
    __FAILURE_HANDLE(device_free_staging(&(*device)->gbuffer_meta_dst));
  }

  __FAILURE_HANDLE(device_staging_manager_destroy(&(*device)->staging_manager));

  CUDA_FAILURE_HANDLE(cuStreamDestroy((*device)->stream_main));
  CUDA_FAILURE_HANDLE(cuStreamDestroy((*device)->stream_output));
  CUDA_FAILURE_HANDLE(cuStreamDestroy((*device)->stream_abort));
  CUDA_FAILURE_HANDLE(cuStreamDestroy((*device)->stream_callbacks));

  CUDA_FAILURE_HANDLE(cuEventDestroy((*device)->event_queue_render));
  CUDA_FAILURE_HANDLE(cuEventDestroy((*device)->event_queue_gbuffer_meta));

  OPTIX_FAILURE_HANDLE(optixDeviceContextDestroy((*device)->optix_ctx));
  CUDA_FAILURE_HANDLE(cuCtxDestroy((*device)->cuda_ctx));

  __FAILURE_HANDLE(host_free(device));

  return LUMINARY_SUCCESS;
}
