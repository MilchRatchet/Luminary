#include "device.h"

#include <optix_function_table_definition.h>

#include "camera.h"
#include "ceb.h"
#include "cloud.h"
#include "device_light.h"
#include "device_memory.h"
#include "device_texture.h"
#include "device_utils.h"
#include "fog.h"
#include "host/png.h"
#include "internal_error.h"
#include "particles.h"
#include "sky.h"
#include "toy.h"

static const DeviceConstantMemoryMember device_scene_entity_to_const_memory_member[] = {
  DEVICE_CONSTANT_MEMORY_MEMBER_SETTINGS,   // SCENE_ENTITY_SETTINGS
  DEVICE_CONSTANT_MEMORY_MEMBER_CAMERA,     // SCENE_ENTITY_CAMERA
  DEVICE_CONSTANT_MEMORY_MEMBER_OCEAN,      // SCENE_ENTITY_OCEAN
  DEVICE_CONSTANT_MEMORY_MEMBER_SKY,        // SCENE_ENTITY_SKY
  DEVICE_CONSTANT_MEMORY_MEMBER_CLOUD,      // SCENE_ENTITY_CLOUD
  DEVICE_CONSTANT_MEMORY_MEMBER_FOG,        // SCENE_ENTITY_FOG
  DEVICE_CONSTANT_MEMORY_MEMBER_PARTICLES,  // SCENE_ENTITY_PARTICLES
  DEVICE_CONSTANT_MEMORY_MEMBER_TOY         // SCENE_ENTITY_TOY
};
LUM_STATIC_SIZE_ASSERT(device_scene_entity_to_const_memory_member, sizeof(DeviceConstantMemoryMember) * SCENE_ENTITY_GLOBAL_COUNT);

static const size_t device_cuda_const_memory_offsets[] = {
  offsetof(DeviceConstantMemory, ptrs),                          // DEVICE_CONSTANT_MEMORY_MEMBER_PTRS
  offsetof(DeviceConstantMemory, settings),                      // DEVICE_CONSTANT_MEMORY_MEMBER_SETTINGS
  offsetof(DeviceConstantMemory, camera),                        // DEVICE_CONSTANT_MEMORY_MEMBER_CAMERA
  offsetof(DeviceConstantMemory, ocean),                         // DEVICE_CONSTANT_MEMORY_MEMBER_OCEAN
  offsetof(DeviceConstantMemory, sky),                           // DEVICE_CONSTANT_MEMORY_MEMBER_SKY
  offsetof(DeviceConstantMemory, cloud),                         // DEVICE_CONSTANT_MEMORY_MEMBER_CLOUD
  offsetof(DeviceConstantMemory, fog),                           // DEVICE_CONSTANT_MEMORY_MEMBER_FOG
  offsetof(DeviceConstantMemory, particles),                     // DEVICE_CONSTANT_MEMORY_MEMBER_PARTICLES
  offsetof(DeviceConstantMemory, toy),                           // DEVICE_CONSTANT_MEMORY_MEMBER_TOY
  offsetof(DeviceConstantMemory, pixels_per_thread),             // DEVICE_CONSTANT_MEMORY_MEMBER_TASK_META
  offsetof(DeviceConstantMemory, optix_bvh),                     // DEVICE_CONSTANT_MEMORY_MEMBER_OPTIX_BVH
  offsetof(DeviceConstantMemory, non_instanced_triangle_count),  // DEVICE_CONSTANT_MEMORY_MEMBER_TRI_COUNT
  offsetof(DeviceConstantMemory, moon_albedo_tex),               // DEVICE_CONSTANT_MEMORY_MEMBER_MOON_TEX
  offsetof(DeviceConstantMemory, user_selected_x),               // DEVICE_CONSTANT_MEMORY_MEMBER_DYNAMIC
  SIZE_MAX                                                       // DEVICE_CONSTANT_MEMORY_MEMBER_COUNT
};
LUM_STATIC_SIZE_ASSERT(device_cuda_const_memory_offsets, sizeof(size_t) * (DEVICE_CONSTANT_MEMORY_MEMBER_COUNT + 1));

static const size_t device_cuda_const_memory_sizes[] = {
  sizeof(DevicePointers),                      // DEVICE_CONSTANT_MEMORY_MEMBER_PTRS
  sizeof(DeviceRendererSettings),              // DEVICE_CONSTANT_MEMORY_MEMBER_SETTINGS
  sizeof(DeviceCamera),                        // DEVICE_CONSTANT_MEMORY_MEMBER_CAMERA
  sizeof(DeviceOcean),                         // DEVICE_CONSTANT_MEMORY_MEMBER_OCEAN
  sizeof(DeviceSky),                           // DEVICE_CONSTANT_MEMORY_MEMBER_SKY
  sizeof(DeviceCloud),                         // DEVICE_CONSTANT_MEMORY_MEMBER_CLOUD
  sizeof(DeviceFog),                           // DEVICE_CONSTANT_MEMORY_MEMBER_FOG
  sizeof(DeviceParticles),                     // DEVICE_CONSTANT_MEMORY_MEMBER_PARTICLES
  sizeof(DeviceToy),                           // DEVICE_CONSTANT_MEMORY_MEMBER_TOY
  sizeof(uint32_t) * 2,                        // DEVICE_CONSTANT_MEMORY_MEMBER_TASK_META
  sizeof(OptixTraversableHandle) * 4,          // DEVICE_CONSTANT_MEMORY_MEMBER_OPTIX_BVH
  sizeof(uint32_t),                            // DEVICE_CONSTANT_MEMORY_MEMBER_TRI_COUNT
  sizeof(DeviceTextureObject) * 2,             // DEVICE_CONSTANT_MEMORY_MEMBER_MOON_TEX
  sizeof(uint16_t) * 2 + sizeof(uint32_t) * 3  // DEVICE_CONSTANT_MEMORY_MEMBER_DYNAMIC
};
LUM_STATIC_SIZE_ASSERT(device_cuda_const_memory_sizes, sizeof(size_t) * DEVICE_CONSTANT_MEMORY_MEMBER_COUNT);

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

  OptixResult optix_result = optixInit();

  if (optix_result != OPTIX_SUCCESS) {
    crash_message("Failed to initialize OptiX.");
  }

  _device_memory_init();
}

void _device_shutdown(void) {
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

static char* _device_arch_enum_to_string(const DeviceArch arch) {
  switch (arch) {
    default:
    case DEVICE_ARCH_UNKNOWN:
      return "Unknown";
    case DEVICE_ARCH_PASCAL:
      return "Pascal";
    case DEVICE_ARCH_TURING:
      return "Turing";
    case DEVICE_ARCH_AMPERE:
      return "Ampere";
    case DEVICE_ARCH_ADA:
      return "Ada";
    case DEVICE_ARCH_VOLTA:
      return "Volta";
    case DEVICE_ARCH_HOPPER:
      return "Hopper";
  }
}

static LuminaryResult _device_get_properties(DeviceProperties* props, Device* device) {
  __CHECK_NULL_ARGUMENT(props);

  int major;
  CUDA_FAILURE_HANDLE(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device->cuda_device));

  int minor;
  CUDA_FAILURE_HANDLE(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device->cuda_device));

  CUDA_FAILURE_HANDLE(cuDeviceGetName(props->name, 256, device->cuda_device));

  CUDA_FAILURE_HANDLE(cuDeviceTotalMem(&props->memory_size, device->cuda_device));

  switch (major) {
    case 6: {
      if (minor == 0 || minor == 1 || minor == 2) {
        props->arch            = DEVICE_ARCH_PASCAL;
        props->rt_core_version = 0;
      }
      else {
        props->arch            = DEVICE_ARCH_UNKNOWN;
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
      else {
        props->arch            = DEVICE_ARCH_UNKNOWN;
        props->rt_core_version = 0;
      }
    } break;
    case 8: {
      if (minor == 0) {
        // GA100 has no RT cores
        props->arch            = DEVICE_ARCH_AMPERE;
        props->rt_core_version = 0;
      }
      else if (minor == 6 || minor == 7) {
        props->arch            = DEVICE_ARCH_AMPERE;
        props->rt_core_version = 2;
      }
      else if (minor == 9) {
        props->arch            = DEVICE_ARCH_ADA;
        props->rt_core_version = 3;
      }
      else {
        props->arch            = DEVICE_ARCH_UNKNOWN;
        props->rt_core_version = 0;
      }
    } break;
    case 9: {
      if (minor == 0) {
        props->arch            = DEVICE_ARCH_HOPPER;
        props->rt_core_version = 0;
      }
      else {
        props->arch            = DEVICE_ARCH_UNKNOWN;
        props->rt_core_version = 0;
      }
    } break;
    case 10: {
      if (minor == 0) {
        props->arch            = DEVICE_ARCH_BLACKWELL;
        props->rt_core_version = 0;
      }
      else {
        props->arch            = DEVICE_ARCH_UNKNOWN;
        props->rt_core_version = 0;
      }
    } break;
    default:
      props->arch            = DEVICE_ARCH_UNKNOWN;
      props->rt_core_version = 0;
      break;
  }

  if (props->arch == DEVICE_ARCH_UNKNOWN) {
    warn_message(
      "Luminary failed to identify architecture of CUDA compute capability %d.%d. Some features may not be working.", major, minor);
  }

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
    size   = sizeof(DeviceConstantMemory);
  }
  else {
    offset = device_cuda_const_memory_offsets[device->constant_memory_dirty.member];
    size   = device_cuda_const_memory_sizes[device->constant_memory_dirty.member];
  }

  CUDA_FAILURE_HANDLE(
    cuMemcpyHtoDAsync_v2(device->cuda_device_const_memory + offset, device->constant_memory + offset, size, device->stream_main));

  return LUMINARY_SUCCESS;
}

////////////////////////////////////////////////////////////////////
// Embedded data
////////////////////////////////////////////////////////////////////

static LuminaryResult _device_load_moon_textures(Device* device) {
  __CHECK_NULL_ARGUMENT(device);

  uint64_t info = 0;

  void* moon_albedo_data;
  int64_t moon_albedo_data_length;
  ceb_access("moon_albedo.png", &moon_albedo_data, &moon_albedo_data_length, &info);

  if (info) {
    __RETURN_ERROR(LUMINARY_ERROR_NOT_IMPLEMENTED, "Failed to load moon_albedo texture. Luminary was not compiled correctly.");
  }

  void* moon_normal_data;
  int64_t moon_normal_data_length;
  ceb_access("moon_normal.png", &moon_normal_data, &moon_normal_data_length, &info);

  if (info) {
    __RETURN_ERROR(LUMINARY_ERROR_NOT_IMPLEMENTED, "Failed to load moon_normal texture. Luminary was not compiled correctly.");
  }

  Texture* moon_albedo_tex;
  __FAILURE_HANDLE(png_load(&moon_albedo_tex, moon_albedo_data, moon_albedo_data_length, "moon_albedo.png"));

  __FAILURE_HANDLE(device_texture_create(&device->moon_albedo_tex, moon_albedo_tex, device->stream_main));

  __FAILURE_HANDLE(texture_destroy(&moon_albedo_tex));

  Texture* moon_normal_tex;
  __FAILURE_HANDLE(png_load(&moon_normal_tex, moon_normal_data, moon_normal_data_length, "moon_normal.png"));

  __FAILURE_HANDLE(device_texture_create(&device->moon_normal_tex, moon_normal_tex, device->stream_main));

  __FAILURE_HANDLE(texture_destroy(&moon_normal_tex));

  __FAILURE_HANDLE(device_struct_texture_object_convert(device->moon_albedo_tex, &device->constant_memory->moon_albedo_tex));
  __FAILURE_HANDLE(device_struct_texture_object_convert(device->moon_normal_tex, &device->constant_memory->moon_normal_tex));

  __FAILURE_HANDLE(_device_set_constant_memory_dirty(device, DEVICE_CONSTANT_MEMORY_MEMBER_MOON_TEX));

  return LUMINARY_SUCCESS;
}

static LuminaryResult _device_load_bluenoise_texture(Device* device) {
  __CHECK_NULL_ARGUMENT(device);

  uint64_t info = 0;

  void* bluenoise_1D_data;
  int64_t bluenoise_1D_data_length;
  ceb_access("bluenoise_1D.bin", &bluenoise_1D_data, &bluenoise_1D_data_length, &info);

  if (info) {
    __RETURN_ERROR(LUMINARY_ERROR_NOT_IMPLEMENTED, "Failed to load bluenoise_1D texture. Luminary was not compiled correctly.");
  }

  void* bluenoise_2D_data;
  int64_t bluenoise_2D_data_length;
  ceb_access("bluenoise_2D.bin", &bluenoise_2D_data, &bluenoise_2D_data_length, &info);

  if (info) {
    __RETURN_ERROR(LUMINARY_ERROR_NOT_IMPLEMENTED, "Failed to load bluenoise_2D texture. Luminary was not compiled correctly.");
  }

  __FAILURE_HANDLE(device_malloc(&device->buffers.bluenoise_1D, bluenoise_1D_data_length));
  __FAILURE_HANDLE(
    device_upload((void*) device->buffers.bluenoise_1D, bluenoise_1D_data, 0, bluenoise_1D_data_length, device->stream_main));

  __FAILURE_HANDLE(device_malloc(&device->buffers.bluenoise_2D, bluenoise_2D_data_length));
  __FAILURE_HANDLE(
    device_upload((void*) device->buffers.bluenoise_2D, bluenoise_2D_data, 0, bluenoise_2D_data_length, device->stream_main));

  DEVICE_UPDATE_CONSTANT_MEMORY(ptrs.bluenoise_1D, DEVICE_PTR(device->buffers.bluenoise_1D));
  DEVICE_UPDATE_CONSTANT_MEMORY(ptrs.bluenoise_2D, DEVICE_PTR(device->buffers.bluenoise_2D));

  return LUMINARY_SUCCESS;
}

static LuminaryResult _device_load_light_bridge_lut(Device* device) {
  uint64_t info = 0;

  void* lut_data;
  int64_t lut_length;
  ceb_access("bridge_lut.bin", &lut_data, &lut_length, &info);

  if (info) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Failed to load bridge_lut texture.");
  }

  __FAILURE_HANDLE(device_malloc(&device->buffers.bridge_lut, lut_length));
  __FAILURE_HANDLE(device_upload((void*) device->buffers.bridge_lut, lut_data, 0, lut_length, device->stream_main));

  DEVICE_UPDATE_CONSTANT_MEMORY(ptrs.bridge_lut, DEVICE_PTR(device->buffers.bridge_lut));

  return LUMINARY_SUCCESS;
}

static LuminaryResult _device_free_embedded_data(Device* device) {
  __CHECK_NULL_ARGUMENT(device);

  __FAILURE_HANDLE(device_texture_destroy(&device->moon_albedo_tex));
  __FAILURE_HANDLE(device_texture_destroy(&device->moon_normal_tex));

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

#define __DEVICE_BUFFER_ALLOCATE(buffer, size)                                      \
  {                                                                                 \
    __DEVICE_BUFFER_FREE(buffer);                                                   \
    __FAILURE_HANDLE(device_malloc(&device->buffers.buffer, size));                 \
    DEVICE_UPDATE_CONSTANT_MEMORY(ptrs.buffer, DEVICE_PTR(device->buffers.buffer)); \
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

  const uint32_t internal_pixel_count = device->constant_memory->settings.width * device->constant_memory->settings.height;

  const uint32_t thread_count      = THREADS_PER_BLOCK * BLOCKS_PER_GRID;
  const uint32_t pixels_per_thread = 1 + ((internal_pixel_count + thread_count - 1) / thread_count);
  const uint32_t max_task_count    = pixels_per_thread * thread_count;

  __DEVICE_BUFFER_ALLOCATE(trace_tasks, sizeof(TraceTask) * max_task_count);
  __DEVICE_BUFFER_ALLOCATE(aux_data, sizeof(ShadingTaskAuxData) * max_task_count);
  __DEVICE_BUFFER_ALLOCATE(trace_counts, sizeof(uint16_t) * thread_count);
  __DEVICE_BUFFER_ALLOCATE(trace_results, sizeof(TraceResult) * max_task_count);
  __DEVICE_BUFFER_ALLOCATE(task_counts, sizeof(uint16_t) * 7 * thread_count);
  __DEVICE_BUFFER_ALLOCATE(task_offsets, sizeof(uint16_t) * 6 * thread_count);
  __DEVICE_BUFFER_ALLOCATE(ior_stack, sizeof(uint32_t) * internal_pixel_count);
  __DEVICE_BUFFER_ALLOCATE(frame_variance, sizeof(float) * internal_pixel_count);
  __DEVICE_BUFFER_ALLOCATE(frame_accumulate, sizeof(RGBF) * internal_pixel_count);
  __DEVICE_BUFFER_ALLOCATE(frame_direct_buffer, sizeof(RGBF) * internal_pixel_count);
  __DEVICE_BUFFER_ALLOCATE(frame_direct_accumulate, sizeof(RGBF) * internal_pixel_count);
  __DEVICE_BUFFER_ALLOCATE(frame_indirect_buffer, sizeof(RGBF) * internal_pixel_count);
  __DEVICE_BUFFER_ALLOCATE(frame_indirect_accumulate, sizeof(RGBF) * internal_pixel_count);
  __DEVICE_BUFFER_ALLOCATE(frame_post, sizeof(RGBF) * internal_pixel_count);
  __DEVICE_BUFFER_ALLOCATE(frame_final, sizeof(RGBF) * (internal_pixel_count >> 2));
  __DEVICE_BUFFER_ALLOCATE(records, sizeof(RGBF) * internal_pixel_count);
  __DEVICE_BUFFER_ALLOCATE(hit_id_history, sizeof(uint32_t) * internal_pixel_count);

  __FAILURE_HANDLE(device_memset(device->buffers.hit_id_history, 0, 0, sizeof(uint32_t) * internal_pixel_count, device->stream_main));

  DEVICE_UPDATE_CONSTANT_MEMORY(max_task_count, max_task_count);
  DEVICE_UPDATE_CONSTANT_MEMORY(pixels_per_thread, pixels_per_thread);

  return LUMINARY_SUCCESS;
}

static LuminaryResult _device_free_buffers(Device* device) {
  __CHECK_NULL_ARGUMENT(device);

  __DEVICE_BUFFER_FREE(trace_tasks);
  __DEVICE_BUFFER_FREE(aux_data);
  __DEVICE_BUFFER_FREE(trace_counts);
  __DEVICE_BUFFER_FREE(trace_results);
  __DEVICE_BUFFER_FREE(task_counts);
  __DEVICE_BUFFER_FREE(task_offsets);
  __DEVICE_BUFFER_FREE(ior_stack);
  __DEVICE_BUFFER_FREE(frame_variance);
  __DEVICE_BUFFER_FREE(frame_accumulate);
  __DEVICE_BUFFER_FREE(frame_direct_buffer);
  __DEVICE_BUFFER_FREE(frame_direct_accumulate);
  __DEVICE_BUFFER_FREE(frame_indirect_buffer);
  __DEVICE_BUFFER_FREE(frame_indirect_accumulate);
  __DEVICE_BUFFER_FREE(frame_post);
  __DEVICE_BUFFER_FREE(frame_final);
  __DEVICE_BUFFER_FREE(records);
  __DEVICE_BUFFER_FREE(buffer_8bit);
  __DEVICE_BUFFER_FREE(hit_id_history);
  __DEVICE_BUFFER_FREE(textures);
  __DEVICE_BUFFER_FREE(cloud_noise);
  __DEVICE_BUFFER_FREE(sky_ms_luts);
  __DEVICE_BUFFER_FREE(sky_tm_luts);
  __DEVICE_BUFFER_FREE(sky_hdri_luts);
  __DEVICE_BUFFER_FREE(bsdf_energy_lut);
  __DEVICE_BUFFER_FREE(bluenoise_1D);
  __DEVICE_BUFFER_FREE(bluenoise_2D);
  __DEVICE_BUFFER_FREE(bridge_lut);
  __DEVICE_BUFFER_FREE(materials);
  __DEVICE_BUFFER_FREE(triangles);
  __DEVICE_BUFFER_FREE(instances);
  __DEVICE_BUFFER_FREE(instance_transforms);
  __DEVICE_BUFFER_FREE(light_instance_map);
  __DEVICE_BUFFER_FREE(bottom_level_light_trees);
  __DEVICE_BUFFER_FREE(bottom_level_light_paths);
  __DEVICE_BUFFER_FREE(top_level_light_tree);
  __DEVICE_BUFFER_FREE(top_level_light_paths);
  __DEVICE_BUFFER_FREE(particle_quads);
  __DEVICE_BUFFER_FREE(stars);
  __DEVICE_BUFFER_FREE(stars_offsets);

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

  __FAILURE_HANDLE(_device_reset_constant_memory_dirty(device));

  // Device has no samples queued by default.
  __FAILURE_HANDLE(sample_count_reset(&device->sample_count, 0));

  CUDA_FAILURE_HANDLE(cuDeviceGet(&device->cuda_device, device->index));

  __FAILURE_HANDLE(_device_get_properties(&device->properties, device));

  ////////////////////////////////////////////////////////////////////
  // CUDA context creation
  ////////////////////////////////////////////////////////////////////

  CUDA_FAILURE_HANDLE(cuCtxCreate(&device->cuda_ctx, 0, device->cuda_device));

  CUDA_FAILURE_HANDLE(cuCtxPushCurrent(device->cuda_ctx));

  ////////////////////////////////////////////////////////////////////
  // OptiX context creation
  ////////////////////////////////////////////////////////////////////

  OptixDeviceContextOptions optix_device_context_options;
  memset(&optix_device_context_options, 0, sizeof(OptixDeviceContextOptions));

#ifdef OPTIX_VALIDATION
  optix_device_context_options.logCallbackData     = (void*) 0;
  optix_device_context_options.logCallbackFunction = _device_optix_log_callback;
  optix_device_context_options.logCallbackLevel    = 3;
  optix_device_context_options.validationMode      = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
#endif

  OPTIX_FAILURE_HANDLE(optixDeviceContextCreate((CUcontext) 0, &optix_device_context_options, &device->optix_ctx));

  ////////////////////////////////////////////////////////////////////
  // Stream creation
  ////////////////////////////////////////////////////////////////////

  CUDA_FAILURE_HANDLE(cuStreamCreate(&device->stream_main, CU_STREAM_NON_BLOCKING));
  CUDA_FAILURE_HANDLE(cuStreamCreate(&device->stream_secondary, CU_STREAM_NON_BLOCKING));

  ////////////////////////////////////////////////////////////////////
  // Constant memory initialization
  ////////////////////////////////////////////////////////////////////

  __FAILURE_HANDLE(device_malloc_staging(&device->constant_memory, sizeof(DeviceConstantMemory), true));
  memset(device->constant_memory, 0, sizeof(DeviceConstantMemory));

  __FAILURE_HANDLE(device_staging_manager_create(&device->staging_manager, device));

  __FAILURE_HANDLE(array_create(&device->textures, sizeof(DeviceTexture*), 16));

  ////////////////////////////////////////////////////////////////////
  // Optix data
  ////////////////////////////////////////////////////////////////////

  __FAILURE_HANDLE(array_create(&device->optix_mesh_bvhs, sizeof(OptixBVH*), 4));
  __FAILURE_HANDLE(optix_bvh_instance_cache_create(&device->optix_instance_cache, device));
  __FAILURE_HANDLE(optix_bvh_create(&device->optix_bvh_ias));

  CUDA_FAILURE_HANDLE(cuCtxPopCurrent(&device->cuda_ctx));

  *_device = device;

  return LUMINARY_SUCCESS;
}

LuminaryResult device_compile_kernels(Device* device, CUlibrary library) {
  __CHECK_NULL_ARGUMENT(device);

  CUDA_FAILURE_HANDLE(cuCtxPushCurrent(device->cuda_ctx));

  for (uint32_t kernel_id = 0; kernel_id < CUDA_KERNEL_TYPE_COUNT; kernel_id++) {
    __FAILURE_HANDLE(kernel_create(&device->cuda_kernels[kernel_id], device, library, kernel_id));
  }

  for (uint32_t kernel_id = 0; kernel_id < OPTIX_KERNEL_TYPE_COUNT; kernel_id++) {
    __FAILURE_HANDLE(optix_kernel_create(&device->optix_kernels[kernel_id], device, kernel_id));
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

  CUDA_FAILURE_HANDLE(cuCtxPushCurrent(device->cuda_ctx));

  __FAILURE_HANDLE(_device_load_light_bridge_lut(device));
  __FAILURE_HANDLE(_device_load_bluenoise_texture(device));
  __FAILURE_HANDLE(_device_load_moon_textures(device));

  CUDA_FAILURE_HANDLE(cuCtxPopCurrent(&device->cuda_ctx));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_update_scene_entity(Device* device, const void* object, SceneEntity entity) {
  __CHECK_NULL_ARGUMENT(device);
  __CHECK_NULL_ARGUMENT(object);

  CUDA_FAILURE_HANDLE(cuCtxPushCurrent(device->cuda_ctx));

  const DeviceConstantMemoryMember member = device_scene_entity_to_const_memory_member[entity];
  const size_t member_offset              = device_cuda_const_memory_offsets[member];
  const size_t member_size                = device_cuda_const_memory_sizes[member];

  memcpy(((uint8_t*) device->constant_memory) + member_offset, object, member_size);

  __FAILURE_HANDLE(_device_set_constant_memory_dirty(device, member));

  CUDA_FAILURE_HANDLE(cuCtxPopCurrent(&device->cuda_ctx));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_allocate_work_buffers(Device* device) {
  __CHECK_NULL_ARGUMENT(device);

  CUDA_FAILURE_HANDLE(cuCtxPushCurrent(device->cuda_ctx));

  __FAILURE_HANDLE(_device_allocate_work_buffers(device));

  CUDA_FAILURE_HANDLE(cuCtxPopCurrent(&device->cuda_ctx));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_upload_meshes(Device* device, const ARRAY DeviceMesh** meshes) {
  __CHECK_NULL_ARGUMENT(device);

  CUDA_FAILURE_HANDLE(cuCtxPushCurrent(device->cuda_ctx));

  uint32_t mesh_count;
  __FAILURE_HANDLE(array_get_num_elements(meshes, &mesh_count));

  uint32_t non_instanced_triangle_count = 0;

  for (uint32_t mesh_id = 0; mesh_id < mesh_count; mesh_id++) {
    const DeviceMesh* mesh = meshes[mesh_id];

    uint32_t triangle_count;
    __FAILURE_HANDLE(array_get_num_elements(mesh->triangles, &triangle_count));

    non_instanced_triangle_count += triangle_count;
  }

  __DEVICE_BUFFER_ALLOCATE(triangles, sizeof(DeviceTriangle) * non_instanced_triangle_count);

  uint32_t triangles_offset = 0;

  for (uint32_t mesh_id = 0; mesh_id < mesh_count; mesh_id++) {
    const DeviceMesh* mesh = meshes[mesh_id];

    uint32_t triangle_count;
    __FAILURE_HANDLE(array_get_num_elements(mesh->triangles, &triangle_count));

    __FAILURE_HANDLE(device_staging_manager_register(
      device->staging_manager, (void const*) mesh->triangles, (void*) device->buffers.triangles, triangles_offset,
      sizeof(DeviceTriangle) * triangle_count));

    triangles_offset += triangle_count;
  }

  device->constant_memory->non_instanced_triangle_count = non_instanced_triangle_count;

  DEVICE_UPDATE_CONSTANT_MEMORY(non_instanced_triangle_count, non_instanced_triangle_count);

  CUDA_FAILURE_HANDLE(cuCtxPopCurrent(&device->cuda_ctx));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_build_mesh_bvh(Device* device, const Mesh* mesh) {
  __CHECK_NULL_ARGUMENT(device);
  __CHECK_NULL_ARGUMENT(mesh);

  CUDA_FAILURE_HANDLE(cuCtxPushCurrent(device->cuda_ctx));

  OptixBVH* mesh_bvh;
  __FAILURE_HANDLE(optix_bvh_create(&mesh_bvh));
  __FAILURE_HANDLE(optix_bvh_gas_build(mesh_bvh, device, mesh, OPTIX_BVH_TYPE_DEFAULT));

  __FAILURE_HANDLE(array_push(&device->optix_mesh_bvhs, &mesh_bvh));

  CUDA_FAILURE_HANDLE(cuCtxPopCurrent(&device->cuda_ctx));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_add_textures(Device* device, const Texture** textures, uint32_t num_textures) {
  __CHECK_NULL_ARGUMENT(device);
  __CHECK_NULL_ARGUMENT(textures);

  CUDA_FAILURE_HANDLE(cuCtxPushCurrent(device->cuda_ctx));

  for (uint32_t texture_id = 0; texture_id < num_textures; texture_id++) {
    DeviceTexture* device_texture;
    __FAILURE_HANDLE(device_texture_create(&device_texture, textures[texture_id], device->stream_main));

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

  CUDA_FAILURE_HANDLE(cuCtxPushCurrent(device->cuda_ctx));

  __FAILURE_HANDLE(optix_bvh_instance_cache_update(device->optix_instance_cache, instance_updates));
  __FAILURE_HANDLE(optix_bvh_ias_build(device->optix_bvh_ias, device));

  DEVICE_UPDATE_CONSTANT_MEMORY(optix_bvh, device->optix_bvh_ias->traversable);

  CUDA_FAILURE_HANDLE(cuCtxPopCurrent(&device->cuda_ctx));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_apply_material_updates(
  Device* device, const ARRAY MaterialUpdate* updates, const ARRAY DeviceMaterialCompressed* materials) {
  __CHECK_NULL_ARGUMENT(device);
  __CHECK_NULL_ARGUMENT(updates);
  __CHECK_NULL_ARGUMENT(materials);

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

LuminaryResult device_destroy(Device** device) {
  __CHECK_NULL_ARGUMENT(device);
  __CHECK_NULL_ARGUMENT(*device);

  CUDA_FAILURE_HANDLE(cuCtxPushCurrent((*device)->cuda_ctx));
  CUDA_FAILURE_HANDLE(cuCtxSynchronize());

  __FAILURE_HANDLE(_device_free_embedded_data(*device));
  __FAILURE_HANDLE(_device_free_buffers(*device));

  uint32_t num_meshes;
  __FAILURE_HANDLE(array_get_num_elements((*device)->optix_mesh_bvhs, &num_meshes));

  for (uint32_t mesh_id = 0; mesh_id < num_meshes; mesh_id++) {
    __FAILURE_HANDLE(optix_bvh_destroy(&(*device)->optix_mesh_bvhs[mesh_id]));
  }

  __FAILURE_HANDLE(optix_bvh_instance_cache_destroy(&(*device)->optix_instance_cache));
  __FAILURE_HANDLE(optix_bvh_destroy(&(*device)->optix_bvh_ias));

  __FAILURE_HANDLE(array_destroy(&(*device)->optix_mesh_bvhs));

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

  __FAILURE_HANDLE(device_staging_manager_destroy(&(*device)->staging_manager));

  CUDA_FAILURE_HANDLE(cuStreamDestroy((*device)->stream_main));
  CUDA_FAILURE_HANDLE(cuStreamDestroy((*device)->stream_secondary));

  OPTIX_FAILURE_HANDLE(optixDeviceContextDestroy((*device)->optix_ctx));
  CUDA_FAILURE_HANDLE(cuCtxDestroy((*device)->cuda_ctx));

  __FAILURE_HANDLE(host_free(device));

  return LUMINARY_SUCCESS;
}
