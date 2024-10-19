#include "device.h"

#include <optix_function_table_definition.h>

#include "ceb.h"
#include "device_memory.h"
#include "device_texture.h"
#include "device_utils.h"
#include "host/png.h"
#include "internal_error.h"
#include "light.h"
#include "optixrt.h"

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

  device->constant_memory_dirty = true;

  return LUMINARY_SUCCESS;
}

static LuminaryResult _device_free_embedded_data(Device* device) {
  __CHECK_NULL_ARGUMENT(device);

  __FAILURE_HANDLE(device_texture_destroy(&device->moon_albedo_tex));
  __FAILURE_HANDLE(device_texture_destroy(&device->moon_normal_tex));

  return LUMINARY_SUCCESS;
}

#define __DEVICE_BUFFER_FREE(buffer)          \
  if (buffer) {                               \
    __FAILURE_HANDLE(device_free(&(buffer))); \
  }

static LuminaryResult _device_free_buffers(Device* device) {
  __CHECK_NULL_ARGUMENT(device);

  __DEVICE_BUFFER_FREE(device->buffers.trace_tasks);
  __DEVICE_BUFFER_FREE(device->buffers.aux_data);
  __DEVICE_BUFFER_FREE(device->buffers.trace_counts);
  __DEVICE_BUFFER_FREE(device->buffers.trace_results);
  __DEVICE_BUFFER_FREE(device->buffers.task_counts);
  __DEVICE_BUFFER_FREE(device->buffers.task_offsets);
  __DEVICE_BUFFER_FREE(device->buffers.ior_stack);
  __DEVICE_BUFFER_FREE(device->buffers.frame_variance);
  __DEVICE_BUFFER_FREE(device->buffers.frame_accumulate);
  __DEVICE_BUFFER_FREE(device->buffers.frame_direct_buffer);
  __DEVICE_BUFFER_FREE(device->buffers.frame_direct_accumulate);
  __DEVICE_BUFFER_FREE(device->buffers.frame_indirect_buffer);
  __DEVICE_BUFFER_FREE(device->buffers.frame_indirect_accumulate);
  __DEVICE_BUFFER_FREE(device->buffers.frame_post);
  __DEVICE_BUFFER_FREE(device->buffers.frame_final);
  __DEVICE_BUFFER_FREE(device->buffers.records);
  __DEVICE_BUFFER_FREE(device->buffers.buffer_8bit);
  __DEVICE_BUFFER_FREE(device->buffers.hit_id_history);
  __DEVICE_BUFFER_FREE(device->buffers.albedo_atlas);
  __DEVICE_BUFFER_FREE(device->buffers.luminance_atlas);
  __DEVICE_BUFFER_FREE(device->buffers.material_atlas);
  __DEVICE_BUFFER_FREE(device->buffers.normal_atlas);
  __DEVICE_BUFFER_FREE(device->buffers.cloud_noise);
  __DEVICE_BUFFER_FREE(device->buffers.sky_ms_luts);
  __DEVICE_BUFFER_FREE(device->buffers.sky_tm_luts);
  __DEVICE_BUFFER_FREE(device->buffers.sky_hdri_luts);
  __DEVICE_BUFFER_FREE(device->buffers.bsdf_energy_lut);
  __DEVICE_BUFFER_FREE(device->buffers.bluenoise_1D);
  __DEVICE_BUFFER_FREE(device->buffers.bluenoise_2D);
  __DEVICE_BUFFER_FREE(device->buffers.bridge_lut);
  __DEVICE_BUFFER_FREE(device->buffers.materials);
  __DEVICE_BUFFER_FREE(device->buffers.triangles);
  __DEVICE_BUFFER_FREE(device->buffers.instances);
  __DEVICE_BUFFER_FREE(device->buffers.instance_transforms);
  __DEVICE_BUFFER_FREE(device->buffers.light_instance_map);
  __DEVICE_BUFFER_FREE(device->buffers.bottom_level_light_trees);
  __DEVICE_BUFFER_FREE(device->buffers.bottom_level_light_paths);
  __DEVICE_BUFFER_FREE(device->buffers.top_level_light_tree);
  __DEVICE_BUFFER_FREE(device->buffers.top_level_light_paths);
  __DEVICE_BUFFER_FREE(device->buffers.particle_quads);
  __DEVICE_BUFFER_FREE(device->buffers.stars);
  __DEVICE_BUFFER_FREE(device->buffers.stars_offsets);

  device->constant_memory_dirty = true;

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

  device->index                 = index;
  device->optix_callback_error  = false;
  device->exit_requested        = false;
  device->constant_memory_dirty = true;

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
    __FAILURE_HANDLE(optixrt_kernel_create(&device->optix_kernels[kernel_id], device, kernel_id));
  }

  OPTIX_CHECK_CALLBACK_ERROR(device);

  CUDA_FAILURE_HANDLE(cuCtxPopCurrent(&device->cuda_ctx));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_load_embedded_data(Device* device) {
  __CHECK_NULL_ARGUMENT(device);

  CUDA_FAILURE_HANDLE(cuCtxPushCurrent(device->cuda_ctx));

  __FAILURE_HANDLE(light_load_bridge_lut(device));
  __FAILURE_HANDLE(_device_load_bluenoise_texture(device));
  __FAILURE_HANDLE(_device_load_moon_textures(device));

  device->constant_memory_dirty = true;

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

  for (uint32_t kernel_id = 0; kernel_id < CUDA_KERNEL_TYPE_COUNT; kernel_id++) {
    __FAILURE_HANDLE(kernel_destroy(&(*device)->cuda_kernels[kernel_id]));
  }

  for (uint32_t kernel_id = 0; kernel_id < OPTIX_KERNEL_TYPE_COUNT; kernel_id++) {
    __FAILURE_HANDLE(optixrt_kernel_destroy(&(*device)->optix_kernels[kernel_id]));
  }

  CUDA_FAILURE_HANDLE(cuStreamDestroy((*device)->stream_main));
  CUDA_FAILURE_HANDLE(cuStreamDestroy((*device)->stream_secondary));

  OPTIX_FAILURE_HANDLE(optixDeviceContextDestroy((*device)->optix_ctx));
  CUDA_FAILURE_HANDLE(cuCtxDestroy((*device)->cuda_ctx));

  __FAILURE_HANDLE(host_free(device));

  return LUMINARY_SUCCESS;
}
