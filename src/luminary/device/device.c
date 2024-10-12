#include "device.h"

#include <optix_function_table_definition.h>

#include "device_memory.h"
#include "device_utils.h"
#include "internal_error.h"
#include "optixrt.h"

void _device_init(void) {
  OptixResult result = optixInit();

  if (result != OPTIX_SUCCESS) {
    crash_message("Failed to init optix.");
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

static LuminaryResult _device_get_properties(DeviceProperties* props, const uint32_t index) {
  __CHECK_NULL_ARGUMENT(props);

  // TODO: Use correct device ID.
  struct cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, index);

  switch (prop.major) {
    case 6: {
      if (prop.minor == 0 || prop.minor == 1 || prop.minor == 2) {
        props->arch            = DEVICE_ARCH_PASCAL;
        props->rt_core_version = 0;
      }
      else {
        props->arch            = DEVICE_ARCH_UNKNOWN;
        props->rt_core_version = 0;
      }
    } break;
    case 7: {
      if (prop.minor == 0 || prop.minor == 2) {
        props->arch            = DEVICE_ARCH_VOLTA;
        props->rt_core_version = 0;
      }
      else if (prop.minor == 5) {
        props->arch            = DEVICE_ARCH_TURING;
        props->rt_core_version = 1;

        // TU116 and TU117 do not have RT cores, these can be detected by searching for GTX in the name
        for (int i = 0; i < 256; i++) {
          if (prop.name[i] == 'G') {
            props->rt_core_version = 0;
          }
        }
      }
      else {
        props->arch            = DEVICE_ARCH_UNKNOWN;
        props->rt_core_version = 0;
      }
    } break;
    case 8: {
      if (prop.minor == 0) {
        // GA100 has no RT cores
        props->arch            = DEVICE_ARCH_AMPERE;
        props->rt_core_version = 0;
      }
      else if (prop.minor == 6 || prop.minor == 7) {
        props->arch            = DEVICE_ARCH_AMPERE;
        props->rt_core_version = 2;
      }
      else if (prop.minor == 9) {
        props->arch            = DEVICE_ARCH_ADA;
        props->rt_core_version = 3;
      }
      else {
        props->arch            = DEVICE_ARCH_UNKNOWN;
        props->rt_core_version = 0;
      }
    } break;
    case 9: {
      if (prop.minor == 0) {
        props->arch            = DEVICE_ARCH_HOPPER;
        props->rt_core_version = 0;
      }
      else {
        props->arch            = DEVICE_ARCH_UNKNOWN;
        props->rt_core_version = 0;
      }
    } break;
    case 10: {
      if (prop.minor == 0) {
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
      "Luminary failed to identify architecture of CUDA compute capability %d.%d. Some features may not be working.", prop.major,
      prop.minor);
  }

  props->memory_size = prop.totalGlobalMem;

  memcpy((void*) props->name, prop.name, 256);

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
  __DEVICE_BUFFER_FREE(device->buffers.sky_moon_albedo_tex);
  __DEVICE_BUFFER_FREE(device->buffers.sky_moon_normal_tex);
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

  device->index                    = index;
  device->optix_callback_error     = false;
  device->exit_requested           = false;
  device->accumulated_sample_count = 0;

  CUDA_FAILURE_HANDLE(cudaInitDevice(device->index, cudaDeviceScheduleAuto, cudaInitDeviceFlagsAreValid));

  __FAILURE_HANDLE(_device_get_properties(&device->properties, device->index));

  CUDA_FAILURE_HANDLE(cudaSetDevice(device->index));

  OptixDeviceContextOptions optix_device_context_options;
  memset(&optix_device_context_options, 0, sizeof(OptixDeviceContextOptions));

#ifdef OPTIX_VALIDATION
  optix_device_context_options.logCallbackData     = (void*) 0;
  optix_device_context_options.logCallbackFunction = _device_optix_log_callback;
  optix_device_context_options.logCallbackLevel    = 3;
  optix_device_context_options.validationMode      = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
#endif

  OPTIX_FAILURE_HANDLE(optixDeviceContextCreate((CUcontext) 0, &optix_device_context_options, &device->optix_ctx));

  *_device = device;

  return LUMINARY_SUCCESS;
}

LuminaryResult device_compile_kernels(Device* device) {
  __CHECK_NULL_ARGUMENT(device);

  for (uint32_t kernel_id = 0; kernel_id < OPTIX_KERNEL_TYPE_COUNT; kernel_id++) {
    __FAILURE_HANDLE(optixrt_kernel_create(&device->optix_kernels[kernel_id], device, kernel_id));
  }

  OPTIX_CHECK_CALLBACK_ERROR(device);

  return LUMINARY_SUCCESS;
}

LuminaryResult device_destroy(Device** device) {
  __CHECK_NULL_ARGUMENT(device);

  CUDA_FAILURE_HANDLE(cudaSetDevice((*device)->index));
  CUDA_FAILURE_HANDLE(cudaDeviceSynchronize());

  __FAILURE_HANDLE(_device_free_buffers(*device));

  for (uint32_t kernel_id = 0; kernel_id < OPTIX_KERNEL_TYPE_COUNT; kernel_id++) {
    __FAILURE_HANDLE(optixrt_kernel_destroy(&(*device)->optix_kernels[kernel_id]));
  }

  OPTIX_FAILURE_HANDLE(optixDeviceContextDestroy((*device)->optix_ctx));

  __FAILURE_HANDLE(host_free(device));

  return LUMINARY_SUCCESS;
}
