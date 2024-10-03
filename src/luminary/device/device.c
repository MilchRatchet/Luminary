#include "device.h"

#include "device_utils.h"
#include "internal_error.h"
#include "optixrt.h"

void _device_init(void) {
  OptixResult result = optixInit();

  if (result != OPTIX_SUCCESS) {
    crash_message("Failed to init optix.");
  }
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

static LuminaryResult _device_compile_optix_kernel(Device* device, OptixKernelType type) {
  __CHECK_NULL_ARGUMENT(device);

  __FAILURE_HANDLE(optixrt_kernel_create(&device->optix_kernels[type], device, type));

  return LUMINARY_SUCCESS;
}

////////////////////////////////////////////////////////////////////
// External API implementation
////////////////////////////////////////////////////////////////////

LuminaryResult device_create(Device** _device, uint32_t index) {
  __CHECK_NULL_ARGUMENT(_device);

  Device* device;
  __FAILURE_HANDLE(host_malloc(&device, sizeof(Device)));

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

  ////////////////////////////////////////////////////////////////////
  // Compile OptiX kernels
  ////////////////////////////////////////////////////////////////////

  __FAILURE_HANDLE(_device_compile_optix_kernel(device, OPTIX_KERNEL_TYPE_RAYTRACE_GEOMETRY));
  __FAILURE_HANDLE(_device_compile_optix_kernel(device, OPTIX_KERNEL_TYPE_RAYTRACE_PARTICLES));
  __FAILURE_HANDLE(_device_compile_optix_kernel(device, OPTIX_KERNEL_TYPE_SHADING_GEOMETRY));
  __FAILURE_HANDLE(_device_compile_optix_kernel(device, OPTIX_KERNEL_TYPE_SHADING_VOLUME));
  __FAILURE_HANDLE(_device_compile_optix_kernel(device, OPTIX_KERNEL_TYPE_SHADING_PARTICLES));
  __FAILURE_HANDLE(_device_compile_optix_kernel(device, OPTIX_KERNEL_TYPE_SHADING_VOLUME_BRIDGES));

  *_device = device;

  return LUMINARY_SUCCESS;
}

LuminaryResult device_destroy(Device** device) {
  __CHECK_NULL_ARGUMENT(device);

  __RETURN_ERROR(LUMINARY_ERROR_NOT_IMPLEMENTED, "");
}
