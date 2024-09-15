#include "device.h"

#include "device_utils.h"
#include "internal_error.h"

#define DEVICE_RINGBUFFER_SIZE (0x10000ull)
#define DEVICE_QUEUE_SIZE (0x100ull)

void _device_init(void) {
  OPTIX_CHECK(optixInit());
}

////////////////////////////////////////////////////////////////////
// OptiX Log Callback
////////////////////////////////////////////////////////////////////

static void _device_optix_log_callback(unsigned int level, const char* tag, const char* message, void* cbdata) {
  Device* device = (Device*) cbdata;

  switch (level) {
    case 1:
      device->optix_callback_error = true;
      print_error("[OptiX Log Message][%s] %s", tag, message);
      break;
    case 2:
      print_error("[OptiX Log Message][%s] %s", tag, message);
      break;
    case 3:
      print_warn("[OptiX Log Message][%s] %s", tag, message);
      break;
    default:
      print_info("[OptiX Log Message][%s] %s", tag, message);
      break;
  }
}

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

LuminaryResult _device_get_properties(DeviceProperties* props, const uint32_t index) {
  if (!props) {
    __RETURN_ERROR(LUMINARY_ERROR_ARGUMENT_NULL, "Properties is NULL.");
  }

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

  memcpy(props->name, prop.name, 256);

  return LUMINARY_SUCCESS;
}

////////////////////////////////////////////////////////////////////
// Queue worker functions
////////////////////////////////////////////////////////////////////

static void _device_queue_worker(Device* device) {
  while (!device->exit_requested) {
    QueueEntry entry;
    bool success;
    __FAILURE_HANDLE(queue_pop_blocking(device->work_queue, &entry, &success));

    if (!success)
      return;

    __FAILURE_HANDLE(wall_time_set_string(device->queue_wall_time, entry.name));
    __FAILURE_HANDLE(wall_time_start(device->queue_wall_time));

    __FAILURE_HANDLE(entry.function(device, entry.args));

    __FAILURE_HANDLE(wall_time_stop(device->queue_wall_time));
    __FAILURE_HANDLE(wall_time_set_string(device->queue_wall_time, (const char*) 0));
  }
}

////////////////////////////////////////////////////////////////////
// External API implementation
////////////////////////////////////////////////////////////////////

LuminaryResult device_create(Device** _device) {
  if (!_device) {
    __RETURN_ERROR(LUMINARY_ERROR_ARGUMENT_NULL, "Device is NULL.");
  }

  Device* device;
  __FAILURE_HANDLE(host_malloc(&device, sizeof(Device)));

  device->index                = 0;
  device->optix_callback_error = false;
  device->exit_requested       = false;

  __FAILURE_HANDLE(_device_get_properties(&device->properties, device->index));

  __FAILURE_HANDLE(queue_create(&device->work_queue, sizeof(QueueEntry), DEVICE_QUEUE_SIZE));
  __FAILURE_HANDLE(ringbuffer_create(&device->ringbuffer, DEVICE_RINGBUFFER_SIZE));
  __FAILURE_HANDLE(wall_time_create(&device->queue_wall_time));

  _device = device;

  return LUMINARY_SUCCESS;
}

LuminaryResult device_destroy(Device** device) {
  return LUMINARY_ERROR_NOT_IMPLEMENTED;
}
