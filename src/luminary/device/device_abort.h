#ifndef LUMINARY_DEVICE_ABORT_H
#define LUMINARY_DEVICE_ABORT_H

#include "device_memory.h"
#include "device_utils.h"

struct Device typedef Device;

struct DeviceAbortDeviceBufferPtrs {
  CUdeviceptr abort_flag;
} typedef DeviceAbortDeviceBufferPtrs;

struct DeviceAbort {
  STAGING uint32_t* staging_buffer;
  DEVICE uint32_t* device_buffer;
} typedef DeviceAbort;

DEVICE_CTX_FUNC LuminaryResult device_abort_create(DeviceAbort** abort);
DEVICE_CTX_FUNC LuminaryResult device_abort_set(DeviceAbort* abort, Device* device, bool set_abort);
DEVICE_CTX_FUNC LuminaryResult device_abort_get_ptrs(DeviceAbort* abort, DeviceAbortDeviceBufferPtrs* ptrs);
DEVICE_CTX_FUNC LuminaryResult device_abort_destroy(DeviceAbort** abort);

#endif /* LUMINARY_DEVICE_ABORT_H */
