#include "device_abort.h"

#include "device.h"
#include "internal_error.h"

LuminaryResult device_abort_create(DeviceAbort** abort) {
  __CHECK_NULL_ARGUMENT(abort);

  __FAILURE_HANDLE(host_malloc(abort, sizeof(DeviceAbort)));

  __FAILURE_HANDLE(device_malloc_staging(&(*abort)->staging_buffer, sizeof(uint32_t), DEVICE_MEMORY_STAGING_FLAG_PCIE_TRANSFER_ONLY));
  __FAILURE_HANDLE(device_malloc(&(*abort)->device_buffer, sizeof(uint32_t)));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_abort_set(DeviceAbort* abort, Device* device, bool set_abort) {
  __CHECK_NULL_ARGUMENT(abort);

  *(abort->staging_buffer) = (set_abort) ? 0xFFFFFFFF : 0;
  CUstream stream          = (set_abort) ? device->stream_abort : device->stream_main;

  __FAILURE_HANDLE(device_upload(abort->device_buffer, abort->staging_buffer, 0, sizeof(uint32_t), stream));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_abort_get_ptrs(DeviceAbort* abort, DeviceAbortDeviceBufferPtrs* ptrs) {
  __CHECK_NULL_ARGUMENT(abort);
  __CHECK_NULL_ARGUMENT(ptrs);

  ptrs->abort_flag = DEVICE_CUPTR(abort->device_buffer);

  return LUMINARY_SUCCESS;
}

LuminaryResult device_abort_destroy(DeviceAbort** abort) {
  __CHECK_NULL_ARGUMENT(abort);

  __FAILURE_HANDLE(device_free(&(*abort)->device_buffer));
  __FAILURE_HANDLE(device_free_staging(&(*abort)->staging_buffer));

  __FAILURE_HANDLE(host_free(abort));

  return LUMINARY_SUCCESS;
}
