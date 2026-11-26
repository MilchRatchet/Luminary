#include "device_abort.h"

#include "device.h"
#include "internal_error.h"

LuminaryResult device_abort_create(DeviceAbort** abort) {
  __CHECK_NULL_ARGUMENT(abort);

  __FAILURE_HANDLE(host_malloc(abort, sizeof(DeviceAbort)));

  __FAILURE_HANDLE(device_malloc_staging(&(*abort)->staging_buffer, 2 * sizeof(uint32_t), DEVICE_MEMORY_STAGING_FLAG_PCIE_TRANSFER_ONLY));
  (*abort)->staging_buffer[0] = 0;
  (*abort)->staging_buffer[1] = UINT32_MAX;
  __FAILURE_HANDLE(device_malloc(&(*abort)->device_buffer, sizeof(uint32_t)));
  CUDA_FAILURE_HANDLE(cuEventCreate(&(*abort)->flag_cleared, CU_EVENT_DISABLE_TIMING));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_abort_set(DeviceAbort* abort, Device* device, bool set_abort) {
  __CHECK_NULL_ARGUMENT(abort);

  if (set_abort) {
    CUDA_FAILURE_HANDLE(cuStreamWaitEvent(device->stream_abort, abort->flag_cleared, CU_EVENT_WAIT_DEFAULT));
    __FAILURE_HANDLE(device_upload(abort->device_buffer, abort->staging_buffer + 1, 0, sizeof(uint32_t), device->stream_abort));
    cuStreamQuery(device->stream_abort);
  }
  else {
    __FAILURE_HANDLE(device_upload(abort->device_buffer, abort->staging_buffer, 0, sizeof(uint32_t), device->stream_main));
    CUDA_FAILURE_HANDLE(cuEventRecord(abort->flag_cleared, device->stream_main));
  }

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

  CUDA_FAILURE_HANDLE(cuEventDestroy((*abort)->flag_cleared));
  __FAILURE_HANDLE(device_free(&(*abort)->device_buffer));
  __FAILURE_HANDLE(device_free_staging(&(*abort)->staging_buffer));

  __FAILURE_HANDLE(host_free(abort));

  return LUMINARY_SUCCESS;
}
