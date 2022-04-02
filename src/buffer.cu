#include "buffer.h"
#include "cuda/utils.cuh"

size_t memory_limit = 0;
size_t memory_usage = 0;

extern "C" size_t device_memory_usage() {
  return memory_usage;
}

extern "C" size_t device_memory_limit() {
  return memory_limit;
}

extern "C" void device_set_memory_limit(size_t limit) {
  memory_limit = limit;

  log_message("Set memory limit to %zu bytes.", limit);
}

extern "C" void device_malloc(void** buffer, size_t size) {
  memory_usage += size;

  if (memory_usage > (size_t) (0.7 * memory_limit)) {
    warn_message("Device is running low on memory.");
  }

  gpuErrchk(cudaMalloc(buffer, size));
}

extern "C" size_t device_malloc_pitch(void** buffer, size_t rowstride, size_t num_rows) {
  size_t pitch;

  gpuErrchk(cudaMallocPitch(buffer, &pitch, rowstride, num_rows));

  memory_usage += pitch * num_rows;

  if (memory_usage > (size_t) (0.7 * memory_limit)) {
    warn_message("Device is running low on memory.");
  }

  return pitch;
}

extern "C" void device_free(void* buffer, size_t size) {
  memory_usage -= size;

  gpuErrchk(cudaFree(buffer));
}

extern "C" void device_upload(void* dst, void* src, size_t size) {
  gpuErrchk(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
  log_message("Uploaded %zu bytes to device.", size);
}

extern "C" void device_buffer_init(DeviceBuffer** buffer) {
  (*buffer) = (DeviceBuffer*) malloc(sizeof(DeviceBuffer));

  (*buffer)->allocated = 0;
}

extern "C" void device_buffer_free(DeviceBuffer* buffer) {
  if (!buffer) {
    error_message("DeviceBuffer is NULL.");
    return;
  }

  if (!device_buffer_is_allocated(buffer)) {
    log_message("Unallocated device buffer was attempted to be freed.");
    return;
  }

  device_free(buffer->device_pointer, buffer->size);

  buffer->allocated = 0;
  log_message("Freed device buffer %zu.", buffer->device_pointer);
}

extern "C" void device_buffer_malloc(DeviceBuffer* buffer, size_t element_size, size_t count) {
  if (!buffer) {
    error_message("DeviceBuffer is NULL.");
    return;
  }

  size_t size = element_size * count;

  if (device_buffer_is_allocated(buffer)) {
    if (buffer->size == size) {
      log_message("Device buffer is already allocated with the correct size. No action is taken.");
      return;
    }
    else {
      device_buffer_free(buffer);
    }
  }

  device_malloc(&buffer->device_pointer, size);

  buffer->size      = size;
  buffer->allocated = 1;
  log_message("Allocated device buffer %zu with size %zu.", buffer->device_pointer, buffer->size);
}

extern "C" void device_buffer_upload(DeviceBuffer* buffer, void* data) {
  if (!buffer) {
    error_message("DeviceBuffer is NULL.");
    return;
  }

  if (!device_buffer_is_allocated(buffer)) {
    error_message("Cannot upload to unallocated device buffer.");
    return;
  }

  if (!data) {
    error_message("Source pointer is NULL.");
    return;
  }

  device_upload(buffer->device_pointer, data, buffer->size);
  log_message("Copied %zu bytes to device buffer %zu.", buffer->size, buffer->device_pointer);
}

extern "C" void device_buffer_download_full(DeviceBuffer* buffer, void* dest) {
  if (!buffer) {
    error_message("DeviceBuffer is NULL.");
    return;
  }

  device_buffer_download(buffer, dest, buffer->size);
}

extern "C" void device_buffer_download(DeviceBuffer* buffer, void* dest, size_t size) {
  if (!buffer) {
    error_message("DeviceBuffer is NULL.");
    return;
  }

  if (!device_buffer_is_allocated(buffer)) {
    error_message("Cannot download from unallocated device buffer.");
    return;
  }

  if (!dest) {
    error_message("Destination pointer is NULL.");
    return;
  }

  if (size > buffer->size) {
    error_message("Device buffer holds %zu bytes but %zu are requested for download.", buffer->size, size);
    return;
  }

  gpuErrchk(cudaMemcpy(dest, buffer->device_pointer, size, cudaMemcpyDeviceToHost));
  log_message("Copied %zu bytes from device buffer %zu.", buffer->size, buffer->device_pointer);
}

extern "C" void device_buffer_copy(DeviceBuffer* src, DeviceBuffer* dest) {
  if (!src) {
    error_message("Source is NULL.");
    return;
  }

  if (!dest) {
    error_message("Destination is NULL.");
    return;
  }

  if (!device_buffer_is_allocated(src)) {
    error_message("Source is not allocated.");
    return;
  }

  if (!device_buffer_is_allocated(dest) || dest->size < src->size) {
    warn_message("Destination is not allocated or too small.");
    device_buffer_malloc(dest, 1, device_buffer_get_size(src));
  }

  gpuErrchk(cudaMemcpy(device_buffer_get_pointer(dest), device_buffer_get_pointer(src), src->size, cudaMemcpyDeviceToDevice));
  log_message(
    "Copied %zu bytes from device buffer %zu to device buffer %zu.", src->size, device_buffer_get_pointer(src),
    device_buffer_get_pointer(dest));
}

extern "C" void* device_buffer_get_pointer(DeviceBuffer* buffer) {
  if (!buffer) {
    error_message("DeviceBuffer is NULL.");
    return (void*) 0;
  }

  if (!device_buffer_is_allocated(buffer)) {
    log_message("Device buffer is unallocated.");
    return (void*) 0;
  }

  return buffer->device_pointer;
}

extern "C" size_t device_buffer_get_size(DeviceBuffer* buffer) {
  if (!buffer) {
    error_message("DeviceBuffer is NULL.");
    return 0;
  }

  if (!device_buffer_is_allocated(buffer)) {
    log_message("Device buffer is unallocated.");
    return 0;
  }

  return buffer->size;
}

extern "C" int device_buffer_is_allocated(DeviceBuffer* buffer) {
  if (!buffer) {
    error_message("DeviceBuffer is NULL.");
    return 0;
  }

  return buffer->allocated;
}

extern "C" void device_buffer_destroy(DeviceBuffer* buffer) {
  if (device_buffer_is_allocated(buffer)) {
    device_buffer_free(buffer);
  }

  free(buffer);
}
