#include "buffer.h"

#include <cuda_runtime_api.h>

#include "log.h"

#define gpuBufferErrchk(ans, buf_name, func, line)                                                                \
  {                                                                                                               \
    if (ans != cudaSuccess) {                                                                                     \
      crash_message("Buffer CUDA Error: %s at Buffer %s [%s:%d]", cudaGetErrorString(ans), buf_name, func, line); \
    }                                                                                                             \
  }

size_t memory_limit = 0;
size_t memory_usage = 0;

size_t device_memory_usage() {
  return memory_usage;
}

size_t device_memory_limit() {
  return memory_limit;
}

void device_set_memory_limit(size_t limit) {
  memory_limit = limit;

  log_message("Set memory limit to %zu bytes.", limit);
}

void _device_malloc(void** buffer, size_t size, char* buf_name, char* func, int line) {
  memory_usage += size;

  if (memory_usage > (size_t) (0.7 * memory_limit)) {
    print_warn(
      "[%s:%d] Device is running low on memory (%zu/%zu).", func, line, memory_usage / (1024 * 1024), memory_limit / (1024 * 1024));
  }

  gpuBufferErrchk(cudaMalloc(buffer, size), buf_name, func, line);
}

size_t _device_malloc_pitch(void** buffer, size_t rowstride, size_t num_rows, char* buf_name, char* func, int line) {
  size_t pitch;

  gpuBufferErrchk(cudaMallocPitch(buffer, &pitch, rowstride, num_rows), buf_name, func, line);

  memory_usage += pitch * num_rows;

  if (memory_usage > (size_t) (0.7 * memory_limit)) {
    print_warn(
      "[%s:%d] Device is running low on memory (%zu/%zu).", func, line, memory_usage / (1024 * 1024), memory_limit / (1024 * 1024));
  }

  return pitch;
}

void _device_free(void* buffer, size_t size, char* buf_name, char* func, int line) {
  memory_usage -= size;

  gpuBufferErrchk(cudaFree(buffer), buf_name, func, line);
}

void _device_upload(void* dst, void* src, size_t size, char* dst_name, char* src_name, char* func, int line) {
  gpuBufferErrchk(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice), "none", func, line);
  print_log("[%s:%d] Uploaded %zu bytes from host %s to device %s.", func, line, size, src_name, dst_name);
}

void _device_download(void* dst, void* src, size_t size, char* dst_name, char* src_name, char* func, int line) {
  gpuBufferErrchk(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost), "none", func, line);
  print_log("[%s:%d] Downloaded %zu bytes from device %s to host %s.", func, line, size, src_name, dst_name);
}

void _device_buffer_init(DeviceBuffer** buffer, char* buf_name, char* func, int line) {
  if (*buffer) {
    print_log("[%s:%d] Device buffer %s was already initialized.", func, line, buf_name);
    return;
  }

  (*buffer) = (DeviceBuffer*) calloc(1, sizeof(DeviceBuffer));

  print_log("[%s:%d] Initialized device buffer %s.", func, line, buf_name);
}

void _device_buffer_free(DeviceBuffer* buffer, char* buf_name, char* func, int line) {
  if (!buffer) {
    print_error("[%s:%d] DeviceBuffer %s is NULL.", func, line, buf_name);
    return;
  }

  if (!_device_buffer_is_allocated(buffer, buf_name, func, line)) {
    log_message("Unallocated device buffer was attempted to be freed.");
    return;
  }

  device_free(buffer->device_pointer, buffer->size);

  buffer->allocated = 0;
  print_log("[%s:%d] Freed device buffer %s of size %zu.", func, line, buf_name, buffer->size);
}

void _device_buffer_malloc(DeviceBuffer* buffer, size_t element_size, size_t count, char* buf_name, char* func, int line) {
  if (!buffer) {
    print_error("[%s:%d] DeviceBuffer %s is NULL.", func, line, buf_name);
    return;
  }

  size_t size = element_size * count;

  if (_device_buffer_is_allocated(buffer, buf_name, func, line)) {
    if (buffer->size == size) {
      print_log("[%s:%d] Device buffer %s is already allocated with the correct size. No action is taken.", func, line, buf_name);
      return;
    }
    else {
      _device_buffer_free(buffer, buf_name, func, line);
    }
  }

  _device_malloc(&buffer->device_pointer, size, buf_name, func, line);

  buffer->size      = size;
  buffer->allocated = 1;
  print_log("[%s:%d] Buffer %s allocated at address %llX with size %zu.", func, line, buf_name, buffer->device_pointer, buffer->size);
}

void _device_buffer_zero(DeviceBuffer* buffer, char* buf_name, char* func, int line) {
  if (!buffer) {
    print_error("[%s:%d] DeviceBuffer %s is NULL.", func, line, buf_name);
    return;
  }

  if (!_device_buffer_is_allocated(buffer, buf_name, func, line)) {
    print_error("[%s:%d] DeviceBuffer %s is not allocated.", func, line, buf_name);
    return;
  }

  gpuBufferErrchk(cudaMemset(buffer->device_pointer, 0, buffer->size), buf_name, func, line);

  print_log("[%s:%d] Buffer %s zeroed at address %llX with size %zu.", func, line, buf_name, buffer->device_pointer, buffer->size);
}

void _device_buffer_upload(DeviceBuffer* buffer, void* data, char* buf_name, char* data_name, char* func, int line) {
  if (!buffer) {
    print_error("[%s:%d] DeviceBuffer %s is NULL.", func, line, buf_name);
    return;
  }

  if (!_device_buffer_is_allocated(buffer, buf_name, func, line)) {
    print_error("[%s:%d] Cannot upload to unallocated device buffer %s.", func, line, buf_name);
    return;
  }

  if (!data) {
    print_error("[%s:%d] Source pointer %s is NULL.", func, line, data_name);
    return;
  }

  _device_upload(buffer->device_pointer, data, buffer->size, buf_name, data_name, func, line);
  print_log("[%s:%d] Copied %zu bytes to device buffer %s.", func, line, buffer->size, buf_name);
}

void _device_buffer_download(DeviceBuffer* buffer, void* dest, size_t size, char* buf_name, char* dest_name, char* func, int line) {
  if (!buffer) {
    print_error("[%s:%d] DeviceBuffer %s is NULL.", func, line, buf_name);
    return;
  }

  if (!_device_buffer_is_allocated(buffer, buf_name, func, line)) {
    print_error("[%s:%d] Cannot download from unallocated device buffer %s.", func, line, buf_name);
    return;
  }

  if (!dest) {
    print_error("[%s:%d] Destination %s is NULL.", func, line, dest_name);
    return;
  }

  if (size > buffer->size) {
    print_error("[%s:%d] Device buffer %s holds %zu bytes but %zu are requested for download.", func, line, buf_name, buffer->size, size);
    return;
  }

  gpuBufferErrchk(cudaMemcpy(dest, buffer->device_pointer, size, cudaMemcpyDeviceToHost), buf_name, func, line);
  // print_log("[%s:%d] Copied %zu bytes from device buffer %s to host %s.", func, line, size, buf_name, dest_name);
}

void _device_buffer_download_full(DeviceBuffer* buffer, void* dest, char* buf_name, char* dest_name, char* func, int line) {
  if (!buffer) {
    print_error("[%s:%d] DeviceBuffer %s is NULL.", func, line, buf_name);
    return;
  }

  _device_buffer_download(buffer, dest, buffer->size, buf_name, dest_name, func, line);
}

void _device_buffer_copy(DeviceBuffer* src, DeviceBuffer* dest, char* src_name, char* dest_name, char* func, int line) {
  if (!src) {
    print_error("[%s:%d] Source Buffer %s is NULL.", func, line, src_name);
    return;
  }

  if (!dest) {
    print_error("[%s:%d] Destination Buffer %s is NULL.", func, line, dest_name);
    return;
  }

  if (!_device_buffer_is_allocated(src, src_name, func, line)) {
    print_error("[%s:%d] Source Buffer %s is not allocated.", func, line, src_name);
    return;
  }

  if (!_device_buffer_is_allocated(dest, dest_name, func, line) || dest->size < src->size) {
    print_warn("[%s:%d] Destination Buffer %s is not allocated or too small.", func, line, src_name);
    device_buffer_malloc(dest, 1, device_buffer_get_size(src));
  }

  gpuBufferErrchk(
    cudaMemcpy(device_buffer_get_pointer(dest), device_buffer_get_pointer(src), src->size, cudaMemcpyDeviceToDevice), src_name, func, line);
  print_log("[%s:%d] Copied %zu bytes from device buffer %s to device buffer %s.", func, line, src->size, src_name, dest_name);
}

void* _device_buffer_get_pointer(DeviceBuffer* buffer, char* buf_name, char* func, int line) {
  if (!buffer) {
    print_error("[%s:%d] DeviceBuffer %s is NULL.", func, line, buf_name);
    return (void*) 0;
  }

  if (!_device_buffer_is_allocated(buffer, buf_name, func, line)) {
    print_log("[%s:%d] Device buffer %s is unallocated.", func, line, buf_name);
    return (void*) 0;
  }

  return buffer->device_pointer;
}

size_t _device_buffer_get_size(DeviceBuffer* buffer, char* buf_name, char* func, int line) {
  if (!buffer) {
    print_error("[%s:%d] DeviceBuffer %s is NULL.", func, line, buf_name);
    return 0;
  }

  if (!_device_buffer_is_allocated(buffer, buf_name, func, line)) {
    log_message("Device buffer is unallocated.");
    return 0;
  }

  return buffer->size;
}

int _device_buffer_is_allocated(DeviceBuffer* buffer, char* buf_name, char* func, int line) {
  if (!buffer) {
    print_error("[%s:%d] DeviceBuffer %s is NULL.", func, line, buf_name);
    return 0;
  }

  return buffer->allocated;
}

void _device_buffer_destroy(DeviceBuffer** buffer, char* buf_name, char* func, int line) {
  if (!buffer || !(*buffer)) {
    print_error("[%s:%d] DeviceBuffer %s is NULL.", func, line, buf_name);
    return;
  }

  if (_device_buffer_is_allocated(*buffer, buf_name, func, line)) {
    _device_buffer_free(*buffer, buf_name, func, line);
  }

  free(*buffer);
  *buffer = (DeviceBuffer*) 0;
}
