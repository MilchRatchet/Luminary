#include "device_memory.h"

#include "device_utils.h"
#include "internal_error.h"

struct DeviceMemoryHeader {
  uint64_t magic;
  CUdeviceptr ptr;  // Important that this comes second.
  size_t size;
  size_t pitch;
};

// LUMDEVIM
#define DEVICE_MEMORY_HEADER_MAGIC (0x4D454D49444D554Cull)

static size_t total_memory_allocation[LUMINARY_MAX_NUM_DEVICES];

void _device_memory_init(void) {
  memset(total_memory_allocation, 0, sizeof(size_t) * LUMINARY_MAX_NUM_DEVICES);
}

LuminaryResult _device_malloc(void** _ptr, size_t size, const char* buf_name, const char* func, uint32_t line) {
  __CHECK_NULL_ARGUMENT(_ptr);

  struct DeviceMemoryHeader* header;
  __FAILURE_HANDLE(_host_malloc((void**) &header, sizeof(struct DeviceMemoryHeader), buf_name, func, line));

  CUDA_FAILURE_HANDLE(cudaMalloc((void**) &header->ptr, size));

  header->magic = DEVICE_MEMORY_HEADER_MAGIC;
  header->size  = size;
  header->pitch = 0;

  int current_device;
  CUDA_FAILURE_HANDLE(cudaGetDevice(&current_device));

  total_memory_allocation[current_device] += size;

  *_ptr = (void**) header;

  return LUMINARY_SUCCESS;
}

LuminaryResult _device_malloc2D(void** _ptr, size_t width_in_bytes, size_t height, const char* buf_name, const char* func, uint32_t line) {
  __CHECK_NULL_ARGUMENT(_ptr);

  struct DeviceMemoryHeader* header;
  __FAILURE_HANDLE(_host_malloc((void**) &header, sizeof(struct DeviceMemoryHeader), buf_name, func, line));

  size_t pitch;
  CUDA_FAILURE_HANDLE(cudaMallocPitch((void**) &header->ptr, &pitch, width_in_bytes, height));

  const size_t size = pitch * height;

  header->magic = DEVICE_MEMORY_HEADER_MAGIC;
  header->size  = size;
  header->pitch = pitch;

  int current_device;
  CUDA_FAILURE_HANDLE(cudaGetDevice(&current_device));

  total_memory_allocation[current_device] += size;

  *_ptr = (void**) header;

  return LUMINARY_SUCCESS;
}

LuminaryResult device_upload(DEVICE void* dst, const void* src, size_t dst_offset, size_t size) {
  __CHECK_NULL_ARGUMENT(dst);
  __CHECK_NULL_ARGUMENT(src);

  struct DeviceMemoryHeader* dst_header = (struct DeviceMemoryHeader*) dst;

  if (dst_header->magic != DEVICE_MEMORY_HEADER_MAGIC) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Destination is not device memory.");
  }

  if (dst_offset + size > dst_header->size) {
    __RETURN_ERROR(
      LUMINARY_ERROR_API_EXCEPTION, "Upload exceeds allocated device memory. %llu bytes are allocated and destination is [%llu, %llu].",
      dst_header->size, dst_offset, dst_offset + size);
  }

  CUDA_FAILURE_HANDLE(cudaMemcpy((void*) (dst_header->ptr + dst_offset), src, size, cudaMemcpyHostToDevice));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_memcpy(DEVICE void* dst, DEVICE const void* src, size_t dst_offset, size_t src_offset, size_t size) {
  __CHECK_NULL_ARGUMENT(dst);
  __CHECK_NULL_ARGUMENT(src);

  struct DeviceMemoryHeader* dst_header = (struct DeviceMemoryHeader*) dst;
  struct DeviceMemoryHeader* src_header = (struct DeviceMemoryHeader*) src;

  if (dst_header->magic != DEVICE_MEMORY_HEADER_MAGIC) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Destination is not device memory.");
  }

  if (dst_offset + size > dst_header->size) {
    __RETURN_ERROR(
      LUMINARY_ERROR_API_EXCEPTION,
      "Copy exceeds allocated destination device memory. %llu bytes are allocated and destination is [%llu, %llu].", dst_header->size,
      dst_offset, dst_offset + size);
  }

  if (src_header->magic != DEVICE_MEMORY_HEADER_MAGIC) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Source is not device memory.");
  }

  if (src_offset + size > src_header->size) {
    __RETURN_ERROR(
      LUMINARY_ERROR_API_EXCEPTION, "Download exceeds allocated source device memory. %llu bytes are allocated and source is [%llu, %llu].",
      src_header->size, src_offset, src_offset + size);
  }

  CUDA_FAILURE_HANDLE(
    cudaMemcpy((void*) (dst_header->ptr + dst_offset), (void*) (src_header->ptr + src_offset), size, cudaMemcpyDeviceToDevice));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_download(void* dst, DEVICE const void* src, size_t src_offset, size_t size) {
  __CHECK_NULL_ARGUMENT(dst);
  __CHECK_NULL_ARGUMENT(src);

  struct DeviceMemoryHeader* src_header = (struct DeviceMemoryHeader*) src;

  if (src_header->magic != DEVICE_MEMORY_HEADER_MAGIC) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Source is not device memory.");
  }

  if (src_offset + size > src_header->size) {
    __RETURN_ERROR(
      LUMINARY_ERROR_API_EXCEPTION, "Download exceeds allocated device memory. %llu bytes are allocated and source is [%llu, %llu].",
      src_header->size, src_offset, src_offset + size);
  }

  CUDA_FAILURE_HANDLE(cudaMemcpy(dst, (void*) (src_header->ptr + src_offset), size, cudaMemcpyDeviceToHost));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_upload2D(DEVICE void* dst, const void* src, size_t src_pitch, size_t src_width, size_t src_height) {
  __CHECK_NULL_ARGUMENT(dst);
  __CHECK_NULL_ARGUMENT(src);

  struct DeviceMemoryHeader* dst_header = (struct DeviceMemoryHeader*) dst;

  if (dst_header->magic != DEVICE_MEMORY_HEADER_MAGIC) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Destination is not device memory.");
  }

  if (src_pitch * src_height > dst_header->size) {
    __RETURN_ERROR(
      LUMINARY_ERROR_API_EXCEPTION,
      "Upload exceeds allocated device memory. %llu bytes are allocated and 2D upload would cover %llu bytes.", dst_header->size,
      src_pitch * src_height);
  }

  // TODO: src_width is in bytes, make sure that that is clear.
  CUDA_FAILURE_HANDLE(
    cudaMemcpy2D((void*) dst_header->ptr, dst_header->pitch, src, src_pitch, src_width, src_height, cudaMemcpyHostToDevice));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_memory_get_pitch(DEVICE const void* ptr, size_t* pitch) {
  __CHECK_NULL_ARGUMENT(ptr);
  __CHECK_NULL_ARGUMENT(pitch);

  struct DeviceMemoryHeader* header = (struct DeviceMemoryHeader*) ptr;

  if (header->magic != DEVICE_MEMORY_HEADER_MAGIC) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Destination is not device memory.");
  }

  if (header->pitch == 0) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Device memory was not allocated with a pitch.");
  }

  *pitch = header->pitch;

  return LUMINARY_SUCCESS;
}

LuminaryResult device_memset(DEVICE void* ptr, uint8_t value, size_t offset, size_t size) {
  __CHECK_NULL_ARGUMENT(ptr);

  struct DeviceMemoryHeader* header = (struct DeviceMemoryHeader*) ptr;

  if (header->magic != DEVICE_MEMORY_HEADER_MAGIC) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Pointer is not device memory.");
  }

  if (offset + size > header->size) {
    __RETURN_ERROR(
      LUMINARY_ERROR_API_EXCEPTION, "Memset exceeds allocated device memory. %llu bytes are allocated and destination is [%llu, %llu].",
      header->size, offset, offset + size);
  }

  CUDA_FAILURE_HANDLE(cudaMemset((void*) (header->ptr + offset), value, size));

  return LUMINARY_SUCCESS;
}

LuminaryResult _device_free(DEVICE void** ptr, const char* buf_name, const char* func, uint32_t line) {
  __CHECK_NULL_ARGUMENT(ptr);
  __CHECK_NULL_ARGUMENT(*ptr);

  struct DeviceMemoryHeader* header = (struct DeviceMemoryHeader*) *ptr;

  if (header->magic != DEVICE_MEMORY_HEADER_MAGIC) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Pointer is not device memory.");
  }

  int current_device;
  CUDA_FAILURE_HANDLE(cudaGetDevice(&current_device));

  if (header->size > total_memory_allocation[current_device]) {
    __RETURN_ERROR(
      LUMINARY_ERROR_MEMORY_LEAK, "Device memory allocation is %llu bytes large but only %llu bytes are allocated in total.", header->size,
      total_memory_allocation[current_device]);
  }

  total_memory_allocation[current_device] -= header->size;

  CUDA_FAILURE_HANDLE(cudaFree((void*) header->ptr));

  __FAILURE_HANDLE(_host_free((void**) &header, buf_name, func, line));

  return LUMINARY_SUCCESS;
}
