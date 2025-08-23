#include "device_memory.h"

#include "device_utils.h"
#include "internal_error.h"

struct DeviceMemoryHeader {
  uint64_t magic;
  CUdeviceptr ptr;  // Important that this comes second.
  size_t size;
  size_t pitch;
  CUdevice device;
};

// LUMDEVIM
#define DEVICE_MEMORY_HEADER_MAGIC (0x4D454D49444D554Cull)
#define DEVICE_MEMORY_HEADER_FREED_MAGIC (69ull)

static size_t total_memory_allocation[LUMINARY_MAX_NUM_DEVICES];

static_assert(sizeof(CUdevice) == sizeof(int), "This code assumes that CUDevice is just an alias for int.");

void _device_memory_init(void) {
  memset(total_memory_allocation, 0, sizeof(size_t) * LUMINARY_MAX_NUM_DEVICES);
}

void _device_memory_shutdown(void) {
  for (uint32_t device_id = 0; device_id < LUMINARY_MAX_NUM_DEVICES; device_id++) {
    if (total_memory_allocation[device_id]) {
      luminary_print_error("Device %u leaked %llu bytes.", device_id, total_memory_allocation[device_id]);
    }
  }
}

LuminaryResult _device_malloc(DEVICE void** _ptr, size_t size, const char* buf_name, const char* func, uint32_t line) {
  __CHECK_NULL_ARGUMENT(_ptr);

  struct DeviceMemoryHeader* header;
  __FAILURE_HANDLE(_host_malloc((void**) &header, sizeof(struct DeviceMemoryHeader), buf_name, func, line));

  if (size > 0) {
    CUDA_FAILURE_HANDLE(cuMemAlloc(&header->ptr, size));
  }
  else {
    header->ptr = (CUdeviceptr) 0;
  }

  CUdevice device;
  CUDA_FAILURE_HANDLE(cuCtxGetDevice(&device));

  header->magic  = DEVICE_MEMORY_HEADER_MAGIC;
  header->size   = size;
  header->pitch  = 0;
  header->device = device;

  total_memory_allocation[device] += size;

  *_ptr = (void**) header;

  return LUMINARY_SUCCESS;
}

LuminaryResult _device_malloc2D(void** _ptr, size_t width_in_bytes, size_t height, const char* buf_name, const char* func, uint32_t line) {
  __CHECK_NULL_ARGUMENT(_ptr);

  struct DeviceMemoryHeader* header;
  __FAILURE_HANDLE(_host_malloc((void**) &header, sizeof(struct DeviceMemoryHeader), buf_name, func, line));

  size_t pitch;
  if (width_in_bytes > 0 && height) {
    CUDA_FAILURE_HANDLE(cuMemAllocPitch(&header->ptr, &pitch, width_in_bytes, height, 16));
  }
  else {
    header->ptr = (CUdeviceptr) 0;
    pitch       = 0;
  }

  const size_t size = pitch * height;

  CUdevice device;
  CUDA_FAILURE_HANDLE(cuCtxGetDevice(&device));

  header->magic  = DEVICE_MEMORY_HEADER_MAGIC;
  header->size   = size;
  header->pitch  = pitch;
  header->device = device;

  total_memory_allocation[device] += size;

  *_ptr = (void**) header;

  return LUMINARY_SUCCESS;
}

LuminaryResult device_upload(DEVICE void* dst, const void* src, size_t dst_offset, size_t size, CUstream stream) {
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

  CUDA_FAILURE_HANDLE(cuMemcpyHtoDAsync(dst_header->ptr + dst_offset, src, size, stream));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_memcpy(DEVICE void* dst, DEVICE const void* src, size_t dst_offset, size_t src_offset, size_t size, CUstream stream) {
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

  CUDA_FAILURE_HANDLE(cuMemcpyAsync(dst_header->ptr + dst_offset, src_header->ptr + src_offset, size, stream));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_download(void* dst, DEVICE const void* src, size_t src_offset, size_t size, CUstream stream) {
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

  CUDA_FAILURE_HANDLE(cuMemcpyDtoHAsync(dst, src_header->ptr + src_offset, size, stream));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_upload2D(DEVICE void* dst, const void* src, size_t src_pitch, size_t src_width, size_t src_height, CUstream stream) {
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
  CUDA_MEMCPY2D copy_info;
  copy_info.dstDevice     = dst_header->ptr;
  copy_info.dstMemoryType = CU_MEMORYTYPE_DEVICE;
  copy_info.dstPitch      = dst_header->pitch;
  copy_info.dstXInBytes   = 0;
  copy_info.dstY          = 0;
  copy_info.srcHost       = src;
  copy_info.srcMemoryType = CU_MEMORYTYPE_HOST;
  copy_info.srcPitch      = src_pitch;
  copy_info.srcXInBytes   = 0;
  copy_info.srcY          = 0;
  copy_info.WidthInBytes  = src_width;
  copy_info.Height        = src_height;

  CUDA_FAILURE_HANDLE(cuMemcpy2DAsync(&copy_info, stream));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_download2D(
  void* dst, const DEVICE void* src, size_t src_pitch, size_t src_width, size_t src_height, CUstream stream) {
  __CHECK_NULL_ARGUMENT(dst);
  __CHECK_NULL_ARGUMENT(src);

  struct DeviceMemoryHeader* src_header = (struct DeviceMemoryHeader*) src;

  if (src_header->magic != DEVICE_MEMORY_HEADER_MAGIC) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Destination is not device memory.");
  }

  if (src_pitch * src_height > src_header->size) {
    __RETURN_ERROR(
      LUMINARY_ERROR_API_EXCEPTION,
      "Download exceeds allocated device memory. %llu bytes are allocated and 2D upload would cover %llu bytes.", src_header->size,
      src_pitch * src_height);
  }

  // TODO: src_width is in bytes, make sure that that is clear.
  CUDA_MEMCPY2D copy_info;
  copy_info.dstHost       = dst;
  copy_info.dstMemoryType = CU_MEMORYTYPE_HOST;
  copy_info.dstPitch      = src_width;
  copy_info.dstXInBytes   = 0;
  copy_info.dstY          = 0;
  copy_info.srcDevice     = src_header->ptr;
  copy_info.srcMemoryType = CU_MEMORYTYPE_DEVICE;
  copy_info.srcPitch      = src_pitch;
  copy_info.srcXInBytes   = 0;
  copy_info.srcY          = 0;
  copy_info.WidthInBytes  = src_width;
  copy_info.Height        = src_height;

  CUDA_FAILURE_HANDLE(cuMemcpy2DAsync(&copy_info, stream));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_memory_get_pitch(DEVICE const void* ptr, size_t* pitch) {
  __CHECK_NULL_ARGUMENT(ptr);
  __CHECK_NULL_ARGUMENT(pitch);

  struct DeviceMemoryHeader* header = (struct DeviceMemoryHeader*) ptr;

  if (header->magic != DEVICE_MEMORY_HEADER_MAGIC) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Destination is not device memory.");
  }

  if ((header->ptr != (CUdeviceptr) 0) && (header->pitch == 0)) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Device memory was not allocated with a pitch.");
  }

  *pitch = header->pitch;

  return LUMINARY_SUCCESS;
}

LuminaryResult device_memory_get_size(DEVICE const void* ptr, size_t* size) {
  __CHECK_NULL_ARGUMENT(ptr);
  __CHECK_NULL_ARGUMENT(size);

  struct DeviceMemoryHeader* header = (struct DeviceMemoryHeader*) ptr;

  if (header->magic != DEVICE_MEMORY_HEADER_MAGIC) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Destination is not device memory.");
  }

  *size = header->size;

  return LUMINARY_SUCCESS;
}

LuminaryResult device_memset(DEVICE void* ptr, uint8_t value, size_t offset, size_t size, CUstream stream) {
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

  CUDA_FAILURE_HANDLE(cuMemsetD8Async(header->ptr + offset, value, size, stream));

  return LUMINARY_SUCCESS;
}

LuminaryResult _device_free(DEVICE void** ptr, const char* buf_name, const char* func, uint32_t line) {
  __CHECK_NULL_ARGUMENT(ptr);
  __CHECK_NULL_ARGUMENT(*ptr);

  struct DeviceMemoryHeader* header = (struct DeviceMemoryHeader*) *ptr;

  if (header->magic != DEVICE_MEMORY_HEADER_MAGIC) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Pointer is not device memory.");
  }

  header->magic = DEVICE_MEMORY_HEADER_FREED_MAGIC;

  CUdevice device;
  CUDA_FAILURE_HANDLE(cuCtxGetDevice(&device));

  if (device != header->device) {
    __RETURN_ERROR(
      LUMINARY_ERROR_API_EXCEPTION, "Device memory allocation belongs to device %u but was attempted to be freed for device %u.",
      header->device, device);
  }

  if (header->size > total_memory_allocation[device]) {
    __RETURN_ERROR(
      LUMINARY_ERROR_MEMORY_LEAK, "Device memory allocation is %llu bytes large but only %llu bytes are allocated in total.", header->size,
      total_memory_allocation[device]);
  }

  total_memory_allocation[device] -= header->size;

  if (header->ptr) {
    CUDA_FAILURE_HANDLE(cuMemFree(header->ptr));
  }

  __FAILURE_HANDLE(_host_free((void**) &header, buf_name, func, line));

  *ptr = (void*) 0;

  return LUMINARY_SUCCESS;
}

LuminaryResult device_memory_get_total_allocation_size(CUdevice device, size_t* size) {
  __CHECK_NULL_ARGUMENT(size);

  *size = total_memory_allocation[device];

  return LUMINARY_SUCCESS;
}

struct DeviceStagingMemoryHeader {
  uint64_t magic;
  uint64_t size;
};

// LUMDEVIS
#define DEVICE_STAGING_MEMORY_HEADER_MAGIC (0x53454D49444D554Cull)
#define DEVICE_STAGING_MEMORY_HEADER_FREED_MAGIC (70ull)

LuminaryResult _device_malloc_staging(STAGING void** ptr, size_t size, bool upload_only) {
  __CHECK_NULL_ARGUMENT(ptr);

  struct DeviceStagingMemoryHeader* header;

  // CU_MEMHOSTALLOC_WRITECOMBINED allows for fast transfer over PCI-E bus but is very slow to read from on the CPU.
  // Hence we can't use it for stuff like downloading images from the GPU as it would be slow to then use the image on the CPU.
  // TODO: Implement a memcpy STAGING->HOST using the SSE4 instruction MOVNTDQA which allows me to use this flag also for downloads
  // because I can then use this memcpy to retrieve the data on the CPU.
  const uint32_t flags = (upload_only) ? CU_MEMHOSTALLOC_WRITECOMBINED : 0;

  CUDA_FAILURE_HANDLE(cuMemHostAlloc((void**) &header, size + sizeof(struct DeviceStagingMemoryHeader), flags));

  header->magic = DEVICE_STAGING_MEMORY_HEADER_MAGIC;
  header->size  = size;

  *ptr = (void*) (header + 1);

  return LUMINARY_SUCCESS;
}

LuminaryResult _device_free_staging(STAGING void** ptr) {
  __CHECK_NULL_ARGUMENT(ptr);
  __CHECK_NULL_ARGUMENT(*ptr);

  struct DeviceStagingMemoryHeader* header = ((struct DeviceStagingMemoryHeader*) *ptr) - 1;

  if (header->magic != DEVICE_STAGING_MEMORY_HEADER_MAGIC) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Pointer is not staging memory.");
  }

  header->magic = DEVICE_STAGING_MEMORY_HEADER_FREED_MAGIC;

  CUDA_FAILURE_HANDLE(cuMemFreeHost(header));

  *ptr = (void*) 0;

  return LUMINARY_SUCCESS;
}
