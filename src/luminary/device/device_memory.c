#include "device_memory.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <threads.h>

#include "device_utils.h"
#include "internal_error.h"

struct DeviceMemoryHeader {
  uint64_t magic;
  CUdeviceptr ptr;  // Important that this comes second.
  size_t size;
  size_t pitch;
  CUdevice device;
#ifdef LUMINARY_ENABLE_DEBUG_API
  char* name;
  struct DeviceMemoryHeader* next;
#endif
};

// LUMDEVIM
#define DEVICE_MEMORY_HEADER_MAGIC (0x4D454D49444D554Cull)
#define DEVICE_MEMORY_HEADER_FREED_MAGIC (69ull)

static size_t total_memory_allocation[LUMINARY_MAX_NUM_DEVICES];
static mtx_t _device_memory_mutex;
#ifdef LUMINARY_ENABLE_DEBUG_API
static struct DeviceMemoryHeader* _debug_device_memory_allocations;
#endif

static_assert(sizeof(CUdevice) == sizeof(int), "This code assumes that CUDevice is just an alias for int.");

#ifdef LUMINARY_ENABLE_DEBUG_API
static LuminaryResult _device_memory_set_allocation_name(
  struct DeviceMemoryHeader* header, const char* buf_name, const char* func, uint32_t line) {
  const int name_size = snprintf((char*) 0, 0, "[%s:%u]: %s", func, line, buf_name);

  if (name_size < 0) {
    __RETURN_ERROR(LUMINARY_ERROR_OUT_OF_MEMORY, "Failed to format device allocation tracking name.");
  }

  header->name = (char*) malloc((size_t) name_size + 1);

  if (header->name == (char*) 0) {
    __RETURN_ERROR(LUMINARY_ERROR_OUT_OF_MEMORY, "Failed to allocate device allocation tracking name.");
  }

  if (snprintf(header->name, (size_t) name_size + 1, "[%s:%u]: %s", func, line, buf_name) < 0) {
    free(header->name);
    header->name = (char*) 0;
    __RETURN_ERROR(LUMINARY_ERROR_OUT_OF_MEMORY, "Failed to create device allocation tracking name.");
  }

  return LUMINARY_SUCCESS;
}
#endif

static void _device_memory_add_allocation(struct DeviceMemoryHeader* header) {
  mtx_lock(&_device_memory_mutex);

  total_memory_allocation[header->device] += header->size;

#ifdef LUMINARY_ENABLE_DEBUG_API
  header->next                     = _debug_device_memory_allocations;
  _debug_device_memory_allocations = header;
#endif

  mtx_unlock(&_device_memory_mutex);
}

static LuminaryResult _device_memory_remove_allocation(struct DeviceMemoryHeader* header) {
  mtx_lock(&_device_memory_mutex);

  const size_t total_size = total_memory_allocation[header->device];
  if (header->size > total_size) {
    mtx_unlock(&_device_memory_mutex);
    __RETURN_ERROR(
      LUMINARY_ERROR_MEMORY_LEAK, "Device memory allocation is %llu bytes large but only %llu bytes are allocated in total.", header->size,
      total_size);
  }

  total_memory_allocation[header->device] -= header->size;

#ifdef LUMINARY_ENABLE_DEBUG_API
  struct DeviceMemoryHeader** allocation = &_debug_device_memory_allocations;
  while (*allocation && *allocation != header)
    allocation = &(*allocation)->next;

  if (*allocation == header)
    *allocation = header->next;
#endif

  mtx_unlock(&_device_memory_mutex);

  return LUMINARY_SUCCESS;
}

void _device_memory_init(void) {
  memset(total_memory_allocation, 0, sizeof(size_t) * LUMINARY_MAX_NUM_DEVICES);
  mtx_init(&_device_memory_mutex, mtx_plain);
#ifdef LUMINARY_ENABLE_DEBUG_API
  _debug_device_memory_allocations = (struct DeviceMemoryHeader*) 0;
#endif
}

void _device_memory_shutdown(void) {
  mtx_lock(&_device_memory_mutex);

  for (uint32_t device_id = 0; device_id < LUMINARY_MAX_NUM_DEVICES; device_id++) {
    if (total_memory_allocation[device_id]) {
      luminary_print_error("Device %u leaked %llu bytes.", device_id, total_memory_allocation[device_id]);
    }
  }

#ifdef LUMINARY_ENABLE_DEBUG_API
  struct DeviceMemoryHeader* allocation = _debug_device_memory_allocations;
  while (allocation) {
    struct DeviceMemoryHeader* next = allocation->next;
    free(allocation->name);
    allocation->name = (char*) 0;
    allocation       = next;
  }
  _debug_device_memory_allocations = (struct DeviceMemoryHeader*) 0;
#endif

  mtx_unlock(&_device_memory_mutex);
  mtx_destroy(&_device_memory_mutex);
}

LuminaryResult _device_malloc(DEVICE void** _ptr, size_t size, const char* buf_name, const char* func, uint32_t line) {
  __CHECK_NULL_ARGUMENT(_ptr);

  struct DeviceMemoryHeader* header;
  __FAILURE_HANDLE(_host_malloc((void**) &header, sizeof(struct DeviceMemoryHeader), buf_name, func, line));

#ifdef LUMINARY_ENABLE_DEBUG_API
  memset(header, 0, sizeof(struct DeviceMemoryHeader));
  __FAILURE_HANDLE(_device_memory_set_allocation_name(header, buf_name, func, line));
#endif

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

  _device_memory_add_allocation(header);

  *_ptr = (void**) header;

  return LUMINARY_SUCCESS;
}

LuminaryResult _device_malloc2D(void** _ptr, size_t width_in_bytes, size_t height, const char* buf_name, const char* func, uint32_t line) {
  __CHECK_NULL_ARGUMENT(_ptr);

  struct DeviceMemoryHeader* header;
  __FAILURE_HANDLE(_host_malloc((void**) &header, sizeof(struct DeviceMemoryHeader), buf_name, func, line));

#ifdef LUMINARY_ENABLE_DEBUG_API
  memset(header, 0, sizeof(struct DeviceMemoryHeader));
  __FAILURE_HANDLE(_device_memory_set_allocation_name(header, buf_name, func, line));
#endif

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

  _device_memory_add_allocation(header);

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
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Source is not device memory.");
  }

  if (src_pitch * src_height > src_header->size) {
    __RETURN_ERROR(
      LUMINARY_ERROR_API_EXCEPTION,
      "Download exceeds allocated device memory. %llu bytes are allocated and 2D download would cover %llu bytes.", src_header->size,
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

  CUdevice device;
  CUDA_FAILURE_HANDLE(cuCtxGetDevice(&device));

  if (device != header->device) {
    __RETURN_ERROR(
      LUMINARY_ERROR_API_EXCEPTION, "Device memory allocation belongs to device %u but was attempted to be freed for device %u.",
      header->device, device);
  }

  __FAILURE_HANDLE(_device_memory_remove_allocation(header));

  if (header->ptr) {
    CUDA_FAILURE_HANDLE(cuMemFree(header->ptr));
  }

  header->magic = DEVICE_MEMORY_HEADER_FREED_MAGIC;

#ifdef LUMINARY_ENABLE_DEBUG_API
  free(header->name);
#endif

  __FAILURE_HANDLE(_host_free((void**) &header, buf_name, func, line));

  *ptr = (void*) 0;

  return LUMINARY_SUCCESS;
}

LuminaryResult device_memory_get_total_allocation_size(CUdevice device, size_t* size) {
  __CHECK_NULL_ARGUMENT(size);

  if (device < 0 || device >= LUMINARY_MAX_NUM_DEVICES) {
    __RETURN_ERROR(LUMINARY_ERROR_INVALID_DEVICE, "Invalid CUDA device ordinal was passed.");
  }

  mtx_lock(&_device_memory_mutex);
  *size = total_memory_allocation[device];
  mtx_unlock(&_device_memory_mutex);

  return LUMINARY_SUCCESS;
}

LuminaryResult device_memory_get_allocations(CUdevice device, LuminaryDeviceMemoryAllocationList* list) {
  __CHECK_NULL_ARGUMENT(list);

  if (device < 0 || device >= LUMINARY_MAX_NUM_DEVICES) {
    __RETURN_ERROR(LUMINARY_ERROR_INVALID_DEVICE, "Invalid CUDA device ordinal was passed.");
  }

#ifdef LUMINARY_ENABLE_DEBUG_API
  mtx_lock(&_device_memory_mutex);

  uint64_t allocation_count = 0;
  for (struct DeviceMemoryHeader* allocation = _debug_device_memory_allocations; allocation; allocation = allocation->next) {
    if (allocation->device == device)
      allocation_count++;
  }

  if (allocation_count > SIZE_MAX / sizeof(LuminaryDeviceMemoryAllocation)) {
    mtx_unlock(&_device_memory_mutex);
    __RETURN_ERROR(LUMINARY_ERROR_OUT_OF_MEMORY, "Device allocation snapshot is too large.");
  }

  LuminaryDeviceMemoryAllocation* copied_allocations = (LuminaryDeviceMemoryAllocation*) 0;
  if (allocation_count > 0) {
    copied_allocations = (LuminaryDeviceMemoryAllocation*) calloc((size_t) allocation_count, sizeof(LuminaryDeviceMemoryAllocation));

    if (copied_allocations == (LuminaryDeviceMemoryAllocation*) 0) {
      mtx_unlock(&_device_memory_mutex);
      __RETURN_ERROR(LUMINARY_ERROR_OUT_OF_MEMORY, "Failed to allocate device allocation snapshot.");
    }
  }

  uint64_t copied_count = 0;
  for (struct DeviceMemoryHeader* allocation = _debug_device_memory_allocations; allocation; allocation = allocation->next) {
    if (allocation->device != device)
      continue;

    const size_t name_size = strlen(allocation->name) + 1;
    char* copied_name      = (char*) malloc(name_size);

    if (copied_name == (char*) 0) {
      for (uint64_t cleanup = 0; cleanup < copied_count; cleanup++)
        free((void*) copied_allocations[cleanup].name);

      free(copied_allocations);
      mtx_unlock(&_device_memory_mutex);
      __RETURN_ERROR(LUMINARY_ERROR_OUT_OF_MEMORY, "Failed to copy device allocation snapshot name.");
    }

    memcpy(copied_name, allocation->name, name_size);
    copied_allocations[copied_count].device_address = (uint64_t) allocation->ptr;
    copied_allocations[copied_count].size           = allocation->size;
    copied_allocations[copied_count].name           = copied_name;
    copied_count++;
  }

  list->allocations = copied_allocations;
  list->count       = allocation_count;

  mtx_unlock(&_device_memory_mutex);
#endif

  return LUMINARY_SUCCESS;
}

LuminaryResult device_memory_allocation_list_destroy(LuminaryDeviceMemoryAllocationList* list) {
  __CHECK_NULL_ARGUMENT(list);

#ifdef LUMINARY_ENABLE_DEBUG_API
  for (uint64_t allocation = 0; allocation < list->count; allocation++)
    free((void*) list->allocations[allocation].name);

  free(list->allocations);
  list->allocations = (LuminaryDeviceMemoryAllocation*) 0;
  list->count       = 0;
#endif

  return LUMINARY_SUCCESS;
}

struct DeviceStagingMemoryHeader {
  uint64_t magic;
  uint64_t size;
};

// LUMDEVIS
#define DEVICE_STAGING_MEMORY_HEADER_MAGIC (0x53454D49444D554Cull)
#define DEVICE_STAGING_MEMORY_HEADER_FREED_MAGIC (70ull)

LuminaryResult _device_malloc_staging(STAGING void** ptr, size_t size, DeviceMemoryStagingFlags flags) {
  __CHECK_NULL_ARGUMENT(ptr);

  struct DeviceStagingMemoryHeader* header;

  // CU_MEMHOSTALLOC_WRITECOMBINED allows for fast transfer over PCI-E bus but is very slow to read from on the CPU.
  // Hence we can't use it for stuff like downloading images from the GPU as it would be slow to then use the image on the CPU.
  // TODO: Implement a memcpy STAGING->HOST using the SSE4 instruction MOVNTDQA which allows me to use this flag also for downloads
  // because I can then use this memcpy to retrieve the data on the CPU.
  uint32_t cuda_flags = 0;

  if ((flags & DEVICE_MEMORY_STAGING_FLAG_PCIE_TRANSFER_ONLY) != 0)
    cuda_flags |= CU_MEMHOSTALLOC_WRITECOMBINED;

  if ((flags & DEVICE_MEMORY_STAGING_FLAG_SHARED) != 0)
    cuda_flags |= CU_MEMHOSTALLOC_PORTABLE;

  CUDA_FAILURE_HANDLE(cuMemHostAlloc((void**) &header, size + sizeof(struct DeviceStagingMemoryHeader), cuda_flags));

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
