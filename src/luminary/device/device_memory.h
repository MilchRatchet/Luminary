#ifndef LUMINARY_DEVICE_MEMORY_H
#define LUMINARY_DEVICE_MEMORY_H

#include "device_utils.h"

#define STAGING

#define DEVICE_PTR_GATHER_ABSTRACTION(device_mem, type) (*((type*) (((uint64_t*) (device_mem)) + 1)))

#define DEVICE_PTR(device_mem) DEVICE_PTR_GATHER_ABSTRACTION(device_mem, void*)
#define DEVICE_CUPTR(device_mem) DEVICE_PTR_GATHER_ABSTRACTION(device_mem, CUdeviceptr)

#define device_malloc(ptr, size) _device_malloc((void**) ptr, size, (const char*) #ptr, (const char*) __func__, __LINE__)
#define device_malloc2D(ptr, width_in_bytes, height) \
  _device_malloc2D((void**) ptr, width_in_bytes, height, (const char*) #ptr, (const char*) __func__, __LINE__)
#define device_free(ptr) _device_free((void**) ptr, (const char*) #ptr, (const char*) __func__, __LINE__)

void _device_memory_init(void);
void _device_memory_shutdown(void);

LuminaryResult _device_malloc(DEVICE void** ptr, size_t size, const char* buf_name, const char* func, uint32_t line);
LuminaryResult _device_malloc2D(void** ptr, size_t width_in_bytes, size_t height, const char* buf_name, const char* func, uint32_t line);
LuminaryResult device_upload(DEVICE void* dst, const void* src, size_t dst_offset, size_t size, CUstream stream);
LuminaryResult device_memcpy(DEVICE void* dst, DEVICE const void* src, size_t dst_offset, size_t src_offset, size_t size, CUstream stream);
LuminaryResult device_download(void* dst, DEVICE const void* src, size_t src_offset, size_t size, CUstream stream);
LuminaryResult device_upload2D(DEVICE void* dst, const void* src, size_t src_pitch, size_t src_width, size_t src_height, CUstream stream);
LuminaryResult device_memory_get_pitch(DEVICE const void* ptr, size_t* pitch);
LuminaryResult device_memset(DEVICE void* ptr, uint8_t value, size_t offset, size_t size, CUstream stream);
LuminaryResult _device_free(DEVICE void** ptr, const char* buf_name, const char* func, uint32_t line);

#define device_malloc_staging(ptr, size, upload_only) _device_malloc_staging((void**) ptr, size, upload_only)
#define device_free_staging(ptr) _device_free_staging((void**) ptr)

LuminaryResult _device_malloc_staging(STAGING void** ptr, size_t size, bool upload_only);
LuminaryResult _device_free_staging(STAGING void** ptr);

#endif /* LUMINARY_DEVICE_MEMORY_H */
