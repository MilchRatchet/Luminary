#ifndef LUMINARY_DEVICE_MEMORY_H
#define LUMINARY_DEVICE_MEMORY_H

#include "device_utils.h"

#define DEVICE

#define DEVICE_PTR(device_mem) (*((CUdeviceptr*) (((uint64_t*) device_mem) + 1)))

#define device_malloc(ptr, size) _host_malloc((void**) ptr, size, (const char*) #ptr, (const char*) __func__, __LINE__)
#define device_free(ptr) _host_free((void**) ptr, (const char*) #ptr, (const char*) __func__, __LINE__)

LuminaryResult _device_malloc(void** ptr, size_t size, const char* buf_name, const char* func, uint32_t line);
LuminaryResult device_upload(DEVICE void* dst, const void* src, size_t dst_offset, size_t size);
LuminaryResult device_memcpy(DEVICE void* dst, DEVICE const void* src, size_t dst_offset, size_t src_offset, size_t size);
LuminaryResult device_download(void* dst, DEVICE const void* src, size_t src_offset, size_t size);
LuminaryResult device_memset(DEVICE void* ptr, uint8_t value, size_t offset, size_t size);
LuminaryResult _device_free(DEVICE void** ptr, const char* buf_name, const char* func, uint32_t line);

#endif /* LUMINARY_DEVICE_MEMORY_H */
