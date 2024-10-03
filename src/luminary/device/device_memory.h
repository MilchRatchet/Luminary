#ifndef LUMINARY_DEVICE_MEMORY_H
#define LUMINARY_DEVICE_MEMORY_H

#include "device_utils.h"

#define DEVICE

#define DEVICE_PTR(device_mem) (*((CUdeviceptr*) (((uint64_t*) (device_mem)) + 1)))

#define device_malloc(ptr, size) _device_malloc((void**) ptr, size, (const char*) #ptr, (const char*) __func__, __LINE__)
#define device_malloc2D(ptr, size) _device_malloc2D((void**) ptr, size, (const char*) #ptr, (const char*) __func__, __LINE__)
#define device_free(ptr) _device_free((void**) ptr, (const char*) #ptr, (const char*) __func__, __LINE__)

LuminaryResult _device_malloc(void** ptr, size_t size, const char* buf_name, const char* func, uint32_t line);
LuminaryResult _device_malloc2D(void** ptr, size_t width_in_bytes, size_t height, const char* buf_name, const char* func, uint32_t line);
LuminaryResult device_upload(DEVICE void* dst, const void* src, size_t dst_offset, size_t size);
LuminaryResult device_memcpy(DEVICE void* dst, DEVICE const void* src, size_t dst_offset, size_t src_offset, size_t size);
LuminaryResult device_download(void* dst, DEVICE const void* src, size_t src_offset, size_t size);
LuminaryResult device_upload2D(DEVICE void* dst, const void* src, size_t src_pitch, size_t src_width, size_t src_height);
LuminaryResult device_memset(DEVICE void* ptr, uint8_t value, size_t offset, size_t size);
LuminaryResult _device_free(DEVICE void** ptr, const char* buf_name, const char* func, uint32_t line);

#endif /* LUMINARY_DEVICE_MEMORY_H */
