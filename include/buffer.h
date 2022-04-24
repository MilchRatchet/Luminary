#ifndef BUFFER_H
#define BUFFER_H

#include "utils.h"

#if __cplusplus
extern "C" {
#endif

#define device_malloc(buffer, size) _device_malloc(buffer, size, (char*) #buffer, (char*) __func__, __LINE__)
#define device_malloc_pitch(buffer, rowstride, num_rows) \
  _device_malloc_pitch(buffer, rowstride, num_rows, (char*) #buffer, (char*) __func__, __LINE__)
#define device_free(buffer, size) _device_free(buffer, size, (char*) #buffer, (char*) __func__, __LINE__)
#define device_upload(dst, src, size) _device_upload(dst, src, size, (char*) #dst, (char*) #src, (char*) __func__, __LINE__)
#define device_buffer_init(buffer) _device_buffer_init(buffer, (char*) #buffer, (char*) __func__, __LINE__)
#define device_buffer_free(buffer) _device_buffer_free(buffer, (char*) #buffer, (char*) __func__, __LINE__)
#define device_buffer_malloc(buffer, element_size, count) \
  _device_buffer_malloc(buffer, element_size, count, (char*) #buffer, (char*) __func__, __LINE__)
#define device_buffer_upload(buffer, data) _device_buffer_upload(buffer, data, (char*) #buffer, (char*) #data, (char*) __func__, __LINE__)
#define device_buffer_download(buffer, dest, size) \
  _device_buffer_download(buffer, dest, size, (char*) #buffer, (char*) #dest, (char*) __func__, __LINE__)
#define device_buffer_download_full(buffer, dest) \
  _device_buffer_download_full(buffer, dest, (char*) #buffer, (char*) #dest, (char*) __func__, __LINE__)
#define device_buffer_copy(src, dest) _device_buffer_copy(src, dest, (char*) #src, (char*) #dest, (char*) __func__, __LINE__)
#define device_buffer_get_pointer(buffer) _device_buffer_get_pointer(buffer, (char*) #buffer, (char*) __func__, __LINE__)
#define device_buffer_get_size(buffer) _device_buffer_get_size(buffer, (char*) #buffer, (char*) __func__, __LINE__)
#define device_buffer_is_allocated(buffer) _device_buffer_is_allocated(buffer, (char*) #buffer, (char*) __func__, __LINE__)
#define device_buffer_destroy(buffer) _device_buffer_destroy(buffer, (char*) #buffer, (char*) __func__, __LINE__)

size_t device_memory_usage();
size_t device_memory_limit();
void device_set_memory_limit(size_t limit);
void _device_malloc(void** buffer, size_t size, char* buf_name, char* func, int line);
size_t _device_malloc_pitch(void** buffer, size_t rowstride, size_t num_rows, char* buf_name, char* func, int line);
void _device_free(void* buffer, size_t size, char* buf_name, char* func, int line);
void _device_upload(void* dst, void* src, size_t size, char* dst_name, char* src_name, char* func, int line);
void _device_buffer_init(DeviceBuffer** buffer, char* buf_name, char* func, int line);
void _device_buffer_free(DeviceBuffer* buffer, char* buf_name, char* func, int line);
void _device_buffer_malloc(DeviceBuffer* buffer, size_t element_size, size_t count, char* buf_name, char* func, int line);
void _device_buffer_upload(DeviceBuffer* buffer, void* data, char* buf_name, char* data_name, char* func, int line);
void _device_buffer_download(DeviceBuffer* buffer, void* dest, size_t size, char* buf_name, char* dest_name, char* func, int line);
void _device_buffer_download_full(DeviceBuffer* buffer, void* dest, char* buf_name, char* dest_name, char* func, int line);
void _device_buffer_copy(DeviceBuffer* src, DeviceBuffer* dest, char* src_name, char* dest_name, char* func, int line);
void* _device_buffer_get_pointer(DeviceBuffer* buffer, char* buf_name, char* func, int line);
size_t _device_buffer_get_size(DeviceBuffer* buffer, char* buf_name, char* func, int line);
int _device_buffer_is_allocated(DeviceBuffer* buffer, char* buf_name, char* func, int line);
void _device_buffer_destroy(DeviceBuffer* buffer, char* buf_name, char* func, int line);

#if __cplusplus
}
#endif

#endif /* BUFFER_H */
