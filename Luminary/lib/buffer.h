#ifndef BUFFER_H
#define BUFFER_H

#include "utils.h"

#if __cplusplus
extern "C" {
#endif

size_t device_memory_usage();
size_t device_memory_limit();
void device_set_memory_limit(size_t limit);
void device_malloc(void** buffer, size_t size);
size_t device_malloc_pitch(void** buffer, size_t rowstride, size_t num_rows);
void device_free(void* buffer, size_t size);
void device_buffer_init(DeviceBuffer** buffer);
void device_buffer_free(DeviceBuffer* buffer);
void device_buffer_malloc(DeviceBuffer* buffer, size_t element_size, size_t count);
void device_buffer_upload(DeviceBuffer* buffer, void* data);
void device_buffer_download(DeviceBuffer* buffer, void* dest, size_t size);
void device_buffer_download_full(DeviceBuffer* buffer, void* dest);
void device_buffer_copy(DeviceBuffer* src, DeviceBuffer* dest);
void* device_buffer_get_pointer(DeviceBuffer* buffer);
size_t device_buffer_get_size(DeviceBuffer* buffer);
int device_buffer_is_allocated(DeviceBuffer* buffer);
void device_buffer_destroy(DeviceBuffer* buffer);

#if __cplusplus
}
#endif

#endif /* BUFFER_H */
