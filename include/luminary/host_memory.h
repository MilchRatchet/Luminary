#ifndef LUMINARY_HOST_MEMORY_H
#define LUMINARY_HOST_MEMORY_H

#include "error.h"

#define host_malloc(ptr, size) _host_malloc((void**) ptr, size, (const char*) #ptr, (const char*) __func__, __LINE__)
#define host_realloc(ptr, size) _host_realloc((void**) ptr, size, (const char*) #ptr, (const char*) __func__, __LINE__)
#define host_free(ptr) _host_free((void**) ptr, (const char*) #ptr, (const char*) __func__, __LINE__)

LuminaryResult _host_malloc(void** ptr, size_t size, const char* buf_name, const char* func, uint32_t line);
LuminaryResult _host_realloc(void** ptr, size_t size, const char* buf_name, const char* func, uint32_t line);
LuminaryResult _host_free(void** ptr, const char* buf_name, const char* func, uint32_t line);

#endif /* LUMINARY_HOST_MEMORY_H */
