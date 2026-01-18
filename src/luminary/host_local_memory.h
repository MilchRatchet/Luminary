#ifndef LUMINARY_HOST_LOCAL_MEMORY_H
#define LUMINARY_HOST_LOCAL_MEMORY_H

#include "utils.h"

#define LOCAL

LuminaryResult _host_memory_local_init();
LuminaryResult _host_memory_local_shutdown();

#define host_malloc_local(ptr, size) _host_malloc_local((void**) (ptr), (size))
#define host_realloc_local(ptr, size) _host_realloc_local((void**) (ptr), (size))
#define host_free_local(ptr) _host_free_local((void**) (ptr))

LuminaryResult _host_malloc_local(void** ptr, size_t size);
LuminaryResult _host_realloc_local(void** ptr, size_t size);
LuminaryResult _host_free_local(void** ptr);

#endif /* LUMINARY_HOST_LOCAL_MEMORY_H */
