#ifndef LUMINARY_ERROR_REGISTRY_H
#define LUMINARY_ERROR_REGISTRY_H

#include "utils.h"

void _error_registry_init();
void _error_registry_uninit();

void error_registry_set_kind(LuminaryResult result, LuminaryErrorKind kind);
void error_registry_set_message(LuminaryResult result, const char* format, ...);

void _error_registry_add_stacktrace(LuminaryResult result, const char* function_name, const char* file_name, uint64_t line);

#define error_registry_add_stacktrace(result) \
  _error_registry_add_stacktrace(result, (const char*) __func__, (const char*) __FILE__, __LINE__)

void error_registry_allocate(LuminaryResult* result);
void error_registry_register_thread(uint64_t host_id);
void error_registry_get_last(uint64_t host_id, LuminaryResult* result);
void error_registry_get_from_result(LuminaryResult result, LuminaryError** error);

#endif /* LUMINARY_ERROR_REGISTRY_H */
