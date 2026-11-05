#ifndef LUMINARY_API_STACK_TRACE_H
#define LUMINARY_API_STACK_TRACE_H

#include <luminary/error.h>
#include <stddef.h>

struct LuminaryNativeStackTrace {
  char* text;
  size_t length;
} typedef LuminaryNativeStackTrace;

/*
 * Captures and symbolizes a bounded snapshot of the calling thread's native stack.
 * The destination must be zero-initialized and must not own an existing snapshot.
 * The returned text is null-terminated and remains owned by the result until destroyed.
 * Luminary must have been initialized before calling this function.
 */
LUMINARY_API LuminaryResult luminary_get_native_stack_trace(LuminaryNativeStackTrace* stack_trace);

/* Frees the returned text and clears the result. */
LUMINARY_API LuminaryResult luminary_native_stack_trace_destroy(LuminaryNativeStackTrace* stack_trace);

#endif /* LUMINARY_API_STACK_TRACE_H */
