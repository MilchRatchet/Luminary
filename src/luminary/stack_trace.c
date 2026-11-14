#if defined(__linux__) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif

#include <inttypes.h>
#include <luminary/stack_trace.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "internal_error.h"
#include "internal_stack_trace.h"

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

// Windows must be included before these
#include <dbghelp.h>
#include <threads.h>
#elif defined(__linux__)
#include <dlfcn.h>
#include <execinfo.h>
#else
#error "Native stack traces are supported on Windows and Linux only."
#endif

#define STACK_TRACE_MAX_FRAMES 64

struct StackTraceBuffer {
  char* data;
  size_t length;
  size_t capacity;
} typedef StackTraceBuffer;

#if defined(_WIN32)
static mtx_t _stack_trace_mutex;
static bool _stack_trace_mutex_initialized;
static bool _stack_trace_symbols_initialized;

void _stack_trace_init(void) {
  if (mtx_init(&_stack_trace_mutex, mtx_plain) != thrd_success)
    return;

  _stack_trace_mutex_initialized = true;

  if (mtx_lock(&_stack_trace_mutex) != thrd_success)
    return;

  SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS);
  _stack_trace_symbols_initialized = SymInitialize(GetCurrentProcess(), (PCSTR) 0, TRUE) != FALSE;

  mtx_unlock(&_stack_trace_mutex);
}

void _stack_trace_shutdown(void) {
  if (_stack_trace_mutex_initialized == false)
    return;

  if (mtx_lock(&_stack_trace_mutex) == thrd_success) {
    if (_stack_trace_symbols_initialized)
      SymCleanup(GetCurrentProcess());

    _stack_trace_symbols_initialized = false;
    mtx_unlock(&_stack_trace_mutex);
  }

  mtx_destroy(&_stack_trace_mutex);
  _stack_trace_mutex_initialized = false;
}
#else
void _stack_trace_init(void) {
}
void _stack_trace_shutdown(void) {
}
#endif

static bool _stack_trace_buffer_append(StackTraceBuffer* buffer, const char* format, ...) {
  va_list args;
  va_start(args, format);

  va_list args_copy;
  va_copy(args_copy, args);

  const int formatted_size = vsnprintf((char*) 0, 0, format, args_copy);
  va_end(args_copy);

  if (formatted_size < 0) {
    va_end(args);
    return false;
  }

  if (buffer->length > SIZE_MAX - (size_t) formatted_size - 1) {
    va_end(args);
    return false;
  }

  const size_t required_capacity = buffer->length + (size_t) formatted_size + 1;

  if (required_capacity > buffer->capacity) {
    size_t new_capacity = buffer->capacity == 0 ? 512 : buffer->capacity;

    while (new_capacity < required_capacity) {
      if (new_capacity > SIZE_MAX / 2) {
        new_capacity = required_capacity;
        break;
      }

      new_capacity *= 2;
    }

    char* new_data = realloc(buffer->data, new_capacity);

    if (new_data == (char*) 0) {
      va_end(args);
      return false;
    }

    buffer->data     = new_data;
    buffer->capacity = new_capacity;
  }

  const int write_result = vsnprintf(buffer->data + buffer->length, buffer->capacity - buffer->length, format, args);
  va_end(args);

  if (write_result < 0)
    return false;

  buffer->length += (size_t) write_result;
  return true;
}

#if defined(_WIN32)
static bool _stack_trace_buffer_append_frame(StackTraceBuffer* buffer, size_t frame_index, void* frame, bool symbols_available) {
  const char* function_name = "??";
  const char* module_name   = "unknown";
  uintptr_t frame_offset    = (uintptr_t) frame;
  bool symbol_found         = false;

  char symbol_storage[sizeof(SYMBOL_INFO) + MAX_SYM_NAME] = {0};
  SYMBOL_INFO* symbol_info                                = (SYMBOL_INFO*) symbol_storage;
  symbol_info->SizeOfStruct                               = sizeof(SYMBOL_INFO);
  symbol_info->MaxNameLen                                 = MAX_SYM_NAME;

  IMAGEHLP_MODULE64 module_info = {0};
  module_info.SizeOfStruct      = sizeof(IMAGEHLP_MODULE64);

  const DWORD64 frame_address = (DWORD64) (uintptr_t) frame;

  if (symbols_available) {
    DWORD64 symbol_displacement = 0;
    if (SymFromAddr(GetCurrentProcess(), frame_address, &symbol_displacement, symbol_info)) {
      function_name = symbol_info->Name;
      frame_offset  = (uintptr_t) symbol_displacement;
      symbol_found  = true;
    }

    if (SymGetModuleInfo64(GetCurrentProcess(), frame_address, &module_info)) {
      if (module_info.ModuleName[0] != '\0')
        module_name = module_info.ModuleName;

      if (symbol_found == false && frame_address >= module_info.BaseOfImage)
        frame_offset = (uintptr_t) (frame_address - module_info.BaseOfImage);
    }
  }

  return _stack_trace_buffer_append(
    buffer, "#%zu %s (%s+0x%llx) [%p]\n", frame_index, function_name, module_name, (unsigned long long) frame_offset, frame);
}
#elif defined(__linux__)
static bool _stack_trace_buffer_append_frame(StackTraceBuffer* buffer, size_t frame_index, void* frame) {
  const char* function_name = "??";
  const char* module_name   = "unknown";
  uintptr_t frame_offset    = (uintptr_t) frame;

  Dl_info symbol_info = {0};
  if (dladdr(frame, &symbol_info) != 0) {
    if (symbol_info.dli_sname != (const char*) 0)
      function_name = symbol_info.dli_sname;

    if (symbol_info.dli_fname != (const char*) 0)
      module_name = symbol_info.dli_fname;

    if (symbol_info.dli_saddr != (void*) 0)
      frame_offset = (uintptr_t) frame - (uintptr_t) symbol_info.dli_saddr;
    else if (symbol_info.dli_fbase != (void*) 0)
      frame_offset = (uintptr_t) frame - (uintptr_t) symbol_info.dli_fbase;
  }

  return _stack_trace_buffer_append(
    buffer, "#%zu %s (%s+0x%" PRIxPTR ") [%p]\n", frame_index, function_name, module_name, frame_offset, frame);
}
#endif

LuminaryResult luminary_get_native_stack_trace(LuminaryNativeStackTrace* stack_trace) {
  __CHECK_NULL_ARGUMENT(stack_trace);

  if (stack_trace->text != (char*) 0 || stack_trace->length != 0)
    __RETURN_ERROR(LUMINARY_ERROR_INVALID_API_ARGUMENT, "Stack trace destination must be empty.");

  void* frames[STACK_TRACE_MAX_FRAMES];
  size_t frame_count = 0;

#if defined(_WIN32)
  frame_count = CaptureStackBackTrace(1, STACK_TRACE_MAX_FRAMES, frames, (PDWORD) 0);
#elif defined(__linux__)
  void* captured_frames[STACK_TRACE_MAX_FRAMES + 2];
  const int captured_frame_count = backtrace(captured_frames, STACK_TRACE_MAX_FRAMES + 2);

  if (captured_frame_count > 2) {
    frame_count = (size_t) captured_frame_count - 2;
    if (frame_count > STACK_TRACE_MAX_FRAMES)
      frame_count = STACK_TRACE_MAX_FRAMES;

    for (size_t frame = 0; frame < frame_count; frame++)
      frames[frame] = captured_frames[frame + 2];
  }
#endif

  StackTraceBuffer buffer = {0};

  if (frame_count == 0 && _stack_trace_buffer_append(&buffer, "No stack frames captured.\n") == false) {
    free(buffer.data);
    __RETURN_ERROR(LUMINARY_ERROR_OUT_OF_MEMORY, "Failed to format native stack trace.");
  }

#if defined(_WIN32)
  bool mutex_locked      = false;
  bool symbols_available = false;

  if (_stack_trace_mutex_initialized && mtx_lock(&_stack_trace_mutex) == thrd_success) {
    mutex_locked      = true;
    symbols_available = _stack_trace_symbols_initialized;
  }
#endif

  for (size_t frame = 0; frame < frame_count; frame++) {
#if defined(_WIN32)
    const bool success = _stack_trace_buffer_append_frame(&buffer, frame, frames[frame], symbols_available);
#elif defined(__linux__)
    const bool success = _stack_trace_buffer_append_frame(&buffer, frame, frames[frame]);
#endif

    if (success == false) {
#if defined(_WIN32)
      if (mutex_locked)
        mtx_unlock(&_stack_trace_mutex);
#endif
      free(buffer.data);
      __RETURN_ERROR(LUMINARY_ERROR_OUT_OF_MEMORY, "Failed to format native stack trace.");
    }
  }

#if defined(_WIN32)
  if (mutex_locked)
    mtx_unlock(&_stack_trace_mutex);
#endif

  stack_trace->text   = buffer.data;
  stack_trace->length = buffer.length;

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_native_stack_trace_destroy(LuminaryNativeStackTrace* stack_trace) {
  __CHECK_NULL_ARGUMENT(stack_trace);

  free(stack_trace->text);
  stack_trace->text   = (char*) 0;
  stack_trace->length = 0;

  return LUMINARY_SUCCESS;
}
