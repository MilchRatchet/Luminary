#include "error_registry.h"

#include <signal.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <threads.h>

static _Thread_local uint64_t _thread_local_host_id = LUMINARY_HOST_ID_UNKNOWN;

static _Atomic bool _error_has_occurred = false;

struct ErrorRegistry {
  mtx_t mutex;

  LuminaryError** errors;
  uint64_t num_errors;
  uint64_t num_allocated_errors;
};

// TODO: Replace puts with crash_message!
// TODO: We should always preallocate some errors as we might not be able to once we have errors.

static struct ErrorRegistry _registry;

static void _error_registry_lock() {
  const int retval = mtx_lock(&_registry.mutex);

  if (retval != thrd_success) {
    puts("Failed to lock the mutex for the Luminary error registry.");
    exit(SIGABRT);
  }
}

static void _error_registry_unlock() {
  const int retval = mtx_unlock(&_registry.mutex);

  if (retval != thrd_success) {
    puts("Failed to unlock the mutex for the Luminary error registry.");
    exit(SIGABRT);
  }
}

void _error_registry_init() {
  // This is not allowed to fail.
  const int retval = mtx_init(&_registry.mutex, mtx_recursive);

  if (retval != thrd_success) {
    puts("Failed to initialize the mutex for the Luminary error registry.");
    exit(SIGABRT);
  }

  atomic_store_explicit(&_error_has_occurred, false, memory_order_relaxed);

  _registry.errors = (LuminaryError**) malloc(sizeof(LuminaryError*) * 32);

  _registry.num_errors           = 0;
  _registry.num_allocated_errors = 32;
}

void _error_registry_uninit() {
  _error_registry_lock();

  // TODO: Free stack traces
  for (uint64_t error_id = 0; error_id < _registry.num_errors; error_id++)
    free(_registry.errors[error_id]);

  free(_registry.errors);

  _error_registry_unlock();

  mtx_destroy(&_registry.mutex);

  memset(&_registry, 0, sizeof(struct ErrorRegistry));
}

void error_registry_new_entry(LuminaryResult* result, LuminaryErrorKind kind, const char* format, ...) {
  const uint64_t host_id = _thread_local_host_id;

  _error_registry_lock();

  if (_registry.num_allocated_errors == _registry.num_errors) {
    _registry.num_allocated_errors *= 2;

    LuminaryError** new_ptr = (LuminaryError**) realloc(_registry.errors, sizeof(LuminaryError*) * _registry.num_allocated_errors);

    if (new_ptr == (LuminaryError**) 0) {
      puts("Failed to grow the error list for the Luminary error registry.");
      exit(SIGABRT);
    }

    _registry.errors = new_ptr;
  }

  LuminaryError* new_error = (LuminaryError*) malloc(sizeof(LuminaryError));

  if (new_error == (LuminaryError*) 0) {
    puts("Failed to allocate new error for the Luminary error registry.");
    exit(SIGABRT);
  }

  new_error->host_id = host_id;
  new_error->kind    = kind;
  new_error->message = (const char*) 0;
  new_error->trace   = (LuminaryStackTrace*) 0;

  va_list args;
  va_start(args, format);

  va_list args_copy;
  va_copy(args_copy, args);

  int string_length = vsnprintf((char* const) 0, 0, format, args);

  bool success = (string_length >= 0);

  char* message_string = (char*) 0;
  if (success) {
    // vsnprintf returns size excluding the NULL terminator but for allocation reasons we care about the size including the NULL
    // terminator.
    int required_size = string_length + 1;

    message_string = (char*) malloc(required_size);

    success &= message_string != (char*) 0;

    const int retval = (success) ? vsnprintf(message_string, required_size, format, args_copy) : -1;

    success &= retval >= 0;
  }

  va_end(args_copy);
  va_end(args);

  if (success) {
    new_error->message = message_string;
  }

  if (success == false && message_string != (char*) 0) {
    free(message_string);
  }

  uint64_t error_id = _registry.num_errors++;

  _registry.errors[error_id] = new_error;

  _error_registry_unlock();

  atomic_store_explicit(&_error_has_occurred, true, memory_order_relaxed);

  *result = error_id + 1;
}

void _error_registry_add_stacktrace(LuminaryResult result, const char* function_name, const char* file_name, uint64_t line) {
  if (result == LUMINARY_SUCCESS)
    return;

  _error_registry_lock();

  uint64_t error_id = result - 1;

  if (error_id >= _registry.num_errors) {
    puts("Error id is out of bounds in Luminary error registry.");
    exit(SIGABRT);
  }

  LuminaryStackTrace* new_stacktrace = (LuminaryStackTrace*) malloc(sizeof(LuminaryStackTrace));

  if (new_stacktrace != (LuminaryStackTrace*) 0) {
    new_stacktrace->function_name = function_name;
    new_stacktrace->file_name     = file_name;
    new_stacktrace->line          = line;
    new_stacktrace->caller        = (LuminaryStackTrace*) 0;

    if (_registry.errors[error_id]->trace == (LuminaryStackTrace*) 0) {
      _registry.errors[error_id]->trace = new_stacktrace;
    }
    else {
      LuminaryStackTrace* stacktrace = _registry.errors[error_id]->trace;

      while (stacktrace->caller != (LuminaryStackTrace*) 0)
        stacktrace = stacktrace->caller;

      stacktrace->caller = new_stacktrace;
    }
  }

  _error_registry_unlock();
}

void error_registry_make_host_current(uint64_t host_id) {
  _thread_local_host_id = host_id;
}

void error_registry_get_last(uint64_t host_id, LuminaryResult* result) {
  LuminaryResult return_val = LUMINARY_SUCCESS;

  if (atomic_load_explicit(&_error_has_occurred, memory_order_relaxed) == false) {
    *result = return_val;
    return;
  }

  _error_registry_lock();

  for (uint64_t error_id = 0; error_id < _registry.num_errors; error_id++) {
    const uint64_t error_host_id = _registry.errors[error_id]->host_id;
    if (error_host_id == host_id || error_host_id == LUMINARY_HOST_ID_UNKNOWN)
      return_val = error_id + 1;
  }

  _error_registry_unlock();

  *result = return_val;
}

void error_registry_get_from_result(LuminaryResult result, const LuminaryError** error) {
  if (error == (const LuminaryError**) 0)
    return;

  if (result == LUMINARY_SUCCESS) {
    *error = (const LuminaryError*) 0;
    return;
  }

  _error_registry_lock();

  uint64_t error_id = result - 1;

  const LuminaryError* return_val = (const LuminaryError*) 0;

  if (error_id < _registry.num_errors) {
    return_val = _registry.errors[result - 1];
  }

  _error_registry_unlock();

  *error = return_val;
}
