#include "error_registry.h"

#include <signal.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <threads.h>

static _Atomic uint64_t _thread_local_allocator_counter = 0;
static _Thread_local uint64_t _thread_local_id          = (uint64_t) -1;

struct HostThreadMapEntry {
  uint64_t host_id;
  uint64_t thread_id;
};

struct ErrorRegistry {
  mtx_t mutex;

  LuminaryError** errors;
  uint64_t num_errors;
  uint64_t num_allocated_errors;

  struct HostThreadMapEntry* host_thread_map;
  uint64_t num_host_thread_map_entries;
  uint64_t num_allocated_host_thread_map_entries;
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

  _registry.errors = (LuminaryError**) malloc(sizeof(LuminaryError*) * 32);

  _registry.num_errors           = 0;
  _registry.num_allocated_errors = 32;

  _registry.host_thread_map = (struct HostThreadMapEntry*) malloc(sizeof(struct HostThreadMapEntry) * 128);

  _registry.num_host_thread_map_entries           = 0;
  _registry.num_allocated_host_thread_map_entries = 128;
}

void _error_registry_uninit() {
  _error_registry_lock();

  // TODO: Free stack traces
  for (uint64_t error_id = 0; error_id < _registry.num_errors; error_id++)
    free(_registry.errors[error_id]);

  free(_registry.errors);
  free(_registry.host_thread_map);

  _error_registry_unlock();

  mtx_destroy(&_registry.mutex);

  memset(&_registry, 0, sizeof(struct ErrorRegistry));
}

void error_registry_set_kind(LuminaryResult result, LuminaryErrorKind kind) {
  if (result == LUMINARY_SUCCESS)
    return;

  _error_registry_lock();

  uint64_t error_id = result - 1;

  if (error_id >= _registry.num_errors) {
    puts("Error id is out of bounds in Luminary error registry.");
    exit(SIGABRT);
  }

  _registry.errors[error_id]->kind = kind;

  _error_registry_unlock();
}

void error_registry_set_message(LuminaryResult result, const char* format, ...) {
  if (result == LUMINARY_SUCCESS)
    return;

  _error_registry_lock();

  uint64_t error_id = result - 1;

  if (error_id >= _registry.num_errors) {
    puts("Error id is out of bounds in Luminary error registry.");
    exit(SIGABRT);
  }

  va_list args;
  va_start(args, format);

  va_list args_copy;
  va_copy(args_copy, args);

  int string_length = vsnprintf((char* const) 0, 0, format, args);

  if (string_length < 0) {
    puts("vsnprintf returned an error in Luminary error registry.");
    exit(SIGABRT);
  }

  // vsnprintf returns size excluding the NULL terminator but for allocation reasons we care about the size including the NULL
  // terminator.
  int required_size = string_length + 1;

  bool success = true;

  char* message_string = (char*) malloc(required_size);

  success &= message_string != (char*) 0;

  const int retval = (success) ? vsnprintf(message_string, required_size, format, args_copy) : -1;

  success &= retval >= 0;

  va_end(args_copy);
  va_end(args);

  if (success) {
    _registry.errors[error_id]->message = message_string;
  }

  if (success == false && message_string != (char*) 0) {
    free(message_string);
  }

  _error_registry_unlock();
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

void error_registry_allocate(LuminaryResult* result) {
  const uint64_t thread_id = _thread_local_id;

  if (thread_id == (uint64_t) -1) {
    puts("Thread has not been registered in Luminary error registry.");
    exit(SIGABRT);
  }

  _error_registry_lock();

  uint64_t host_id = (uint64_t) -1;

  for (uint64_t entry_id = 0; entry_id < _registry.num_host_thread_map_entries; entry_id++) {
    if (_registry.host_thread_map[entry_id].thread_id == thread_id) {
      host_id = _registry.host_thread_map[entry_id].host_id;
      break;
    }
  }

  if (host_id == (uint64_t) -1) {
    puts("Failed to find host corresponding to failed thread in Luminary error registry.");
    exit(SIGABRT);
  }

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
  new_error->kind    = LUMINARY_ERROR_UNKNOWN;
  new_error->message = (const char*) 0;
  new_error->trace   = (LuminaryStackTrace*) 0;

  uint64_t error_id = _registry.num_errors++;

  _registry.errors[error_id] = new_error;

  _error_registry_unlock();

  *result = error_id + 1;
}

void error_registry_register_thread(uint64_t host_id) {
  _error_registry_lock();

  if (_thread_local_id != (uint64_t) -1) {
    puts("Thread was registered twice in Luminary error registry.");
    exit(SIGABRT);
  }

  if (_registry.num_host_thread_map_entries == _registry.num_allocated_host_thread_map_entries) {
    _registry.num_allocated_host_thread_map_entries *= 2;

    struct HostThreadMapEntry* new_ptr = (struct HostThreadMapEntry*) realloc(
      _registry.host_thread_map, sizeof(struct HostThreadMapEntry) * _registry.num_allocated_host_thread_map_entries);

    if (new_ptr == (struct HostThreadMapEntry*) 0) {
      puts("Failed to grow the host thread map for the Luminary error registry.");
      exit(SIGABRT);
    }

    _registry.host_thread_map = new_ptr;
  }

  const uint64_t thread_id = atomic_fetch_add(&_thread_local_allocator_counter, 1);

  _thread_local_id = thread_id;

  struct HostThreadMapEntry new_entry = {.host_id = host_id, .thread_id = thread_id};

  for (uint64_t entry_id = 0; entry_id < _registry.num_host_thread_map_entries; entry_id++) {
    if (_registry.host_thread_map[entry_id].thread_id == thread_id) {
      if (_registry.host_thread_map[entry_id].host_id == host_id) {
        puts("Thread was registered twice for the same hosts in the Luminary error registry.");
      }
      else {
        puts("Thread was registered twice for different hosts in the Luminary error registry.");
      }

      exit(SIGABRT);
    }
  }

  _registry.host_thread_map[_registry.num_host_thread_map_entries++] = new_entry;

  _error_registry_unlock();
}

void error_registry_get_last(uint64_t host_id, LuminaryResult* result) {
  if (_registry.num_errors == 0) {
    *result = LUMINARY_SUCCESS;
    return;
  }

  LuminaryResult return_val = LUMINARY_SUCCESS;

  _error_registry_lock();

  for (uint64_t error_id = 0; error_id < _registry.num_errors; error_id++) {
    if (_registry.errors[error_id]->host_id == host_id)
      return_val = error_id + 1;
  }

  _error_registry_unlock();

  *result = return_val;
}

void error_registry_get_from_result(LuminaryResult result, LuminaryError** error) {
  if (result == LUMINARY_SUCCESS) {
    *error = (LuminaryError*) 0;
  }

  *error = _registry.errors[result - 1];
}
