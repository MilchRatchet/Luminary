#ifndef LUMINARY_HOST_OUTPUT_HANDLER_H
#define LUMINARY_HOST_OUTPUT_HANDLER_H

#include "mutex.h"
#include "utils.h"

struct OutputObject {
  bool populated;
  bool allocated;
  uint64_t time_stamp;
  uint32_t reference_count;
  uint32_t width;
  uint32_t height;
  void* data;
  uint32_t promise_reference;
} typedef OutputObject;

struct OutputPromise {
  bool populated;
  OutputRequestProperties properties;
  uint32_t handle;
} typedef OutputPromise;

struct OutputHandler {
  OutputProperties properties;
  Mutex* mutex;
  ARRAY OutputObject* objects;
  ARRAY OutputPromise* promises;
} typedef OutputHandler;

struct OutputCopyHandle {
  bool is_recurring;
  uint32_t sample_count;
  uint32_t width;
  uint32_t height;
  void* src;
} typedef OutputCopyHandle;

LuminaryResult output_handler_create(OutputHandler** output);
LuminaryResult output_handler_set_properties(OutputHandler* output, OutputProperties properties);
LuminaryResult output_handler_add_request(OutputHandler* output, OutputRequestProperties properties, uint32_t* promise_handle);
LuminaryResult output_handler_acquire(OutputHandler* output, uint32_t* handle);
LuminaryResult output_handler_release(OutputHandler* output, uint32_t handle);
LuminaryResult output_handler_acquire_from_promise(OutputHandler* output, uint32_t promise_handle, uint32_t* handle);
LuminaryResult output_handler_acquire_new(OutputHandler* output, uint32_t width, uint32_t height, uint32_t* handle);
LuminaryResult output_handler_acquire_from_request_new(
  OutputHandler* output, uint32_t width, uint32_t height, uint32_t sample_count, uint32_t* handle);
LuminaryResult output_handler_release_new(OutputHandler* output, uint32_t handle);
LuminaryResult output_handler_get_buffer(OutputHandler* output, uint32_t handle, void** buffer);
LuminaryResult output_handler_destroy(OutputHandler** output);

#endif /* LUMINARY_HOST_OUTPUT_HANDLER_H */
