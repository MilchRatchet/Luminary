#include "host_output_handler.h"

#include <string.h>

#include "internal_error.h"
#include "internal_walltime.h"

#define OUTPUT_HANDLE_INVALID LUMINARY_OUTPUT_HANDLE_INVALID
#define MIN_OUTPUT_HANDLE_COUNT 4

LuminaryResult output_handler_create(OutputHandler** output) {
  __CHECK_NULL_ARGUMENT(output);

  __FAILURE_HANDLE(host_malloc(output, sizeof(OutputHandler)));
  memset(*output, 0, sizeof(OutputHandler));

  __FAILURE_HANDLE(mutex_create(&(*output)->mutex));
  __FAILURE_HANDLE(array_create(&(*output)->objects, sizeof(OutputObject), 16));
  __FAILURE_HANDLE(array_create(&(*output)->promises, sizeof(OutputObject), 16));

  return LUMINARY_SUCCESS;
}

LuminaryResult output_handler_set_properties(OutputHandler* output, OutputProperties properties) {
  __CHECK_NULL_ARGUMENT(output);

  __FAILURE_HANDLE_LOCK_CRITICAL();
  __FAILURE_HANDLE_CRITICAL(mutex_lock(output->mutex));

  output->properties = properties;

  __FAILURE_HANDLE_UNLOCK_CRITICAL();
  __FAILURE_HANDLE(mutex_unlock(output->mutex));

  __FAILURE_HANDLE_CHECK_CRITICAL();

  return LUMINARY_SUCCESS;
}

LuminaryResult output_handler_add_request(OutputHandler* output, OutputRequestProperties properties, uint32_t* promise_handle) {
  __CHECK_NULL_ARGUMENT(output);
  __CHECK_NULL_ARGUMENT(promise_handle);

  __FAILURE_HANDLE_LOCK_CRITICAL();
  __FAILURE_HANDLE_CRITICAL(mutex_lock(output->mutex));

  uint32_t selected_handle = OUTPUT_HANDLE_INVALID;

  uint32_t num_promises;
  __FAILURE_HANDLE_CRITICAL(array_get_num_elements(output->promises, &num_promises));

  for (uint32_t promise_id = 0; promise_id < num_promises; promise_id++) {
    OutputPromise* promise = output->promises + promise_id;

    if (promise->pending)
      continue;

    selected_handle = promise_id;
    break;
  }

  if (selected_handle == OUTPUT_HANDLE_INVALID) {
    OutputPromise new_object;
    memset(&new_object, 0, sizeof(OutputPromise));

    __FAILURE_HANDLE_CRITICAL(array_push(&output->promises, &new_object));

    selected_handle = num_promises;
  }

  OutputPromise* promise = output->promises + selected_handle;
  promise->pending       = true;
  promise->properties    = properties;
  promise->handle        = OUTPUT_HANDLE_INVALID;

  *promise_handle = selected_handle;

  __FAILURE_HANDLE_UNLOCK_CRITICAL();
  __FAILURE_HANDLE(mutex_unlock(output->mutex));

  __FAILURE_HANDLE_CHECK_CRITICAL();

  return LUMINARY_SUCCESS;
}

LuminaryResult output_handler_acquire(OutputHandler* output, uint32_t* handle) {
  __CHECK_NULL_ARGUMENT(output);
  __CHECK_NULL_ARGUMENT(handle);

  __FAILURE_HANDLE_LOCK_CRITICAL();
  __FAILURE_HANDLE_CRITICAL(mutex_lock(output->mutex));

  uint64_t latest_time_stamp = 0;
  uint32_t latest_handle     = OUTPUT_HANDLE_INVALID;

  uint32_t num_outputs;
  __FAILURE_HANDLE_CRITICAL(array_get_num_elements(output->objects, &num_outputs));

  for (uint32_t output_id = 0; output_id < num_outputs; output_id++) {
    OutputObject* object = output->objects + output_id;

    if (!object->populated)
      continue;

    if (object->time_stamp <= latest_time_stamp)
      continue;

    if (object->descriptor.meta_data.width != output->properties.width || object->descriptor.meta_data.height != output->properties.height)
      continue;

    // Standard acquire function cannot return outputs generated from requests.
    if (object->promise_reference != OUTPUT_HANDLE_INVALID)
      continue;

    latest_handle     = output_id;
    latest_time_stamp = object->time_stamp;
  }

  if (latest_handle != OUTPUT_HANDLE_INVALID) {
    output->objects[latest_handle].reference_count++;
  }

  *handle = latest_handle;

  __FAILURE_HANDLE_UNLOCK_CRITICAL();
  __FAILURE_HANDLE(mutex_unlock(output->mutex));

  __FAILURE_HANDLE_CHECK_CRITICAL();

  return LUMINARY_SUCCESS;
}

LuminaryResult output_handler_release(OutputHandler* output, uint32_t handle) {
  __CHECK_NULL_ARGUMENT(output);

  if (handle == OUTPUT_HANDLE_INVALID)
    return LUMINARY_SUCCESS;

  __FAILURE_HANDLE_LOCK_CRITICAL();
  __FAILURE_HANDLE_CRITICAL(mutex_lock(output->mutex));

  uint32_t num_outputs;
  __FAILURE_HANDLE_CRITICAL(array_get_num_elements(output->objects, &num_outputs));

  if (handle >= num_outputs) {
    __RETURN_ERROR_CRITICAL(LUMINARY_ERROR_API_EXCEPTION, "Invalid output handle %u cannot be released.", handle);
  }

  if (output->objects[handle].reference_count == 0) {
    __RETURN_ERROR_CRITICAL(LUMINARY_ERROR_API_EXCEPTION, "Output handle %u was not previously acquired.", handle);
  }

  output->objects[handle].reference_count--;

  __FAILURE_HANDLE_UNLOCK_CRITICAL();
  __FAILURE_HANDLE(mutex_unlock(output->mutex));

  __FAILURE_HANDLE_CHECK_CRITICAL();

  return LUMINARY_SUCCESS;
}

LuminaryResult output_handler_acquire_from_promise(OutputHandler* output, uint32_t promise_handle, uint32_t* handle) {
  __CHECK_NULL_ARGUMENT(output);
  __CHECK_NULL_ARGUMENT(handle);

  if (promise_handle == OUTPUT_HANDLE_INVALID) {
    *handle = OUTPUT_HANDLE_INVALID;
    return LUMINARY_SUCCESS;
  }

  __FAILURE_HANDLE_LOCK_CRITICAL();
  __FAILURE_HANDLE_CRITICAL(mutex_lock(output->mutex));

  OutputPromise* promise = output->promises + promise_handle;

  uint32_t returned_handle = promise->handle;

  if (returned_handle != OUTPUT_HANDLE_INVALID) {
    output->objects[returned_handle].reference_count++;
    output->objects[returned_handle].promise_reference = OUTPUT_HANDLE_INVALID;
    promise->pending                                   = false;
  }

  *handle = returned_handle;

  __FAILURE_HANDLE_UNLOCK_CRITICAL();
  __FAILURE_HANDLE(mutex_unlock(output->mutex));

  __FAILURE_HANDLE_CHECK_CRITICAL();

  return LUMINARY_SUCCESS;
}

static LuminaryResult _output_handler_get_handle_for_write(OutputHandler* output, OutputDescriptor descriptor, uint32_t* selected_handle) {
  __CHECK_NULL_ARGUMENT(output);

  uint32_t num_outputs;
  __FAILURE_HANDLE(array_get_num_elements(output->objects, &num_outputs));

  *selected_handle              = OUTPUT_HANDLE_INVALID;
  uint64_t earliest_time_stamp  = UINT64_MAX;
  bool selected_handle_is_valid = true;

  uint32_t num_valid_outputs = 0;

  for (uint32_t output_id = 0; output_id < num_outputs; output_id++) {
    OutputObject* object = output->objects + output_id;

    if (object->reference_count)
      continue;

    // Handles that use other dimensions are always eligible for overwriting.
    const bool handle_is_still_valid = ((object->descriptor.meta_data.width == descriptor.meta_data.width)
                                        && (object->descriptor.meta_data.height == descriptor.meta_data.height))
                                       || (descriptor.is_recurring_output == false);

    if (handle_is_still_valid) {
      num_valid_outputs++;

      if (object->time_stamp >= earliest_time_stamp)
        continue;
    }

    *selected_handle    = output_id;
    earliest_time_stamp = object->time_stamp;

    // If we have found an invalid handle, we can safely just overwrite that one.
    if (!handle_is_still_valid) {
      selected_handle_is_valid = false;
      break;
    }
  }

  // Make sure we are always multi-buffered, otherwise we may overwrite the only existing valid buffer.
  if (num_valid_outputs < MIN_OUTPUT_HANDLE_COUNT && selected_handle_is_valid) {
    *selected_handle = OUTPUT_HANDLE_INVALID;
  }

  return LUMINARY_SUCCESS;
}

LuminaryResult output_handler_acquire_new(OutputHandler* output, OutputDescriptor descriptor, uint32_t* handle) {
  __CHECK_NULL_ARGUMENT(output);
  __CHECK_NULL_ARGUMENT(handle);

  __FAILURE_HANDLE_LOCK_CRITICAL();
  __FAILURE_HANDLE_CRITICAL(mutex_lock(output->mutex));

  uint32_t num_outputs;
  __FAILURE_HANDLE_CRITICAL(array_get_num_elements(output->objects, &num_outputs));

  uint32_t selected_handle;
  __FAILURE_HANDLE_CRITICAL(_output_handler_get_handle_for_write(output, descriptor, &selected_handle));

  if (selected_handle == OUTPUT_HANDLE_INVALID) {
    OutputObject new_object;
    memset(&new_object, 0, sizeof(OutputObject));

    __FAILURE_HANDLE_CRITICAL(array_push(&output->objects, &new_object));

    selected_handle = num_outputs;
  }

  OutputObject* selected_output = output->objects + selected_handle;

  const bool requires_allocation = !selected_output->allocated
                                   || (selected_output->descriptor.meta_data.width != descriptor.meta_data.width)
                                   || (selected_output->descriptor.meta_data.height != descriptor.meta_data.height);

  if (requires_allocation) {
    if (selected_output->allocated) {
      __FAILURE_HANDLE_CRITICAL(host_free(&output->objects[selected_handle].descriptor.data));
    }

    __FAILURE_HANDLE_CRITICAL(host_malloc(
      &output->objects[selected_handle].descriptor.data, sizeof(ARGB8) * descriptor.meta_data.width * descriptor.meta_data.height));
  }

  selected_output->populated                      = false;
  selected_output->allocated                      = true;
  selected_output->reference_count                = 1;
  selected_output->descriptor.is_recurring_output = descriptor.is_recurring_output;
  selected_output->descriptor.meta_data           = descriptor.meta_data;
  selected_output->promise_reference              = OUTPUT_HANDLE_INVALID;

  *handle = selected_handle;

  __FAILURE_HANDLE_UNLOCK_CRITICAL();
  __FAILURE_HANDLE(mutex_unlock(output->mutex));

  __FAILURE_HANDLE_CHECK_CRITICAL();

  return LUMINARY_SUCCESS;
}

LuminaryResult output_handler_acquire_from_request_new(OutputHandler* output, OutputDescriptor descriptor, uint32_t* handle) {
  __CHECK_NULL_ARGUMENT(output);
  __CHECK_NULL_ARGUMENT(handle);

  __FAILURE_HANDLE_LOCK_CRITICAL();
  __FAILURE_HANDLE_CRITICAL(mutex_lock(output->mutex));

  uint32_t selected_promise_handle = OUTPUT_HANDLE_INVALID;

  uint32_t num_promises;
  __FAILURE_HANDLE_CRITICAL(array_get_num_elements(output->promises, &num_promises));

  for (uint32_t promise_id = 0; promise_id < num_promises; promise_id++) {
    OutputPromise* promise = output->promises + promise_id;

    if (promise->pending == false)
      continue;

    if (promise->properties.width != descriptor.meta_data.width)
      continue;

    if (promise->properties.height != descriptor.meta_data.height)
      continue;

    if ((promise->properties.sample_count > 0) && promise->properties.sample_count != descriptor.meta_data.sample_count)
      continue;

    selected_promise_handle = promise_id;
    break;
  }

  if (selected_promise_handle) {
    __RETURN_ERROR_CRITICAL(LUMINARY_ERROR_API_EXCEPTION, "Tried to create an output for a request that has no promise.");
  }

  uint32_t num_outputs;
  __FAILURE_HANDLE_CRITICAL(array_get_num_elements(output->objects, &num_outputs));

  uint32_t selected_handle;
  __FAILURE_HANDLE_CRITICAL(_output_handler_get_handle_for_write(output, descriptor, &selected_handle));

  if (selected_handle == OUTPUT_HANDLE_INVALID) {
    OutputObject new_object;
    memset(&new_object, 0, sizeof(OutputObject));

    __FAILURE_HANDLE_CRITICAL(array_push(&output->objects, &new_object));

    selected_handle = num_outputs;
  }

  OutputPromise* promise = output->promises + selected_promise_handle;

  promise->pending = true;
  promise->handle  = selected_handle;

  OutputObject* selected_output = output->objects + selected_handle;

  const bool requires_allocation = !selected_output->allocated
                                   || (selected_output->descriptor.meta_data.width != descriptor.meta_data.width)
                                   || (selected_output->descriptor.meta_data.height != descriptor.meta_data.height);

  if (requires_allocation) {
    if (selected_output->allocated) {
      __FAILURE_HANDLE_CRITICAL(host_free(&output->objects[selected_handle].descriptor.data));
    }

    __FAILURE_HANDLE_CRITICAL(host_malloc(
      &output->objects[selected_handle].descriptor.data, sizeof(ARGB8) * descriptor.meta_data.width * descriptor.meta_data.height));
  }

  selected_output->populated                      = false;
  selected_output->allocated                      = true;
  selected_output->reference_count                = 1;
  selected_output->descriptor.is_recurring_output = descriptor.is_recurring_output;
  selected_output->descriptor.meta_data           = descriptor.meta_data;
  selected_output->promise_reference              = selected_promise_handle;

  *handle = selected_handle;

  __FAILURE_HANDLE_UNLOCK_CRITICAL();
  __FAILURE_HANDLE(mutex_unlock(output->mutex));

  __FAILURE_HANDLE_CHECK_CRITICAL();

  return LUMINARY_SUCCESS;
}

LuminaryResult output_handler_release_new(OutputHandler* output, uint32_t handle) {
  __CHECK_NULL_ARGUMENT(output);

  __FAILURE_HANDLE_LOCK_CRITICAL();
  __FAILURE_HANDLE_CRITICAL(mutex_lock(output->mutex));

  uint32_t num_outputs;
  __FAILURE_HANDLE_CRITICAL(array_get_num_elements(output->objects, &num_outputs));

  if (handle >= num_outputs) {
    __RETURN_ERROR_CRITICAL(LUMINARY_ERROR_API_EXCEPTION, "Invalid output handle %u cannot be released.", handle);
  }

  if (output->objects[handle].reference_count == 0) {
    __RETURN_ERROR_CRITICAL(LUMINARY_ERROR_API_EXCEPTION, "Output handle %u was not previously acquired.", handle);
  }

  output->objects[handle].populated = true;
  output->objects[handle].reference_count--;

  __FAILURE_HANDLE_CRITICAL(_wall_time_get_timestamp(&output->objects[handle].time_stamp));

  __FAILURE_HANDLE_UNLOCK_CRITICAL();
  __FAILURE_HANDLE(mutex_unlock(output->mutex));

  __FAILURE_HANDLE_CHECK_CRITICAL();

  return LUMINARY_SUCCESS;
}

LuminaryResult output_handler_get_image(OutputHandler* output, uint32_t handle, Image* image) {
  __CHECK_NULL_ARGUMENT(output);
  __CHECK_NULL_ARGUMENT(image);

  if (handle == OUTPUT_HANDLE_INVALID) {
    *image = (Image) {.buffer = (uint8_t*) 0, .width = 0, .height = 0, .ld = 0};
    return LUMINARY_SUCCESS;
  }

  __FAILURE_HANDLE_LOCK_CRITICAL();
  __FAILURE_HANDLE_CRITICAL(mutex_lock(output->mutex));

  uint32_t num_outputs;
  __FAILURE_HANDLE_CRITICAL(array_get_num_elements(output->objects, &num_outputs));

  if (handle >= num_outputs) {
    __RETURN_ERROR_CRITICAL(LUMINARY_ERROR_API_EXCEPTION, "Invalid output handle %u.", handle);
  }

  if (output->objects[handle].reference_count == 0) {
    __RETURN_ERROR_CRITICAL(LUMINARY_ERROR_API_EXCEPTION, "Output handle %u was not previously acquired.", handle);
  }

  OutputObject* object = output->objects + handle;

  *image = (Image) {.buffer                 = object->descriptor.data,
                    .width                  = object->descriptor.meta_data.width,
                    .height                 = object->descriptor.meta_data.height,
                    .ld                     = object->descriptor.meta_data.width,
                    .meta_data.time         = object->descriptor.meta_data.time,
                    .meta_data.sample_count = object->descriptor.meta_data.sample_count};

  __FAILURE_HANDLE_UNLOCK_CRITICAL();
  __FAILURE_HANDLE(mutex_unlock(output->mutex));

  __FAILURE_HANDLE_CHECK_CRITICAL();

  return LUMINARY_SUCCESS;
}

LuminaryResult output_handler_destroy(OutputHandler** output) {
  __CHECK_NULL_ARGUMENT(output);
  __CHECK_NULL_ARGUMENT(*output);

  __FAILURE_HANDLE_LOCK_CRITICAL();
  __FAILURE_HANDLE_CRITICAL(mutex_lock((*output)->mutex));

  uint32_t num_outputs;
  __FAILURE_HANDLE_CRITICAL(array_get_num_elements((*output)->objects, &num_outputs));

  for (uint32_t output_id = 0; output_id < num_outputs; output_id++) {
    OutputObject* object = (*output)->objects + output_id;

    if (object->allocated) {
      __FAILURE_HANDLE_CRITICAL(host_free(&object->descriptor.data));
    }
  }

  __FAILURE_HANDLE_UNLOCK_CRITICAL();
  __FAILURE_HANDLE(mutex_unlock((*output)->mutex));

  __FAILURE_HANDLE_CHECK_CRITICAL();

  __FAILURE_HANDLE(mutex_destroy(&(*output)->mutex));
  __FAILURE_HANDLE(array_destroy(&(*output)->objects));
  __FAILURE_HANDLE(array_destroy(&(*output)->promises));

  __FAILURE_HANDLE(host_free(output));

  return LUMINARY_SUCCESS;
}
