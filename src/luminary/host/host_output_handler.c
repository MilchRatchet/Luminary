#include "host_output_handler.h"

#include <string.h>

#include "internal_error.h"
#include "internal_walltime.h"

#define OUTPUT_OBJECT_HANDLE_INVALID 0xFFFFFFFF
#define MIN_OUTPUT_HANDLE_COUNT 4

LuminaryResult output_handler_create(OutputHandler** output) {
  __CHECK_NULL_ARGUMENT(output);

  __FAILURE_HANDLE(host_malloc(output, sizeof(OutputHandler)));
  memset(*output, 0, sizeof(OutputHandler));

  __FAILURE_HANDLE(mutex_create(&(*output)->mutex));
  __FAILURE_HANDLE(array_create(&(*output)->objects, sizeof(OutputObject), 16));

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

LuminaryResult output_handler_acquire(OutputHandler* output, uint32_t* handle) {
  __CHECK_NULL_ARGUMENT(output);
  __CHECK_NULL_ARGUMENT(handle);

  __FAILURE_HANDLE_LOCK_CRITICAL();
  __FAILURE_HANDLE_CRITICAL(mutex_lock(output->mutex));

  uint64_t latest_time_stamp = 0;
  uint32_t latest_handle     = OUTPUT_OBJECT_HANDLE_INVALID;

  uint32_t num_outputs;
  __FAILURE_HANDLE_CRITICAL(array_get_num_elements(output->objects, &num_outputs));

  for (uint32_t output_id = 0; output_id < num_outputs; output_id++) {
    OutputObject* object = output->objects + output_id;

    if (!object->populated)
      continue;

    if (object->time_stamp <= latest_time_stamp)
      continue;

    if (object->width != output->properties.width || object->height != output->properties.height)
      continue;

    latest_handle     = output_id;
    latest_time_stamp = object->time_stamp;
  }

  if (latest_handle != OUTPUT_OBJECT_HANDLE_INVALID) {
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

  if (handle == OUTPUT_OBJECT_HANDLE_INVALID)
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

static LuminaryResult _output_handler_get_handle_for_write(
  OutputHandler* output, uint32_t width, uint32_t height, uint32_t* selected_handle) {
  __CHECK_NULL_ARGUMENT(output);

  uint32_t num_outputs;
  __FAILURE_HANDLE(array_get_num_elements(output->objects, &num_outputs));

  *selected_handle              = OUTPUT_OBJECT_HANDLE_INVALID;
  uint64_t earliest_time_stamp  = UINT64_MAX;
  bool selected_handle_is_valid = true;

  uint32_t num_valid_outputs = 0;

  for (uint32_t output_id = 0; output_id < num_outputs; output_id++) {
    OutputObject* object = output->objects + output_id;

    if (object->reference_count)
      continue;

    // Handles that use other dimensions are always eligible for overwriting.
    bool handle_is_still_valid = (object->width == width) && (object->height == height);
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
    *selected_handle = OUTPUT_OBJECT_HANDLE_INVALID;
  }

  return LUMINARY_SUCCESS;
}

LuminaryResult output_handler_acquire_new(OutputHandler* output, uint32_t width, uint32_t height, uint32_t* handle) {
  __CHECK_NULL_ARGUMENT(output);
  __CHECK_NULL_ARGUMENT(handle);

  __FAILURE_HANDLE_LOCK_CRITICAL();
  __FAILURE_HANDLE_CRITICAL(mutex_lock(output->mutex));

  uint32_t num_outputs;
  __FAILURE_HANDLE_CRITICAL(array_get_num_elements(output->objects, &num_outputs));

  uint32_t selected_handle;
  __FAILURE_HANDLE_CRITICAL(_output_handler_get_handle_for_write(output, width, height, &selected_handle));

  if (selected_handle == OUTPUT_OBJECT_HANDLE_INVALID) {
    OutputObject new_object;
    memset(&new_object, 0, sizeof(OutputObject));

    __FAILURE_HANDLE_CRITICAL(array_push(&output->objects, &new_object));

    selected_handle = num_outputs;
  }

  OutputObject* selected_output = output->objects + selected_handle;

  bool requires_allocation = !selected_output->allocated || (selected_output->width != width) || (selected_output->height != height);

  if (requires_allocation) {
    if (selected_output->allocated) {
      __FAILURE_HANDLE_CRITICAL(host_free(&output->objects[selected_handle].data));
    }

    __FAILURE_HANDLE_CRITICAL(host_malloc(&output->objects[selected_handle].data, sizeof(XRGB8) * width * height));
  }

  selected_output->populated       = false;
  selected_output->allocated       = true;
  selected_output->reference_count = 1;
  selected_output->width           = width;
  selected_output->height          = height;

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

LuminaryResult output_handler_get_buffer(OutputHandler* output, uint32_t handle, void** buffer) {
  __CHECK_NULL_ARGUMENT(output);
  __CHECK_NULL_ARGUMENT(buffer);

  if (handle == OUTPUT_OBJECT_HANDLE_INVALID) {
    *buffer = (void*) 0;
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

  *buffer = output->objects[handle].data;

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
      __FAILURE_HANDLE_CRITICAL(host_free(&object->data));
    }
  }

  __FAILURE_HANDLE_UNLOCK_CRITICAL();
  __FAILURE_HANDLE(mutex_unlock((*output)->mutex));

  __FAILURE_HANDLE_CHECK_CRITICAL();

  __FAILURE_HANDLE(mutex_destroy(&(*output)->mutex));
  __FAILURE_HANDLE(array_destroy(&(*output)->objects));

  __FAILURE_HANDLE(host_free(output));

  return LUMINARY_SUCCESS;
}
