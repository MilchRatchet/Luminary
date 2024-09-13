#include <string.h>

#include "internal_error.h"
#include "internal_queue.h"
#include "utils.h"

LuminaryResult _queue_create(
  Queue** _queue, size_t size_of_element, size_t num_elements, const char* buf_name, const char* func, uint32_t line) {
  if (!_queue) {
    __RETURN_ERROR(LUMINARY_ERROR_ARGUMENT_NULL, "Queue was NULL.");
  }

  if (size_of_element == 0) {
    __RETURN_ERROR(LUMINARY_ERROR_INVALID_API_ARGUMENT, "Size of element was 0.");
  }

  if (num_elements == 0) {
    __RETURN_ERROR(LUMINARY_ERROR_INVALID_API_ARGUMENT, "Number of elements was 0.");
  }

  Queue* queue;
  __FAILURE_HANDLE(_host_malloc(&queue, sizeof(Queue), buf_name, func, line));

  memset(queue, 0, sizeof(Queue));

  __FAILURE_HANDLE(_host_malloc(&queue->buffer, size_of_element * num_elements, buf_name, func, line));
  queue->element_count     = num_elements;
  queue->element_size      = size_of_element;
  queue->write_ptr         = 0;
  queue->read_ptr          = 0;
  queue->elements_in_queue = 0;

  __FAILURE_HANDLE(mutex_create(&queue->mutex));

  *_queue = queue;

  return LUMINARY_SUCCESS;
}

LuminaryResult queue_push(Queue* queue, void* object) {
  if (!queue) {
    __RETURN_ERROR(LUMINARY_ERROR_ARGUMENT_NULL, "Queue was NULL.");
  }

  if (!queue->buffer) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Queue buffer was NULL.");
  }

  if (!object) {
    __RETURN_ERROR(LUMINARY_ERROR_ARGUMENT_NULL, "Object was NULL.");
  }

  if (queue->elements_in_queue == queue->element_count) {
    __RETURN_ERROR(LUMINARY_ERROR_OUT_OF_MEMORY, "Queue ran out of memory.");
  }

  __FAILURE_HANDLE(mutex_lock(queue->mutex));

  uint8_t* dst_ptr = ((uint8_t) (queue->buffer)) + (queue->write_ptr * queue->element_size);

  memcpy(dst_ptr, object, queue->element_size);

  queue->write_ptr++;
  if (queue->write_ptr >= queue->element_count)
    queue->write_ptr = 0;

  __FAILURE_HANDLE(mutex_unlock(queue->mutex));

  return LUMINARY_SUCCESS;
}

LuminaryResult queue_pop(Queue* queue, void* object, bool* success) {
  if (!queue) {
    __RETURN_ERROR(LUMINARY_ERROR_ARGUMENT_NULL, "Queue was NULL.");
  }

  if (!queue->buffer) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Queue buffer was NULL.");
  }

  if (!object) {
    __RETURN_ERROR(LUMINARY_ERROR_ARGUMENT_NULL, "Object was NULL.");
  }

  if (queue->elements_in_queue == 0) {
    *success = false;
    return LUMINARY_SUCCESS;
  }

  __FAILURE_HANDLE(mutex_lock(queue->mutex));

  uint8_t* src_ptr = ((uint8_t) (queue->buffer)) + (queue->read_ptr * queue->element_size);

  memcpy(object, src_ptr, queue->element_size);

  queue->read_ptr++;
  if (queue->read_ptr >= queue->element_count)
    queue->read_ptr = 0;

  __FAILURE_HANDLE(mutex_unlock(queue->mutex));

  *success = true;

  return LUMINARY_SUCCESS;
}

LuminaryResult _queue_destroy(Queue** queue, const char* buf_name, const char* func, uint32_t line) {
  if (!queue) {
    __RETURN_ERROR(LUMINARY_ERROR_ARGUMENT_NULL, "Queue ptr was NULL.");
  }

  if (!(*queue)) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Queue was NULL.");
  }

  if (!(*queue)->buffer) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Queue buffer was NULL.");
  }

  if ((*queue)->write_ptr != (*queue)->read_ptr) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Queue is not empty.");
  }

  __FAILURE_HANDLE(mutex_destroy(&((*queue)->mutex)));
  __FAILURE_HANDLE(_host_free(&((*queue)->buffer), buf_name, func, line));
  __FAILURE_HANDLE(_host_free(queue, buf_name, func, line));

  return LUMINARY_SUCCESS;
}
