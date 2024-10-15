#include <string.h>

#include "cond_var.h"
#include "internal_error.h"
#include "mutex.h"
#include "utils.h"

struct LuminaryQueue {
  void* buffer;
  size_t element_count;
  size_t element_size;
  size_t read_ptr;
  size_t write_ptr;
  size_t elements_in_queue;
  Mutex* mutex;
  ConditionVariable* cond_var;
  bool shutdown;
};

LuminaryResult _queue_create(
  Queue** _queue, size_t size_of_element, size_t num_elements, const char* buf_name, const char* func, uint32_t line) {
  __CHECK_NULL_ARGUMENT(_queue);

  if (size_of_element == 0) {
    __RETURN_ERROR(LUMINARY_ERROR_INVALID_API_ARGUMENT, "Size of element is 0.");
  }

  if (num_elements == 0) {
    __RETURN_ERROR(LUMINARY_ERROR_INVALID_API_ARGUMENT, "Number of elements is 0.");
  }

  Queue* queue;
  __FAILURE_HANDLE(_host_malloc((void**) &queue, sizeof(Queue), buf_name, func, line));

  memset(queue, 0, sizeof(Queue));

  __FAILURE_HANDLE(_host_malloc((void**) &queue->buffer, size_of_element * num_elements, buf_name, func, line));
  queue->element_count     = num_elements;
  queue->element_size      = size_of_element;
  queue->write_ptr         = 0;
  queue->read_ptr          = 0;
  queue->elements_in_queue = 0;
  queue->shutdown          = false;

  __FAILURE_HANDLE(mutex_create(&queue->mutex));
  __FAILURE_HANDLE(condition_variable_create(&queue->cond_var));

  *_queue = queue;

  return LUMINARY_SUCCESS;
}

LuminaryResult queue_push(Queue* queue, void* object) {
  __CHECK_NULL_ARGUMENT(queue);
  __CHECK_NULL_ARGUMENT(object);

  if (!queue->buffer) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Queue buffer is NULL.");
  }

  if (queue->elements_in_queue == queue->element_count) {
    __RETURN_ERROR(LUMINARY_ERROR_OUT_OF_MEMORY, "Queue ran out of memory.");
  }

  __FAILURE_HANDLE_LOCK_CRITICAL();
  __FAILURE_HANDLE_CRITICAL(mutex_lock(queue->mutex));

  uint8_t* dst_ptr = ((uint8_t*) (queue->buffer)) + (queue->write_ptr * queue->element_size);

  memcpy(dst_ptr, object, queue->element_size);

  queue->write_ptr++;
  if (queue->write_ptr >= queue->element_count)
    queue->write_ptr = 0;

  queue->elements_in_queue++;

  __FAILURE_HANDLE_CRITICAL(condition_variable_signal(queue->cond_var));

  __FAILURE_HANDLE_UNLOCK_CRITICAL();
  __FAILURE_HANDLE(mutex_unlock(queue->mutex));

  __FAILURE_HANDLE_CHECK_CRITICAL();

  return LUMINARY_SUCCESS;
}

LuminaryResult queue_pop(Queue* queue, void* object, bool* success) {
  __CHECK_NULL_ARGUMENT(queue);

  if (!queue->buffer) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Queue buffer is NULL.");
  }

  if (!object) {
    __RETURN_ERROR(LUMINARY_ERROR_ARGUMENT_NULL, "Object is NULL.");
  }

  if (queue->elements_in_queue == 0) {
    *success = false;
    return LUMINARY_SUCCESS;
  }

  __FAILURE_HANDLE(mutex_lock(queue->mutex));

  uint8_t* src_ptr = ((uint8_t*) (queue->buffer)) + (queue->read_ptr * queue->element_size);

  memcpy(object, src_ptr, queue->element_size);

  queue->read_ptr++;
  if (queue->read_ptr >= queue->element_count)
    queue->read_ptr = 0;

  queue->elements_in_queue--;

  __FAILURE_HANDLE(mutex_unlock(queue->mutex));

  *success = true;

  return LUMINARY_SUCCESS;
}

LuminaryResult queue_pop_blocking(Queue* queue, void* object, bool* success) {
  __CHECK_NULL_ARGUMENT(queue);
  __CHECK_NULL_ARGUMENT(object);

  if (!queue->buffer) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Queue buffer is NULL.");
  }

  __FAILURE_HANDLE_LOCK_CRITICAL();
  __FAILURE_HANDLE_CRITICAL(mutex_lock(queue->mutex));

  while (queue->elements_in_queue == 0) {
    if (queue->shutdown) {
      __FAILURE_HANDLE_CRITICAL(mutex_unlock(queue->mutex));

      *success = false;

      return LUMINARY_SUCCESS;
    }

    __FAILURE_HANDLE_CRITICAL(condition_variable_wait(queue->cond_var, queue->mutex));
  }

  uint8_t* src_ptr = ((uint8_t*) (queue->buffer)) + (queue->read_ptr * queue->element_size);

  memcpy(object, src_ptr, queue->element_size);

  queue->read_ptr++;
  if (queue->read_ptr >= queue->element_count)
    queue->read_ptr = 0;

  queue->elements_in_queue--;

  __FAILURE_HANDLE_UNLOCK_CRITICAL();
  __FAILURE_HANDLE(mutex_unlock(queue->mutex));

  __FAILURE_HANDLE_CHECK_CRITICAL();

  *success = true;

  return LUMINARY_SUCCESS;
}

LuminaryResult queue_flush_blocking(Queue* queue) {
  __CHECK_NULL_ARGUMENT(queue);

  queue->shutdown = true;
  __FAILURE_HANDLE(condition_variable_broadcast(queue->cond_var));

  return LUMINARY_SUCCESS;
}

LuminaryResult _queue_destroy(Queue** queue, const char* buf_name, const char* func, uint32_t line) {
  __CHECK_NULL_ARGUMENT(queue);
  __CHECK_NULL_ARGUMENT(*queue);

  if (!(*queue)->buffer) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Queue buffer is NULL.");
  }

  if ((*queue)->write_ptr != (*queue)->read_ptr) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Queue is not empty.");
  }

  __FAILURE_HANDLE(queue_flush_blocking(*queue));

  __FAILURE_HANDLE(mutex_destroy(&((*queue)->mutex)));
  __FAILURE_HANDLE(condition_variable_destroy(&((*queue)->cond_var)));
  __FAILURE_HANDLE(_host_free((void**) &((*queue)->buffer), buf_name, func, line));
  __FAILURE_HANDLE(_host_free((void**) queue, buf_name, func, line));

  return LUMINARY_SUCCESS;
}
