#include "thread.h"

#include <threads.h>

#include "internal_error.h"

struct ThreadMainArgs;

struct Thread {
  thrd_t thread;
  struct ThreadMainArgs* main_args;
  LuminaryResult result;
  bool running;
} typedef Thread;

struct ThreadMainArgs {
  LuminaryResult (*func)(void* args);
  void* args;
  Thread* self;
};

LuminaryResult thread_create(Thread** _thread) {
  if (!_thread) {
    __RETURN_ERROR(LUMINARY_ERROR_ARGUMENT_NULL, "Thread is NULL.");
  }

  Thread* thread;

  __FAILURE_HANDLE(host_malloc(&thread, sizeof(Thread)));

  __FAILURE_HANDLE(host_malloc(&thread->main_args, sizeof(struct ThreadMainArgs)));

  thread->running = false;
  thread->result  = LUMINARY_SUCCESS;

  return LUMINARY_SUCCESS;
}

int _thread_main(struct ThreadMainArgs* args) {
  LuminaryResult result = args->func(args->args);

  args->self->result = result;

  return 0;
}

LuminaryResult thread_start(Thread* thread, LuminaryResult (*func)(void* args), void* args) {
  if (!thread) {
    __RETURN_ERROR(LUMINARY_ERROR_ARGUMENT_NULL, "Thread is NULL.");
  }

  if (thread->running) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Thread is already running.");
  }

  const thrd_start_t start_func = (thrd_start_t) _thread_main;

  thread->main_args->func = func;
  thread->main_args->args = args;
  thread->main_args->self = thread;

  const int retval = thrd_create(&thread->thread, start_func, thread->main_args);

  if (retval != thrd_success) {
    __RETURN_ERROR(LUMINARY_ERROR_C_STD, "thrd_create returned an error.");
  }

  thread->running = true;

  return LUMINARY_SUCCESS;
}

LuminaryResult thread_join(Thread* thread) {
  if (!thread) {
    __RETURN_ERROR(LUMINARY_ERROR_ARGUMENT_NULL, "Thread is NULL.");
  }

  if (!thread->running) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Thread is not running.");
  }

  const int retval = thrd_join(thread->thread, (int*) 0);

  if (retval != thrd_success) {
    __RETURN_ERROR(LUMINARY_ERROR_C_STD, "thrd_join returned an error.");
  }

  thread->running = false;

  return LUMINARY_SUCCESS;
}

LuminaryResult thread_get_last_result(const Thread* thread) {
  if (!thread) {
    __RETURN_ERROR(LUMINARY_ERROR_ARGUMENT_NULL, "Thread is NULL.");
  }

  return thread->result;
}

LuminaryResult thread_destroy(Thread** thread) {
  if (!thread) {
    __RETURN_ERROR(LUMINARY_ERROR_ARGUMENT_NULL, "Thread ptr is NULL.");
  }

  if (!(*thread)) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Thread is NULL.");
  }

  if ((*thread)->running) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Thread is still running.");
  }

  host_free((*thread)->main_args);
  host_free(thread);

  return LUMINARY_SUCCESS;
}
