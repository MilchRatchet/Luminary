#include "queue_worker.h"

#include "internal_error.h"
#include "string.h"

struct QueueWorkerMainArgs {
  char* name;
  Queue* queue;
  WallTime* wall_time;
  void* worker_context;
} typedef QueueWorkerMainArgs;

////////////////////////////////////////////////////////////////////
// Queue worker functions
////////////////////////////////////////////////////////////////////

static LuminaryResult _queue_worker_main(QueueWorkerMainArgs* args) {
  __CHECK_NULL_ARGUMENT(args);
  __CHECK_NULL_ARGUMENT(args->name);
  __CHECK_NULL_ARGUMENT(args->queue);
  __CHECK_NULL_ARGUMENT(args->wall_time);
  __CHECK_NULL_ARGUMENT(args->worker_context);

  bool success = true;

  __FAILURE_HANDLE(wall_time_set_worker_name(args->wall_time, args->name));

  while (success) {
    QueueEntry entry;
    __FAILURE_HANDLE(queue_pop_blocking(args->queue, &entry, &success));

    if (!success)
      break;

    if (entry.deferring_func) {
      bool defer_execution;
      __FAILURE_HANDLE(entry.deferring_func(args->worker_context, entry.args, &defer_execution));
      if (defer_execution) {
        __FAILURE_HANDLE(queue_push(args->queue, &entry));
        continue;
      }
    }

    __FAILURE_HANDLE(wall_time_set_string(args->wall_time, entry.name));
    __FAILURE_HANDLE(wall_time_start(args->wall_time));

    __FAILURE_HANDLE(entry.function(args->worker_context, entry.args));

    if (entry.clear_func) {
      __FAILURE_HANDLE(entry.clear_func(args->worker_context, entry.args));
    }

    __FAILURE_HANDLE(wall_time_stop(args->wall_time));
    __FAILURE_HANDLE(wall_time_set_string(args->wall_time, (const char*) 0));

#ifdef LUMINARY_WORK_QUEUE_STATS_PRINT
    double time;
    __FAILURE_HANDLE(wall_time_get_time(args->wall_time, &time));

    if (time > LUMINARY_WORK_QUEUE_STATS_PRINT_THRESHOLD) {
      warn_message("Queue %s: %s (%fs)", args->name, entry.name, time);
    }
#endif
  }

  __FAILURE_HANDLE(wall_time_set_worker_name(args->wall_time, (const char*) 0));

  return LUMINARY_SUCCESS;
}

static LuminaryResult _queue_worker_start_common(QueueWorker* worker, const char* name, Queue* queue, void* worker_context) {
  __CHECK_NULL_ARGUMENT(worker);
  __CHECK_NULL_ARGUMENT(name);
  __CHECK_NULL_ARGUMENT(queue);
  __CHECK_NULL_ARGUMENT(worker_context);

  if (worker->status != QUEUE_WORKER_STATUS_OFFLINE) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Queue worker must be offline to start it.");
  }

  const size_t name_length = strlen(name);

  QueueWorkerMainArgs* main_args = (QueueWorkerMainArgs*) worker->main_args;

  __FAILURE_HANDLE(host_malloc(&main_args->name, name_length + 1));
  memcpy(main_args->name, name, name_length);
  main_args->name[name_length] = '\0';

  main_args->queue          = queue;
  main_args->wall_time      = worker->wall_time;
  main_args->worker_context = worker_context;

  worker->status = QUEUE_WORKER_STATUS_ONLINE;

  return LUMINARY_SUCCESS;
}

////////////////////////////////////////////////////////////////////
// API
////////////////////////////////////////////////////////////////////

LuminaryResult queue_worker_create(QueueWorker** worker) {
  __CHECK_NULL_ARGUMENT(worker);

  __FAILURE_HANDLE(host_malloc(worker, sizeof(QueueWorker)));
  memset(*worker, 0, sizeof(QueueWorker));

  __FAILURE_HANDLE(thread_create(&(*worker)->thread));
  __FAILURE_HANDLE(wall_time_create(&(*worker)->wall_time));

  __FAILURE_HANDLE(host_malloc(&(*worker)->main_args, sizeof(QueueWorkerMainArgs)));
  memset((*worker)->main_args, 0, sizeof(QueueWorkerMainArgs));

  return LUMINARY_SUCCESS;
}

LuminaryResult queue_worker_start(QueueWorker* worker, const char* name, Queue* queue, void* worker_context) {
  __CHECK_NULL_ARGUMENT(worker);
  __CHECK_NULL_ARGUMENT(name);
  __CHECK_NULL_ARGUMENT(queue);
  __CHECK_NULL_ARGUMENT(worker_context);

  __FAILURE_HANDLE(_queue_worker_start_common(worker, name, queue, worker_context));

  __FAILURE_HANDLE(thread_start(worker->thread, (ThreadMainFunc) _queue_worker_main, (QueueWorkerMainArgs*) worker->main_args));

  return LUMINARY_SUCCESS;
}

LuminaryResult queue_worker_start_synchronous(QueueWorker* worker, const char* name, Queue* queue, void* worker_context) {
  __CHECK_NULL_ARGUMENT(worker);
  __CHECK_NULL_ARGUMENT(name);
  __CHECK_NULL_ARGUMENT(queue);
  __CHECK_NULL_ARGUMENT(worker_context);

  __FAILURE_HANDLE(_queue_worker_start_common(worker, name, queue, worker_context));

  __FAILURE_HANDLE(_queue_worker_main((QueueWorkerMainArgs*) worker->main_args));

  worker->status = QUEUE_WORKER_STATUS_OFFLINE;

  QueueWorkerMainArgs* main_args = (QueueWorkerMainArgs*) worker->main_args;
  __FAILURE_HANDLE(host_free(&main_args->name));

  return LUMINARY_SUCCESS;
}

LuminaryResult queue_worker_is_running(QueueWorker* worker, bool* is_running) {
  __CHECK_NULL_ARGUMENT(worker);
  __CHECK_NULL_ARGUMENT(is_running);

  *is_running = worker->status == QUEUE_WORKER_STATUS_ONLINE;

  return LUMINARY_SUCCESS;
}

LuminaryResult queue_worker_shutdown(QueueWorker* worker) {
  __CHECK_NULL_ARGUMENT(worker);

  if (worker->status == QUEUE_WORKER_STATUS_OFFLINE) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Cannot shutdown a worker thread that hasn't started.");
  }

  __FAILURE_HANDLE(thread_join(worker->thread));
  __FAILURE_HANDLE(thread_get_last_result(worker->thread));

  worker->status = QUEUE_WORKER_STATUS_OFFLINE;

  QueueWorkerMainArgs* main_args = (QueueWorkerMainArgs*) worker->main_args;
  __FAILURE_HANDLE(host_free(&main_args->name));

  return LUMINARY_SUCCESS;
}

LuminaryResult queue_worker_destroy(QueueWorker** worker) {
  __CHECK_NULL_ARGUMENT(worker);
  __CHECK_NULL_ARGUMENT(*worker);

  if ((*worker)->status == QUEUE_WORKER_STATUS_ONLINE) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Tried to destroy running queue worker.");
  }

  __FAILURE_HANDLE(thread_destroy(&(*worker)->thread));
  __FAILURE_HANDLE(wall_time_destroy(&(*worker)->wall_time));

  __FAILURE_HANDLE(host_free(&(*worker)->main_args));

  __FAILURE_HANDLE(host_free(worker));

  return LUMINARY_SUCCESS;
}
