#ifndef LUMINARY_QUEUE_WORKER_H
#define LUMINARY_QUEUE_WORKER_H

#include "thread.h"
#include "utils.h"

enum QueueWorkerStatus {
  QUEUE_WORKER_STATUS_OFFLINE,
  QUEUE_WORKER_STATUS_ONLINE,
  QUEUE_WORKER_STATUS_FATAL_ERROR
} typedef QueueWorkerStatus;

struct QueueWorker {
  QueueWorkerStatus status;
  Thread* thread;
  WallTime* wall_time;
  void* main_args;
} typedef QueueWorker;

typedef LuminaryResult (*QueueEntryFunction)(void* worker, void* args);
typedef LuminaryResult (*QueueEntryDeferringFunction)(void* worker, void* args, bool* defer_execution);

struct QueueEntry {
  const char* name;
  QueueEntryFunction function;
  QueueEntryFunction clear_func;
  QueueEntryDeferringFunction deferring_func;
  void* args;
  bool remove_duplicates;
  bool queuer_cannot_execute;  // Used to avoid self execution on CUDA callback threads.
  bool skip_execution;         // Used to skip the actual work function but still execute the clear func.
} typedef QueueEntry;

LuminaryResult queue_worker_create(QueueWorker** worker);
LuminaryResult queue_worker_start(QueueWorker* worker, const char* name, Queue* queue, void* worker_context);
LuminaryResult queue_worker_start_synchronous(QueueWorker* worker, const char* name, Queue* queue, void* worker_context);
LuminaryResult queue_worker_is_running(QueueWorker* worker, bool* is_running);
LuminaryResult queue_worker_shutdown(QueueWorker* worker);
LuminaryResult queue_worker_destroy(QueueWorker** worker);

#endif /* LUMINARY_QUEUE_WORKER_H */
