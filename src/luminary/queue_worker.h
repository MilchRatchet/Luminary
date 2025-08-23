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

LuminaryResult queue_worker_create(QueueWorker** worker);
LuminaryResult queue_worker_start(QueueWorker* worker, const char* name, Queue* queue, void* worker_context);
LuminaryResult queue_worker_shutdown(QueueWorker* worker);
LuminaryResult queue_worker_destroy(QueueWorker** worker);

#endif /* LUMINARY_QUEUE_WORKER_H */
