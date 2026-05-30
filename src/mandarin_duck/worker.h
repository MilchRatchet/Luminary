#ifndef MANDARIN_DUCK_WORKER_H
#define MANDARIN_DUCK_WORKER_H

#include <SDL3/SDL.h>
#include <stdbool.h>

enum MDPromiseStatus {
  MD_PROMISE_STATUS_PENDING,
  MD_PROMISE_STATUS_RESOLVED,
  MD_PROMISE_STATUS_ERROR,
} typedef MDPromiseStatus;

struct MDPromise {
  MDPromiseStatus status;
  void* result_data;
  SDL_Mutex* mutex;
  void (*continuation)(void* args);
  void* continuation_args;
} typedef MDPromise;

struct MDTask {
  void (*exec)(void* args, MDPromise* promise);
  void* args;
  MDPromise* promise;
} typedef MDTask;

struct MDTaskNode {
  MDTask task;
  struct MDTaskNode* next;
} typedef MDTaskNode;

struct MDThreadPool {
  SDL_Thread** threads;
  int32_t num_threads;
  MDTaskNode* queue_head;
  MDTaskNode* queue_tail;
  SDL_Mutex* queue_mutex;
  SDL_Condition* condition;
  bool shutdown;
} typedef MDThreadPool;

void md_promise_create(MDPromise** promise);
void md_promise_destroy(MDPromise** promise);

void md_thread_pool_create(MDThreadPool** pool);
void md_thread_pool_destroy(MDThreadPool** pool);

MDPromise* md_thread_pool_enqueue(MDThreadPool* pool, void (*exec)(void*, MDPromise*), void* args);

void md_task_list_process(MDTaskNode** head);

#endif /* MANDARIN_DUCK_WORKER_H */
