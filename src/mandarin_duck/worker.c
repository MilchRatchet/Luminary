#include "worker.h"

#include <stdio.h>
#include <stdlib.h>

#include "utils.h"

void md_promise_create(MDPromise** promise) {
  MD_CHECK_NULL_ARGUMENT(promise);

  LUM_FAILURE_HANDLE(host_malloc(promise, sizeof(MDPromise)));

  (*promise)->status            = MD_PROMISE_STATUS_PENDING;
  (*promise)->result_data       = (void*) 0;
  (*promise)->mutex             = SDL_CreateMutex();
  (*promise)->continuation      = NULL;
  (*promise)->continuation_args = (void*) 0;

  if ((*promise)->mutex == (SDL_Mutex*) 0) {
    crash_message("Failed to create mutex.");
  }
}

void md_promise_destroy(MDPromise** promise) {
  MD_CHECK_NULL_ARGUMENT(promise);
  MD_CHECK_NULL_ARGUMENT(*promise);

  if ((*promise)->mutex) {
    SDL_DestroyMutex((*promise)->mutex);
  }

  LUM_FAILURE_HANDLE(host_free(promise));
}

static int _worker_loop(void* arg) {
  MDThreadPool* pool = (MDThreadPool*) arg;

  while (true) {
    SDL_LockMutex(pool->queue_mutex);

    while (pool->queue_head == (MDTaskNode*) 0 && !pool->shutdown) {
      SDL_WaitCondition(pool->condition, pool->queue_mutex);
    }

    if (pool->shutdown && pool->queue_head == (MDTaskNode*) 0) {
      SDL_UnlockMutex(pool->queue_mutex);
      break;
    }

    MDTask task;
    bool success = false;

    if (pool->queue_head != (MDTaskNode*) 0) {
      MDTaskNode* node = pool->queue_head;
      task             = node->task;
      success          = true;

      pool->queue_head = node->next;
      if (pool->queue_head == (MDTaskNode*) 0) {
        pool->queue_tail = (MDTaskNode*) 0;
      }
      LUM_FAILURE_HANDLE(host_free(&node));
    }

    SDL_UnlockMutex(pool->queue_mutex);

    if (success && task.exec) {
      task.exec(task.args, task.promise);

      SDL_LockMutex(task.promise->mutex);
      if (task.promise->status == MD_PROMISE_STATUS_PENDING) {
        task.promise->status = MD_PROMISE_STATUS_RESOLVED;
      }
      SDL_UnlockMutex(task.promise->mutex);
    }
  }

  return 0;
}

void md_thread_pool_create(MDThreadPool** pool) {
  MD_CHECK_NULL_ARGUMENT(pool);

  LUM_FAILURE_HANDLE(host_malloc(pool, sizeof(MDThreadPool)));

  (*pool)->num_threads = 4;
  (*pool)->shutdown    = false;
  (*pool)->queue_head  = (MDTaskNode*) 0;
  (*pool)->queue_tail  = (MDTaskNode*) 0;
  (*pool)->queue_mutex = SDL_CreateMutex();
  (*pool)->condition   = SDL_CreateCondition();

  if ((*pool)->queue_mutex == (SDL_Mutex*) 0 || (*pool)->condition == (SDL_Condition*) 0) {
    crash_message("Failed to create threadpool.");
  }

  LUM_FAILURE_HANDLE(host_malloc(&(*pool)->threads, sizeof(SDL_Thread*) * (*pool)->num_threads));

  for (int i = 0; i < (*pool)->num_threads; i++) {
    char name[32];
    snprintf(name, sizeof(name), "MDWorker-%d", i);
    (*pool)->threads[i] = SDL_CreateThread(_worker_loop, name, (void*) *pool);
  }
}

void md_thread_pool_destroy(MDThreadPool** pool) {
  MD_CHECK_NULL_ARGUMENT(pool);
  MD_CHECK_NULL_ARGUMENT(*pool);

  SDL_LockMutex((*pool)->queue_mutex);
  (*pool)->shutdown = true;
  SDL_BroadcastCondition((*pool)->condition);
  SDL_UnlockMutex((*pool)->queue_mutex);

  for (int i = 0; i < (*pool)->num_threads; i++) {
    if ((*pool)->threads[i]) {
      SDL_WaitThread((*pool)->threads[i], (int*) 0);
    }
  }

  while ((*pool)->queue_head != (MDTaskNode*) 0) {
    MDTaskNode* next = (*pool)->queue_head->next;
    LUM_FAILURE_HANDLE(host_free(&(*pool)->queue_head));
    (*pool)->queue_head = next;
  }

  LUM_FAILURE_HANDLE(host_free(&(*pool)->threads));
  SDL_DestroyMutex((*pool)->queue_mutex);
  SDL_DestroyCondition((*pool)->condition);

  LUM_FAILURE_HANDLE(host_free(pool));
}

MDPromise* md_thread_pool_enqueue(MDThreadPool* pool, void (*exec)(void*, MDPromise*), void* args) {
  MD_CHECK_NULL_ARGUMENT(pool);
  MD_CHECK_NULL_ARGUMENT(exec);

  MDPromise* promise;
  md_promise_create(&promise);

  MDTaskNode* node;
  LUM_FAILURE_HANDLE(host_malloc(&node, sizeof(MDTaskNode)));

  node->task.exec    = exec;
  node->task.args    = args;
  node->task.promise = promise;
  node->next         = (MDTaskNode*) 0;

  SDL_LockMutex(pool->queue_mutex);

  if (pool->queue_tail == (MDTaskNode*) 0) {
    pool->queue_head = node;
    pool->queue_tail = node;
  }
  else {
    pool->queue_tail->next = node;
    pool->queue_tail       = node;
  }

  SDL_SignalCondition(pool->condition);
  SDL_UnlockMutex(pool->queue_mutex);

  return promise;
}

void md_task_list_process(MDTaskNode** head) {
  MDTaskNode* prev = (MDTaskNode*) 0;
  MDTaskNode* curr = *head;
  while (curr) {
    bool resolved = false;
    SDL_LockMutex(curr->task.promise->mutex);
    if (curr->task.promise->status == MD_PROMISE_STATUS_RESOLVED) {
      resolved = true;
    }
    SDL_UnlockMutex(curr->task.promise->mutex);

    if (resolved) {
      if (curr->task.promise->continuation) {
        curr->task.promise->continuation(curr->task.promise->continuation_args);
      }

      md_promise_destroy(&curr->task.promise);
      LUM_FAILURE_HANDLE(host_free(&curr->task.args));

      MDTaskNode* node_to_free = curr;
      curr                     = curr->next;

      if (prev == (MDTaskNode*) 0) {
        *head = curr;
      }
      else {
        prev->next = curr;
      }

      LUM_FAILURE_HANDLE(host_free(&node_to_free));
    }
    else {
      prev = curr;
      curr = curr->next;
    }
  }
}
