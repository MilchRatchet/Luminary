#ifndef LUMINARY_THREAD_H
#define LUMINARY_THREAD_H

#include "utils.h"

struct Thread;
typedef struct Thread Thread;

LuminaryResult thread_create(Thread** thread);
LuminaryResult thread_start(Thread* thread, LuminaryResult (*func)(void* args), void* args);
LuminaryResult thread_join(Thread* thread);
LuminaryResult thread_get_last_result(const Thread* thread);
LuminaryResult thread_destroy(Thread** thread);

#endif /* LUMINARY_THREAD_H */
