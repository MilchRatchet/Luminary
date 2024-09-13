#ifndef LUMINARY_MUTEX_H
#define LUMINARY_MUTEX_H

#include "utils.h"

struct Mutex;
typedef struct Mutex Mutex;

LuminaryResult mutex_create(Mutex** mutex);
LuminaryResult mutex_lock(Mutex* mutex);
LuminaryResult mutex_timed_lock(Mutex* mutex, const double timeout_time, bool* success);
LuminaryResult mutex_try_lock(Mutex* mutex, bool* success);
LuminaryResult mutex_unlock(Mutex* mutex);
LuminaryResult mutex_destroy(Mutex** mutex);

#endif /* LUMINARY_MUTEX_H */
