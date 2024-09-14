#ifndef LUMINARY_COND_VAR_H
#define LUMINARY_COND_VAR_H

#include "mutex.h"
#include "utils.h"

struct ConditionVariable;
typedef struct ConditionVariable ConditionVariable;

LuminaryResult condition_variable_create(ConditionVariable** cond_var);
LuminaryResult condition_variable_signal(ConditionVariable* cond_var);
LuminaryResult condition_variable_broadcast(ConditionVariable* cond_var);
LuminaryResult condition_variable_wait(ConditionVariable* cond_var, Mutex* mutex);
LuminaryResult condition_variable_destroy(ConditionVariable** cond_var);

#endif /* LUMINARY_COND_VAR_H */
