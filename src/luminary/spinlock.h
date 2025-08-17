#ifndef LUMINARY_SPINLOCK_H
#define LUMINARY_SPINLOCK_H

#include <stdatomic.h>
#include <stdbool.h>

#define SpinLockObject _Atomic bool

// https://rigtorp.se/spinlock/

inline void spinlock_lock(SpinLockObject* lock) {
  for (;;) {
    if (!atomic_exchange_explicit(lock, true, memory_order_acquire)) {
      break;
    }

    while (atomic_load_explicit(lock, memory_order_relaxed)) {
      __asm("pause");
    }
  }
}

inline void spinlock_unlock(SpinLockObject* lock) {
  atomic_store_explicit(lock, false, memory_order_release);
}

inline bool spinlock_is_locked(SpinLockObject* lock) {
  return atomic_load_explicit(lock, memory_order_relaxed);
}

#define SpinLockCounter _Atomic uint32_t

inline void spinlock_counter_pop(SpinLockCounter* lock) {
  uint32_t expected;
  for (;;) {
    while ((expected = atomic_load_explicit(lock, memory_order_relaxed)) == 0) {
      __asm("pause");
    }

    if (atomic_compare_exchange_weak_explicit(lock, &expected, expected - 1, memory_order_acquire, memory_order_acquire)) {
      break;
    }
  }
}

inline void spinlock_counter_push(SpinLockCounter* lock) {
  atomic_fetch_add_explicit(lock, 1, memory_order_release);
}

inline void spinlock_count_wait_zero(SpinLockCounter* lock) {
  while (atomic_load_explicit(lock, memory_order_relaxed) != 0) {
    __asm("pause");
  }
}

#endif /* LUMINARY_SPINLOCK_H */
