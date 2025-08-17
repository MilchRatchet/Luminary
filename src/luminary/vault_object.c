#include "vault_object.h"

#include "internal_error.h"
#include "spinlock.h"
#include "string.h"

struct VaultObject {
  void* ptr;
  uint32_t key;
  SpinLockObject spinlock;
  uint32_t num_active_handles;
};

struct VaultHandle {
  VaultObject* object;
  uint32_t key;
};

LuminaryResult vault_object_create(VaultObject** object) {
  __CHECK_NULL_ARGUMENT(object);

  __FAILURE_HANDLE(host_malloc(object, sizeof(VaultObject)));
  memset(*object, 0, sizeof(VaultObject));

  // An object is also a handle.
  (*object)->num_active_handles = 1;

  return LUMINARY_SUCCESS;
}

LuminaryResult vault_object_lock(VaultObject* object) {
  __CHECK_NULL_ARGUMENT(object);

  spinlock_lock(&object->spinlock);

  return LUMINARY_SUCCESS;
}

LuminaryResult vault_object_unlock(VaultObject* object) {
  __CHECK_NULL_ARGUMENT(object);

  spinlock_unlock(&object->spinlock);

  return LUMINARY_SUCCESS;
}

LuminaryResult vault_object_get(VaultObject* object, uint32_t key, void** ptr) {
  __CHECK_NULL_ARGUMENT(object);
  __CHECK_NULL_ARGUMENT(ptr);

  if (key != object->key) {
    *ptr = (void*) 0;
    return LUMINARY_SUCCESS;
  }

  *ptr = object->ptr;

  return LUMINARY_SUCCESS;
}

LuminaryResult vault_object_set(VaultObject* object, uint32_t key, void* ptr) {
  __CHECK_NULL_ARGUMENT(object);

  if (spinlock_is_locked(&object->spinlock) == false) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Tried to write to object without locking.");
  }

  object->key = key;
  object->ptr = ptr;

  return LUMINARY_SUCCESS;
}

LuminaryResult vault_object_reset(VaultObject* object) {
  __CHECK_NULL_ARGUMENT(object);

  __FAILURE_HANDLE_LOCK_CRITICAL();
  __FAILURE_HANDLE(vault_object_lock(object));

  __FAILURE_HANDLE_CRITICAL(vault_object_set(object, 0xFFFFFFFF, (void*) 0));

  __FAILURE_HANDLE_UNLOCK_CRITICAL();
  __FAILURE_HANDLE(vault_object_unlock(object));

  __FAILURE_HANDLE_CHECK_CRITICAL();

  return LUMINARY_SUCCESS;
}

LuminaryResult vault_object_destroy(VaultObject** object) {
  __CHECK_NULL_ARGUMENT(object);
  __CHECK_NULL_ARGUMENT(*object);

  __DEBUG_ASSERT((*object)->num_active_handles > 0);

  (*object)->num_active_handles--;

  // If there are active handles, mark the object for destruction and defer
  if ((*object)->num_active_handles > 0) {
    *object = (VaultObject*) 0;

    return LUMINARY_SUCCESS;
  }

  if (spinlock_is_locked(&(*object)->spinlock)) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Tried to create handle for a locked object.");
  }

  __FAILURE_HANDLE(host_free(object));

  return LUMINARY_SUCCESS;
}

LuminaryResult vault_handle_create(VaultHandle** handle, VaultObject* object) {
  __CHECK_NULL_ARGUMENT(handle);
  __CHECK_NULL_ARGUMENT(object);

  __FAILURE_HANDLE(host_malloc(handle, sizeof(VaultHandle)));
  memset(*handle, 0, sizeof(VaultHandle));

  __FAILURE_HANDLE(vault_object_lock(object));

  (*handle)->object = object;
  (*handle)->key    = object->key;

  object->num_active_handles++;

  __FAILURE_HANDLE(vault_object_unlock(object));

  return LUMINARY_SUCCESS;
}

LuminaryResult vault_handle_lock(VaultHandle* handle) {
  __CHECK_NULL_ARGUMENT(handle);

  __FAILURE_HANDLE(vault_object_lock(handle->object));

  return LUMINARY_SUCCESS;
}

LuminaryResult vault_handle_unlock(VaultHandle* handle) {
  __CHECK_NULL_ARGUMENT(handle);

  __FAILURE_HANDLE(vault_object_unlock(handle->object));

  return LUMINARY_SUCCESS;
}

LuminaryResult vault_handle_get(VaultHandle* handle, void** ptr) {
  __CHECK_NULL_ARGUMENT(handle);
  __CHECK_NULL_ARGUMENT(ptr);

  __FAILURE_HANDLE(vault_object_get(handle->object, handle->key, ptr));

  return LUMINARY_SUCCESS;
}

LuminaryResult vault_handle_destroy(VaultHandle** handle) {
  __CHECK_NULL_ARGUMENT(handle);
  __CHECK_NULL_ARGUMENT(*handle);

  __FAILURE_HANDLE(vault_object_destroy(&(*handle)->object));

  __FAILURE_HANDLE(host_free(handle));

  return LUMINARY_SUCCESS;
}
