#ifndef LUMINARY_VAULT_OBJECT_H
#define LUMINARY_VAULT_OBJECT_H

#include "utils.h"

struct VaultObject typedef VaultObject;
struct VaultHandle typedef VaultHandle;

LuminaryResult vault_object_create(VaultObject** object);
LuminaryResult vault_object_lock(VaultObject* object);
LuminaryResult vault_object_unlock(VaultObject* object);
LuminaryResult vault_object_get(VaultObject* object, uint32_t key, void** ptr);
LuminaryResult vault_object_set(VaultObject* object, uint32_t key, void* ptr);
LuminaryResult vault_object_reset(VaultObject* object);
LuminaryResult vault_object_destroy(VaultObject** object);

LuminaryResult vault_handle_create(VaultHandle** handle, VaultObject* object);
LuminaryResult vault_handle_lock(VaultHandle* handle);
LuminaryResult vault_handle_unlock(VaultHandle* handle);
LuminaryResult vault_handle_get(VaultHandle* handle, void** ptr);
LuminaryResult vault_handle_destroy(VaultHandle** handle);

#endif /* LUMINARY_VAULT_OBJECT_H */
