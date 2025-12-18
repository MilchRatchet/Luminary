#include "device_material_manager.h"

#include "device.h"
#include "internal_error.h"

LuminaryResult material_manager_create(MaterialManager** manager) {
  __CHECK_NULL_ARGUMENT(manager);

  __FAILURE_HANDLE(host_malloc(manager, sizeof(MaterialManager)));
  memset(*manager, 0, sizeof(MaterialManager));

  __FAILURE_HANDLE(array_create(&(*manager)->cached_updates, sizeof(MaterialProcessedUpdate), 16));

  return LUMINARY_SUCCESS;
}

LuminaryResult material_manager_add_updates(MaterialManager* manager, const ARRAY MaterialUpdate* updates) {
  __CHECK_NULL_ARGUMENT(manager);

  uint32_t num_updates;
  __FAILURE_HANDLE(array_get_num_elements(updates, &num_updates));

  bool new_materials_are_added = false;

  for (uint32_t update_id = 0; update_id < num_updates; update_id++) {
    const uint32_t material_id = updates[update_id].material_id;

    if (material_id >= manager->num_materials) {
      manager->num_materials  = material_id + 1;
      new_materials_are_added = true;
    }
  }

  if (new_materials_are_added) {
    if (manager->materials) {
      __FAILURE_HANDLE(host_realloc(&manager->materials, sizeof(DeviceMaterialCompressed) * manager->num_materials));
    }
    else {
      __FAILURE_HANDLE(host_malloc(&manager->materials, sizeof(DeviceMaterialCompressed) * manager->num_materials));
    }

    // We force reallocation so we don't need to track the updates.
    __FAILURE_HANDLE(material_manager_clear_updates(manager));
  }

  for (uint32_t update_id = 0; update_id < num_updates; update_id++) {
    MaterialProcessedUpdate processed_update;
    processed_update.material_id = updates[update_id].material_id;

    __FAILURE_HANDLE(device_struct_material_convert(&updates[update_id].material, &processed_update.material));

    if (new_materials_are_added == false)
      __FAILURE_HANDLE(array_push(&manager->cached_updates, &processed_update));

    memcpy(manager->materials + processed_update.material_id, &processed_update.material, sizeof(DeviceMaterialCompressed));
  }

  return LUMINARY_SUCCESS;
}

LuminaryResult material_manager_clear_updates(MaterialManager* manager) {
  __CHECK_NULL_ARGUMENT(manager);

  __FAILURE_HANDLE(array_clear(manager->cached_updates));

  return LUMINARY_SUCCESS;
}

LuminaryResult material_manager_destroy(MaterialManager** manager) {
  __CHECK_NULL_ARGUMENT(manager);
  __CHECK_NULL_ARGUMENT(*manager);

  __FAILURE_HANDLE(host_free(&(*manager)->materials));
  __FAILURE_HANDLE(array_destroy(&(*manager)->cached_updates));

  __FAILURE_HANDLE(host_free(manager));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_material_manager_create(DeviceMaterialManager** manager) {
  __CHECK_NULL_ARGUMENT(manager);

  __FAILURE_HANDLE(host_malloc(manager, sizeof(DeviceMaterialManager)));
  memset(*manager, 0, sizeof(DeviceMaterialManager));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_material_manager_update(
  DeviceMaterialManager* manager, Device* device, const MaterialManager* shared_manager, bool* buffers_have_changed) {
  __CHECK_NULL_ARGUMENT(manager);
  __CHECK_NULL_ARGUMENT(device);
  __CHECK_NULL_ARGUMENT(shared_manager);
  __CHECK_NULL_ARGUMENT(buffers_have_changed);

  *buffers_have_changed = false;

  if (manager->allocated_num_materials != shared_manager->num_materials) {
    if (manager->materials)
      __FAILURE_HANDLE(device_free(&manager->materials));

    __FAILURE_HANDLE(device_malloc(&manager->materials, shared_manager->num_materials * sizeof(DeviceMaterialCompressed)));

    __FAILURE_HANDLE(device_staging_manager_register(
      device->staging_manager, shared_manager->materials, (DEVICE void*) manager->materials, 0,
      shared_manager->num_materials * sizeof(DeviceMaterialCompressed)));

    manager->allocated_num_materials = shared_manager->num_materials;
    *buffers_have_changed            = true;

    return LUMINARY_SUCCESS;
  }

  uint32_t num_updates;
  __FAILURE_HANDLE(array_get_num_elements(shared_manager->cached_updates, &num_updates));

  for (uint32_t update_id = 0; update_id < num_updates; update_id++) {
    __FAILURE_HANDLE(device_staging_manager_register(
      device->staging_manager, &shared_manager->cached_updates[update_id].material, (DEVICE void*) manager->materials,
      sizeof(DeviceMaterialCompressed) * shared_manager->cached_updates[update_id].material_id, sizeof(DeviceMaterialCompressed)));
  }

  return LUMINARY_SUCCESS;
}

LuminaryResult device_material_manager_get_ptrs(DeviceMaterialManager* manager, DeviceMaterialManagerPtrs* ptrs) {
  __CHECK_NULL_ARGUMENT(manager);
  __CHECK_NULL_ARGUMENT(ptrs);

  ptrs->materials = DEVICE_CUPTR(manager->materials);

  return LUMINARY_SUCCESS;
}

LuminaryResult device_material_manager_destroy(DeviceMaterialManager** manager) {
  __CHECK_NULL_ARGUMENT(manager);

  if ((*manager)->materials)
    __FAILURE_HANDLE(device_free(&(*manager)->materials));

  __FAILURE_HANDLE(host_free(manager));

  return LUMINARY_SUCCESS;
}
