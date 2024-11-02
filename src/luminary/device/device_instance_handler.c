
#include "device_instance_handler.h"

#include "device_mesh.h"
#include "internal_error.h"

LuminaryResult device_instance_handler_create(DeviceInstanceHandler** device_instance_handler) {
  __CHECK_NULL_ARGUMENT(device_instance_handler);

  __FAILURE_HANDLE(host_malloc(device_instance_handler, sizeof(DeviceInstanceHandler)));

  memset(*device_instance_handler, 0, sizeof(DeviceInstanceHandler));

  __FAILURE_HANDLE(array_create(&(*device_instance_handler)->instance_maps, sizeof(uint32_t*), 16));
  __FAILURE_HANDLE(array_create(&(*device_instance_handler)->instancelet_active, sizeof(bool), 16));
  __FAILURE_HANDLE(array_create(&(*device_instance_handler)->instancelets, sizeof(DeviceInstancelet), 16));
  __FAILURE_HANDLE(array_create(&(*device_instance_handler)->transforms, sizeof(DeviceTransform), 16));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_instance_handler_update(
  DeviceInstanceHandler* device_instance_handler, MeshInstanceUpdate* update, ARRAYPTR uint32_t** instancelets_dirty,
  const ARRAY DeviceMesh** device_meshes) {
  __CHECK_NULL_ARGUMENT(device_instance_handler);
  __CHECK_NULL_ARGUMENT(update);
  __CHECK_NULL_ARGUMENT(instancelets_dirty);
  __CHECK_NULL_ARGUMENT(device_meshes);

  uint32_t instances_allocated;
  __FAILURE_HANDLE(array_get_num_elements(device_instance_handler->instance_maps, &instances_allocated));

  ARRAY uint32_t* instancelet_map;
  if (update->instance_id < instances_allocated) {
    instancelet_map = device_instance_handler->instance_maps[update->instance_id];
  }
  else {
    // In theory, new instances should be added chronologically so we should never run into this.
    if (update->instance_id > instances_allocated) {
      __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Instance ID is far beyond the allocated instances on device side.");
    }

    __FAILURE_HANDLE(array_create(&instancelet_map, sizeof(uint32_t), 4));
    __FAILURE_HANDLE(array_push(&device_instance_handler->instance_maps, &instancelet_map));
  }

  uint32_t allocated_instancelets;
  __FAILURE_HANDLE(array_get_num_elements(instancelet_map, &allocated_instancelets));

  uint32_t device_meshes_count;
  __FAILURE_HANDLE(array_get_num_elements(device_meshes, &device_meshes_count));

  if (update->instance.mesh_id >= device_meshes_count) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Instance ID refers to non existent mesh.");
  }

  const DeviceMesh* device_mesh = device_meshes[update->instance.mesh_id];

  uint32_t meshlet_count;
  __FAILURE_HANDLE(array_get_num_elements(device_mesh->meshlet_triangle_offsets, &meshlet_count));

  const bool instance_active = update->instance.active;

  for (uint32_t meshlet_id = 0; meshlet_id < meshlet_count; meshlet_id++) {
    uint32_t instancelet_id;
    if (meshlet_id < allocated_instancelets) {
      instancelet_id = instancelet_map[meshlet_id];
    }
    else {
      // TODO: Search through the instancelet for inactive instancelets and replace those
      instancelet_id = device_instance_handler->num_instancelets;

      // Reload the map because the pointer could have changed.
      __FAILURE_HANDLE(array_push(device_instance_handler->instance_maps + update->instance_id, &instancelet_id));
      instancelet_map = device_instance_handler->instance_maps[update->instance_id];
    }

    DeviceInstancelet device_instance;
    DeviceTransform device_transform;
    __FAILURE_HANDLE(device_struct_instance_convert(&update->instance, &device_instance, &device_transform, device_mesh, meshlet_id));

    // If we had allocated a new instancelet, we must push it so the array grows.
    if (instancelet_id == device_instance_handler->num_instancelets) {
      __FAILURE_HANDLE(array_push(&device_instance_handler->instancelets, &device_instance));
      __FAILURE_HANDLE(array_push(&device_instance_handler->transforms, &device_transform));
      __FAILURE_HANDLE(array_push(&device_instance_handler->instancelet_active, &instance_active));

      device_instance_handler->num_instancelets++;
    }
    else {
      device_instance_handler->instancelets[instancelet_id]       = device_instance;
      device_instance_handler->transforms[instancelet_id]         = device_transform;
      device_instance_handler->instancelet_active[instancelet_id] = instance_active;
    }

    __FAILURE_HANDLE(array_push(instancelets_dirty, &instancelet_id));
  }

  return LUMINARY_SUCCESS;
}

LuminaryResult device_instance_handler_destroy(DeviceInstanceHandler** device_instance_handler) {
  __CHECK_NULL_ARGUMENT(device_instance_handler);

  uint32_t instance_map_count;
  __FAILURE_HANDLE(array_get_num_elements((*device_instance_handler)->instance_maps, &instance_map_count));

  for (uint32_t instance_id = 0; instance_id < instance_map_count; instance_id++) {
    __FAILURE_HANDLE(array_destroy(&(*device_instance_handler)->instance_maps[instance_id]));
  }

  __FAILURE_HANDLE(array_destroy(&(*device_instance_handler)->instance_maps));
  __FAILURE_HANDLE(array_destroy(&(*device_instance_handler)->instancelet_active));
  __FAILURE_HANDLE(array_destroy(&(*device_instance_handler)->instancelets));
  __FAILURE_HANDLE(array_destroy(&(*device_instance_handler)->transforms));

  __FAILURE_HANDLE(host_free(device_instance_handler));

  return LUMINARY_SUCCESS;
}
