#include "device_mesh_instance_manager.h"

#include "device.h"
#include "device_mesh.h"
#include "internal_error.h"
#include "optix_bvh.h"

LuminaryResult mesh_instance_manager_create(MeshInstanceManager** manager) {
  __CHECK_NULL_ARGUMENT(manager);

  __FAILURE_HANDLE(host_malloc(manager, sizeof(MeshInstanceManager)));
  memset(*manager, 0, sizeof(MeshInstanceManager));

  __FAILURE_HANDLE(array_create(&(*manager)->cached_updates, sizeof(MeshInstanceProcessedUpdate), 16));

  return LUMINARY_SUCCESS;
}

LuminaryResult mesh_instance_manager_add_updates(MeshInstanceManager* manager, const ARRAY MeshInstanceUpdate* updates) {
  __CHECK_NULL_ARGUMENT(manager);
  __CHECK_NULL_ARGUMENT(updates);

  uint32_t num_updates;
  __FAILURE_HANDLE(array_get_num_elements(updates, &num_updates));

  bool new_instances_are_added = false;

  for (uint32_t update_id = 0; update_id < num_updates; update_id++) {
    const uint32_t instance_id = updates[update_id].instance_id;

    if (instance_id >= manager->num_instances) {
      manager->num_instances  = instance_id + 1;
      new_instances_are_added = true;
    }
  }

  if (new_instances_are_added) {
    if (manager->instance_transforms) {
      __FAILURE_HANDLE(host_realloc(&manager->instance_transforms, sizeof(DeviceTransform) * manager->num_instances));
    }
    else {
      __FAILURE_HANDLE(host_malloc(&manager->instance_transforms, sizeof(DeviceTransform) * manager->num_instances));
    }

    if (manager->instance_mesh_ids) {
      __FAILURE_HANDLE(host_realloc(&manager->instance_mesh_ids, sizeof(uint32_t) * manager->num_instances));
    }
    else {
      __FAILURE_HANDLE(host_malloc(&manager->instance_mesh_ids, sizeof(uint32_t) * manager->num_instances));
    }
  }

  for (uint32_t update_id = 0; update_id < num_updates; update_id++) {
    MeshInstanceProcessedUpdate processed_update;
    processed_update.instance_id = updates[update_id].instance_id;
    processed_update.instance    = updates[update_id].instance;

    __FAILURE_HANDLE(device_struct_instance_transform_convert(&updates[update_id].instance, &processed_update.transform));
    processed_update.mesh_id = updates[update_id].instance.mesh_id;

    __FAILURE_HANDLE(array_push(&manager->cached_updates, &processed_update));

    memcpy(manager->instance_transforms + processed_update.instance_id, &processed_update.transform, sizeof(DeviceTransform));
    memcpy(manager->instance_mesh_ids + processed_update.instance_id, &processed_update.mesh_id, sizeof(uint32_t));
  }

  return LUMINARY_SUCCESS;
}

LuminaryResult mesh_instance_manager_clear_updates(MeshInstanceManager* manager) {
  __CHECK_NULL_ARGUMENT(manager);

  __FAILURE_HANDLE(array_clear(manager->cached_updates));

  return LUMINARY_SUCCESS;
}

LuminaryResult mesh_instance_manager_destroy(MeshInstanceManager** manager) {
  __CHECK_NULL_ARGUMENT(manager);

  if ((*manager)->instance_transforms)
    __FAILURE_HANDLE(host_free(&(*manager)->instance_transforms));

  if ((*manager)->instance_mesh_ids)
    __FAILURE_HANDLE(host_free(&(*manager)->instance_mesh_ids));

  __FAILURE_HANDLE(array_destroy(&(*manager)->cached_updates));

  __FAILURE_HANDLE(host_free(manager));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_mesh_instance_manager_create(DeviceMeshInstanceManager** manager) {
  __CHECK_NULL_ARGUMENT(manager);

  __FAILURE_HANDLE(host_malloc(manager, sizeof(DeviceMeshInstanceManager)));
  memset(*manager, 0, sizeof(DeviceMeshInstanceManager));

  __FAILURE_HANDLE(array_create(&(*manager)->meshes, sizeof(DeviceMesh*), 16));

  __FAILURE_HANDLE(optix_bvh_instance_cache_create(&(*manager)->optix_instance_cache));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_mesh_instance_manager_add_mesh(
  DeviceMeshInstanceManager* manager, Device* device, const Mesh* mesh, bool* buffers_have_changed) {
  __CHECK_NULL_ARGUMENT(manager);
  __CHECK_NULL_ARGUMENT(device);
  __CHECK_NULL_ARGUMENT(mesh);
  __CHECK_NULL_ARGUMENT(buffers_have_changed);

  *buffers_have_changed = false;

  DeviceMesh* device_mesh;
  __FAILURE_HANDLE(device_mesh_create(&device_mesh));

  __FAILURE_HANDLE(device_mesh_set(device_mesh, device, mesh));

  __FAILURE_HANDLE(array_push(&manager->meshes, &device_mesh));

  uint32_t num_meshes;
  __FAILURE_HANDLE(array_get_num_elements(manager->meshes, &num_meshes));

  if (manager->allocated_num_meshes != num_meshes) {
    if (manager->vertices)
      __FAILURE_HANDLE(device_free(&manager->vertices));

    if (manager->texture_triangles)
      __FAILURE_HANDLE(device_free(&manager->texture_triangles));

    __FAILURE_HANDLE(device_malloc(&manager->vertices, sizeof(DeviceTriangleVertex*) * num_meshes));
    __FAILURE_HANDLE(device_malloc(&manager->texture_triangles, sizeof(DeviceTriangleTexture*) * num_meshes));

    DeviceTriangleVertex** vertices_direct_access;
    __FAILURE_HANDLE(device_staging_manager_register_direct_access(
      device->staging_manager, (DEVICE void*) manager->vertices, 0, sizeof(DeviceTriangleVertex*) * num_meshes,
      (void**) &vertices_direct_access));

    for (uint32_t mesh_id = 0; mesh_id < num_meshes; mesh_id++) {
      vertices_direct_access[mesh_id] = DEVICE_PTR(manager->meshes[mesh_id]->vertices);
    }

    DeviceTriangleVertex** texture_direct_access;
    __FAILURE_HANDLE(device_staging_manager_register_direct_access(
      device->staging_manager, (DEVICE void*) manager->texture_triangles, 0, sizeof(DeviceTriangleTexture*) * num_meshes,
      (void**) &texture_direct_access));

    for (uint32_t mesh_id = 0; mesh_id < num_meshes; mesh_id++) {
      texture_direct_access[mesh_id] = DEVICE_PTR(manager->meshes[mesh_id]->texture_triangles);
    }

    manager->allocated_num_meshes = num_meshes;
    *buffers_have_changed         = true;
  }

  return LUMINARY_SUCCESS;
}

LuminaryResult device_mesh_instance_manager_update(
  DeviceMeshInstanceManager* manager, Device* device, const MeshInstanceManager* shared_manager, bool* buffers_have_changed) {
  __CHECK_NULL_ARGUMENT(manager);
  __CHECK_NULL_ARGUMENT(device);
  __CHECK_NULL_ARGUMENT(shared_manager);
  __CHECK_NULL_ARGUMENT(buffers_have_changed);

  *buffers_have_changed = false;

  uint32_t num_updates;
  __FAILURE_HANDLE(array_get_num_elements(shared_manager->cached_updates, &num_updates));

  if (num_updates == 0)
    return LUMINARY_SUCCESS;

  *buffers_have_changed = true;

  if (manager->allocated_num_instances != shared_manager->num_instances) {
    if (manager->instance_transforms)
      __FAILURE_HANDLE(device_free(&manager->instance_transforms));

    if (manager->instance_mesh_ids)
      __FAILURE_HANDLE(device_free(&manager->instance_mesh_ids));

    __FAILURE_HANDLE(device_malloc(&manager->instance_transforms, sizeof(DeviceTransform) * shared_manager->num_instances));
    __FAILURE_HANDLE(device_malloc(&manager->instance_mesh_ids, sizeof(uint32_t) * shared_manager->num_instances));

    __FAILURE_HANDLE(device_staging_manager_register(
      device->staging_manager, shared_manager->instance_transforms, (DEVICE void*) manager->instance_transforms, 0,
      sizeof(DeviceTransform) * shared_manager->num_instances));

    __FAILURE_HANDLE(device_staging_manager_register(
      device->staging_manager, shared_manager->instance_mesh_ids, (DEVICE void*) manager->instance_mesh_ids, 0,
      sizeof(uint32_t) * shared_manager->num_instances));

    manager->allocated_num_instances = shared_manager->num_instances;
  }
  else {
    for (uint32_t update_id = 0; update_id < num_updates; update_id++) {
      __FAILURE_HANDLE(device_staging_manager_register(
        device->staging_manager, &shared_manager->cached_updates[update_id].transform, (DEVICE void*) manager->instance_transforms,
        sizeof(DeviceTransform) * shared_manager->cached_updates[update_id].instance_id, sizeof(DeviceTransform)));

      __FAILURE_HANDLE(device_staging_manager_register(
        device->staging_manager, &shared_manager->cached_updates[update_id].mesh_id, (DEVICE void*) manager->instance_mesh_ids,
        sizeof(uint32_t) * shared_manager->cached_updates[update_id].instance_id, sizeof(uint32_t)));
    }
  }

  for (uint32_t update_id = 0; update_id < num_updates; update_id++) {
    const uint32_t mesh_id = shared_manager->cached_updates[update_id].mesh_id;

    bool mesh_data_has_changed;
    __FAILURE_HANDLE(device_mesh_process(manager->meshes[mesh_id], device, &mesh_data_has_changed));
  }

  __FAILURE_HANDLE(optix_bvh_instance_cache_update(
    manager->optix_instance_cache, device, shared_manager->cached_updates, (const DeviceMesh**) manager->meshes))

  if (manager->bvh == (OptixBVH*) 0)
    __FAILURE_HANDLE(optix_bvh_create(&manager->bvh));

  __FAILURE_HANDLE(optix_bvh_ias_build(manager->bvh, device, manager->optix_instance_cache));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_mesh_instance_manager_get_ptrs(DeviceMeshInstanceManager* manager, DeviceMeshInstanceManagerPtrs* ptrs) {
  __CHECK_NULL_ARGUMENT(manager);
  __CHECK_NULL_ARGUMENT(ptrs);

  ptrs->vertices            = DEVICE_CUPTR(manager->vertices);
  ptrs->texture_triangles   = DEVICE_CUPTR(manager->texture_triangles);
  ptrs->instance_transforms = DEVICE_CUPTR(manager->instance_transforms);
  ptrs->instance_mesh_ids   = DEVICE_CUPTR(manager->instance_mesh_ids);
  ptrs->bvh                 = (manager->bvh) ? manager->bvh->traversable[OPTIX_BVH_TYPE_DEFAULT] : (OptixTraversableHandle) 0;
  ptrs->bvh_shadow          = (manager->bvh) ? manager->bvh->traversable[OPTIX_BVH_TYPE_SHADOW] : (OptixTraversableHandle) 0;

  return LUMINARY_SUCCESS;
}

LuminaryResult device_mesh_instance_manager_destroy(DeviceMeshInstanceManager** manager) {
  __CHECK_NULL_ARGUMENT(manager);
  __CHECK_NULL_ARGUMENT(*manager);

  if ((*manager)->vertices)
    __FAILURE_HANDLE(device_free(&(*manager)->vertices));

  if ((*manager)->texture_triangles)
    __FAILURE_HANDLE(device_free(&(*manager)->texture_triangles));

  if ((*manager)->instance_transforms)
    __FAILURE_HANDLE(device_free(&(*manager)->instance_transforms));

  if ((*manager)->instance_mesh_ids)
    __FAILURE_HANDLE(device_free(&(*manager)->instance_mesh_ids));

  if ((*manager)->bvh)
    __FAILURE_HANDLE(optix_bvh_destroy(&(*manager)->bvh));

  uint32_t num_meshes;
  __FAILURE_HANDLE(array_get_num_elements((*manager)->meshes, &num_meshes));

  for (uint32_t mesh_id = 0; mesh_id < num_meshes; mesh_id++) {
    __FAILURE_HANDLE(device_mesh_destroy(&(*manager)->meshes[mesh_id]));
  }

  __FAILURE_HANDLE(array_destroy(&(*manager)->meshes));

  __FAILURE_HANDLE(optix_bvh_instance_cache_destroy(&(*manager)->optix_instance_cache));

  __FAILURE_HANDLE(host_free(manager));

  return LUMINARY_SUCCESS;
}
