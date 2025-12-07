#include "device_mesh.h"

#include "device.h"
#include "device_staging_manager.h"
#include "internal_error.h"
#include "struct_interleaving.h"

LuminaryResult device_mesh_create(DeviceMesh** device_mesh) {
  __CHECK_NULL_ARGUMENT(device_mesh);

  __FAILURE_HANDLE(host_malloc(device_mesh, sizeof(DeviceMesh)));
  memset(*device_mesh, 0, sizeof(DeviceMesh));

  (*device_mesh)->id = 0xFFFFFFFF;

  return LUMINARY_SUCCESS;
}

LuminaryResult device_mesh_set(DeviceMesh* device_mesh, Device* device, const Mesh* mesh) {
  __CHECK_NULL_ARGUMENT(device_mesh);
  __CHECK_NULL_ARGUMENT(device);
  __CHECK_NULL_ARGUMENT(mesh);

  device_mesh->triangle_count = mesh->data.triangle_count;

  __FAILURE_HANDLE(device_malloc(&device_mesh->vertices, sizeof(DeviceTriangleVertex) * device_mesh->triangle_count * 3));

  // TODO: This will fail for very large meshes.
  DeviceTriangleVertex* vertex_buffer_access;
  __FAILURE_HANDLE(device_staging_manager_register_direct_access(
    device->staging_manager, device_mesh->vertices, 0, sizeof(DeviceTriangleVertex) * device_mesh->triangle_count * 3,
    (void**) &vertex_buffer_access));

  for (uint32_t vertex_id = 0; vertex_id < device_mesh->triangle_count * 3; vertex_id++) {
    __FAILURE_HANDLE(device_struct_vertex_convert(&mesh->data, vertex_id, vertex_buffer_access + vertex_id));
  }

  __FAILURE_HANDLE(device_malloc(&device_mesh->texture_triangles, sizeof(DeviceTriangleTexture) * device_mesh->triangle_count));

  DeviceTriangleTexture* texture_buffer_access;
  __FAILURE_HANDLE(device_staging_manager_register_direct_access(
    device->staging_manager, device_mesh->texture_triangles, 0, sizeof(DeviceTriangleTexture) * device_mesh->triangle_count,
    (void**) &texture_buffer_access));

  for (uint32_t triangle_id = 0; triangle_id < device_mesh->triangle_count; triangle_id++) {
    __FAILURE_HANDLE(device_struct_triangle_texture_convert(&mesh->data, triangle_id, texture_buffer_access + triangle_id));
  }

  return LUMINARY_SUCCESS;
}

DEVICE_CTX_FUNC LuminaryResult device_mesh_process(DeviceMesh* device_mesh, Device* device, bool* data_has_changed) {
  __CHECK_NULL_ARGUMENT(device_mesh);
  __CHECK_NULL_ARGUMENT(device);
  __CHECK_NULL_ARGUMENT(data_has_changed);

  *data_has_changed = false;

  if (device_mesh->omm == (OpacityMicromap*) 0) {
    __FAILURE_HANDLE(omm_create(&device_mesh->omm));
    __FAILURE_HANDLE(omm_build(device_mesh->omm, device, device_mesh));

    *data_has_changed = true;
  }

  if (device_mesh->bvh == (OptixBVH*) 0) {
    __FAILURE_HANDLE(optix_bvh_create(&device_mesh->bvh));
    __FAILURE_HANDLE(optix_bvh_gas_build(device_mesh->bvh, device, device_mesh));

    *data_has_changed = true;
  }

  return LUMINARY_SUCCESS;
}

LuminaryResult device_mesh_destroy(DeviceMesh** device_mesh) {
  __CHECK_NULL_ARGUMENT(device_mesh);
  __CHECK_NULL_ARGUMENT(*device_mesh);

  if ((*device_mesh)->vertices)
    __FAILURE_HANDLE(device_free(&(*device_mesh)->vertices));

  if ((*device_mesh)->texture_triangles)
    __FAILURE_HANDLE(device_free(&(*device_mesh)->texture_triangles));

  if ((*device_mesh)->bvh)
    __FAILURE_HANDLE(optix_bvh_destroy(&(*device_mesh)->bvh));

  if ((*device_mesh)->omm)
    __FAILURE_HANDLE(omm_destroy(&(*device_mesh)->omm));

  __FAILURE_HANDLE(host_free(device_mesh));

  return LUMINARY_SUCCESS;
}
