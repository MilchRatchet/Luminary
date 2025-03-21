#include "device_mesh.h"

#include "device.h"
#include "device_staging_manager.h"
#include "internal_error.h"
#include "struct_interleaving.h"

LuminaryResult device_mesh_create(DeviceMesh** device_mesh, Device* device, const Mesh* mesh) {
  __CHECK_NULL_ARGUMENT(device_mesh);
  __CHECK_NULL_ARGUMENT(device);
  __CHECK_NULL_ARGUMENT(mesh);

  __FAILURE_HANDLE(host_malloc(device_mesh, sizeof(DeviceMesh)));
  memset(*device_mesh, 0, sizeof(DeviceMesh));

  (*device_mesh)->mesh = mesh;

  const uint32_t triangle_count = mesh->data.triangle_count;

  __FAILURE_HANDLE(device_malloc(&(*device_mesh)->triangles, sizeof(DeviceTriangle) * triangle_count));

  DeviceTriangle* direct_access_buffer;
  __FAILURE_HANDLE(device_staging_manager_register_direct_access(
    device->staging_manager, (*device_mesh)->triangles, 0, sizeof(DeviceTriangle) * triangle_count, (void**) &direct_access_buffer));

  DeviceTriangle* device_triangles;
  __FAILURE_HANDLE(host_malloc(&device_triangles, sizeof(DeviceTriangle) * triangle_count));

  for (uint32_t triangle_id = 0; triangle_id < triangle_count; triangle_id++) {
    __FAILURE_HANDLE(device_struct_triangle_convert(mesh->triangles + triangle_id, device_triangles + triangle_id));
  }

  __FAILURE_HANDLE(struct_triangles_interleave(direct_access_buffer, device_triangles, triangle_count));

  __FAILURE_HANDLE(host_free(&device_triangles));

  __FAILURE_HANDLE(optix_bvh_create(&(*device_mesh)->bvh));

  (*device_mesh)->bvh_is_dirty = true;

  return LUMINARY_SUCCESS;
}

LuminaryResult device_mesh_build_structures(DeviceMesh* device_mesh, OpacityMicromap* omm, Device* device) {
  __CHECK_NULL_ARGUMENT(device_mesh);
  __CHECK_NULL_ARGUMENT(omm);
  __CHECK_NULL_ARGUMENT(device);

  if (device_mesh->bvh_is_dirty == false)
    return LUMINARY_SUCCESS;

  __FAILURE_HANDLE(omm_build(omm, device_mesh->mesh, device));
  __FAILURE_HANDLE(optix_bvh_gas_build(device_mesh->bvh, device, device_mesh->mesh, omm));

  device_mesh->bvh_is_dirty = false;

  return LUMINARY_SUCCESS;
}

LuminaryResult device_mesh_destroy(DeviceMesh** device_mesh) {
  __CHECK_NULL_ARGUMENT(device_mesh);
  __CHECK_NULL_ARGUMENT(*device_mesh);

  __FAILURE_HANDLE(device_free(&(*device_mesh)->triangles));
  __FAILURE_HANDLE(optix_bvh_destroy(&(*device_mesh)->bvh));

  __FAILURE_HANDLE(host_free(device_mesh));

  *device_mesh = (DeviceMesh*) 0;

  return LUMINARY_SUCCESS;
}
