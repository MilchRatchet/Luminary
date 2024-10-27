#include "device_mesh.h"

#include "internal_error.h"

LuminaryResult device_mesh_create(DeviceMesh** device_mesh, const Mesh* mesh) {
  __CHECK_NULL_ARGUMENT(device_mesh);
  __CHECK_NULL_ARGUMENT(mesh);

  __FAILURE_HANDLE(host_malloc(device_mesh, sizeof(DeviceMesh)));

  uint32_t meshlet_count;
  __FAILURE_HANDLE(array_get_num_elements(mesh->meshlets, &meshlet_count));

  __FAILURE_HANDLE(array_create(&(*device_mesh)->meshlet_triangle_offsets, sizeof(uint32_t), meshlet_count));

  uint32_t triangle_count = 0;

  for (uint32_t meshlet_id = 0; meshlet_id < meshlet_count; meshlet_id++) {
    __FAILURE_HANDLE(array_push(&(*device_mesh)->meshlet_triangle_offsets, &triangle_count));

    triangle_count += mesh->meshlets[meshlet_id].triangle_count;
  }

  __FAILURE_HANDLE(array_create(&(*device_mesh)->triangles, sizeof(DeviceTriangle), triangle_count));

  for (uint32_t meshlet_id = 0; meshlet_id < meshlet_count; meshlet_id++) {
    const Meshlet* meshlet = mesh->meshlets + meshlet_id;

    DeviceTriangle triangle;
    for (uint32_t triangle_id = 0; triangle_id < meshlet->triangle_count; triangle_id++) {
      __FAILURE_HANDLE(device_struct_triangle_convert(meshlet->triangles + triangle_id, &triangle));

      __FAILURE_HANDLE(array_push(&(*device_mesh)->triangles, &triangle));
    }
  }

  return LUMINARY_SUCCESS;
}

LuminaryResult device_mesh_destroy(DeviceMesh** device_mesh) {
  __CHECK_NULL_ARGUMENT(device_mesh);
  __CHECK_NULL_ARGUMENT(*device_mesh);

  __FAILURE_HANDLE(array_destroy(&(*device_mesh)->triangles));
  __FAILURE_HANDLE(array_destroy(&(*device_mesh)->meshlet_triangle_offsets));

  __FAILURE_HANDLE(host_free(device_mesh));

  *device_mesh = (DeviceMesh*) 0;

  return LUMINARY_SUCCESS;
}
