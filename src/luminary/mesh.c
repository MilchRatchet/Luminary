#include "mesh.h"

#include <string.h>

#include "bvh.h"
#include "device/device_light.h"
#include "internal_error.h"

// TODO: Make per host instance
static uint32_t mesh_id_counter     = 0;
static uint32_t instance_id_counter = 0;

LuminaryResult mesh_create(Mesh** mesh) {
  __CHECK_NULL_ARGUMENT(mesh);

  __FAILURE_HANDLE(host_malloc(mesh, sizeof(Mesh)));
  memset(*mesh, 0, sizeof(Mesh));

  (*mesh)->id = mesh_id_counter++;

  return LUMINARY_SUCCESS;
}

LuminaryResult mesh_destroy(Mesh** mesh) {
  __CHECK_NULL_ARGUMENT(mesh);

  __FAILURE_HANDLE(host_free(&(*mesh)->triangles));

  if ((*mesh)->data.vertex_buffer) {
    __FAILURE_HANDLE(host_free(&(*mesh)->data.vertex_buffer));
  }

  if ((*mesh)->data.index_buffer) {
    __FAILURE_HANDLE(host_free(&(*mesh)->data.index_buffer));
  }

  __FAILURE_HANDLE(host_free(mesh));

  return LUMINARY_SUCCESS;
}

LuminaryResult mesh_instance_get_default(MeshInstance* instance) {
  __CHECK_NULL_ARGUMENT(instance);

  memset(instance, 0, sizeof(MeshInstance));

  instance->active     = true;
  instance->id         = instance_id_counter++;
  instance->rotation.w = 1.0f;
  instance->scale.x    = 1.0f;
  instance->scale.y    = 1.0f;
  instance->scale.z    = 1.0f;

  return LUMINARY_SUCCESS;
}
