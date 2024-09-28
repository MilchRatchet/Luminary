#include "mesh.h"

#include <string.h>

#include "bvh.h"
#include "internal_error.h"

LuminaryResult mesh_create(Mesh** mesh) {
  __CHECK_NULL_ARGUMENT(mesh);

  __FAILURE_HANDLE(host_malloc(mesh, sizeof(Mesh)));

  memset(*mesh, 0, sizeof(Mesh));

  return LUMINARY_SUCCESS;
}

LuminaryResult mesh_destroy(Mesh** mesh) {
  __CHECK_NULL_ARGUMENT(mesh);

  if ((*mesh)->data) {
    __FAILURE_HANDLE(host_free(&(*mesh)->data->vertex_buffer));
    __FAILURE_HANDLE(host_free(&(*mesh)->data->index_buffer));
    __FAILURE_HANDLE(host_free(&(*mesh)->data));
  }

  if ((*mesh)->light_data) {
    __FAILURE_HANDLE(host_free(&(*mesh)->light_data));

    // TODO: Call light_destroy.
    __RETURN_ERROR(LUMINARY_ERROR_NOT_IMPLEMENTED, "");
  }

  if ((*mesh)->bvh) {
    __FAILURE_HANDLE(bvh_destroy(&(*mesh)->bvh));
  }

  if ((*mesh)->optix_bvh) {
    // TODO: This is device implementation specific so this shouldn't even be here.
    __RETURN_ERROR(LUMINARY_ERROR_NOT_IMPLEMENTED, "");
  }

  __FAILURE_HANDLE(host_free(mesh));

  return LUMINARY_SUCCESS;
}
