#include "mesh.h"

#include <string.h>

#include "bvh.h"
#include "device/light.h"
#include "internal_error.h"

LuminaryResult mesh_create(Mesh** mesh) {
  __CHECK_NULL_ARGUMENT(mesh);

  __FAILURE_HANDLE(host_malloc(mesh, sizeof(Mesh)));

  memset(*mesh, 0, sizeof(Mesh));

  __FAILURE_HANDLE(array_create(&(*mesh)->meshlets, sizeof(Meshlet), 16));

  return LUMINARY_SUCCESS;
}

LuminaryResult mesh_build_meshlets(Mesh* mesh) {
  __CHECK_NULL_ARGUMENT(mesh);

  // TODO: This is really shit if materials change all the time between triangles.

  uint16_t current_material = 0xFFFFu;
  uint16_t current_meshlet  = 0;

  for (uint32_t tri_id = 0; tri_id < mesh->data->triangle_count; tri_id++) {
    const Triangle tri = mesh->triangles[tri_id];

    const uint16_t mat_id = tri.material_id;

    // Find a meshlet to store this triangle in or create a new one
    if (mat_id != current_material) {
      uint32_t meshlet_count;
      __FAILURE_HANDLE(array_get_num_elements(mesh->meshlets, &meshlet_count));

      for (uint32_t meshlet_id = 0; meshlet_id < meshlet_count; meshlet_id++) {
        const Meshlet* meshlet = mesh->meshlets + meshlet_id;

        if (meshlet->material_id == mat_id && meshlet->triangle_count < 0x10000) {
          current_material = mat_id;
          current_meshlet  = meshlet_id;
          break;
        }
      }

      // We didn't find a valid meshlet, so lets create one.
      if (mat_id != current_material) {
        Meshlet meshlet;

        memset(&meshlet, 0, sizeof(Meshlet));

        meshlet.triangle_count = 0;
        meshlet.material_id    = mat_id;
        __FAILURE_HANDLE(host_malloc(&meshlet.index_buffer, sizeof(uint32_t) * 0x10000 * 3));
        __FAILURE_HANDLE(host_malloc(&meshlet.triangles, sizeof(Triangle) * 0x10000));

        __FAILURE_HANDLE(array_push(&mesh->meshlets, &meshlet));

        current_meshlet  = meshlet_count;
        current_material = mat_id;
      }
    }

    Meshlet* meshlet = mesh->meshlets + current_meshlet;

    meshlet->index_buffer[meshlet->triangle_count * 3 + 0] = mesh->data->index_buffer[tri_id * 3 + 0];
    meshlet->index_buffer[meshlet->triangle_count * 3 + 1] = mesh->data->index_buffer[tri_id * 3 + 1];
    meshlet->index_buffer[meshlet->triangle_count * 3 + 2] = mesh->data->index_buffer[tri_id * 3 + 2];

    meshlet->triangles[meshlet->triangle_count] = tri;

    meshlet->triangle_count++;

    // If the meshlet has run full, invalidate our bookmark so we create a new meshlet.
    if (meshlet->triangle_count == 0x10000) {
      current_material = 0xFFFFu;
    }
  }

  uint32_t meshlet_count;
  __FAILURE_HANDLE(array_get_num_elements(mesh->meshlets, &meshlet_count));

  for (uint32_t meshlet_id = 0; meshlet_id < meshlet_count; meshlet_id++) {
    Meshlet* meshlet = mesh->meshlets + meshlet_id;

    __FAILURE_HANDLE(host_realloc(&meshlet->index_buffer, sizeof(uint32_t) * meshlet->triangle_count * 3));
    __FAILURE_HANDLE(host_realloc(&meshlet->triangles, sizeof(Triangle) * meshlet->triangle_count));
  }

  log_message("Created %u meshlets.", meshlet_count);

  return LUMINARY_SUCCESS;
}

LuminaryResult mesh_destroy(Mesh** mesh) {
  __CHECK_NULL_ARGUMENT(mesh);

  uint32_t meshlet_count;
  __FAILURE_HANDLE(array_get_num_elements((*mesh)->meshlets, &meshlet_count));

  for (uint32_t meshlet_id = 0; meshlet_id < meshlet_count; meshlet_id++) {
    Meshlet meshlet = (*mesh)->meshlets[meshlet_id];

    if (meshlet.light_data) {
      __FAILURE_HANDLE(light_tree_destroy(&meshlet));
    }

    __FAILURE_HANDLE(host_free(&meshlet.triangles));
    __FAILURE_HANDLE(host_free(&meshlet.index_buffer));
  }

  __FAILURE_HANDLE(array_destroy(&(*mesh)->meshlets));

  __FAILURE_HANDLE(host_free(&(*mesh)->triangles));

  if ((*mesh)->data) {
    __FAILURE_HANDLE(host_free(&(*mesh)->data->vertex_buffer));
    __FAILURE_HANDLE(host_free(&(*mesh)->data->index_buffer));
    __FAILURE_HANDLE(host_free(&(*mesh)->data));
  }

  if ((*mesh)->bvh) {
    __FAILURE_HANDLE(bvh_destroy(&(*mesh)->bvh));
  }

  if ((*mesh)->optix_bvh) {
    // TODO: This is device implementation specific so this shouldn't even be here.
    __RETURN_ERROR(LUMINARY_ERROR_NOT_IMPLEMENTED, "");
  }

  if ((*mesh)->optix_bvh_shadow) {
    // TODO: This is device implementation specific so this shouldn't even be here.
    __RETURN_ERROR(LUMINARY_ERROR_NOT_IMPLEMENTED, "");
  }

  if ((*mesh)->optix_bvh_light) {
    // TODO: This is device implementation specific so this shouldn't even be here.
    __RETURN_ERROR(LUMINARY_ERROR_NOT_IMPLEMENTED, "");
  }

  __FAILURE_HANDLE(host_free(mesh));

  return LUMINARY_SUCCESS;
}

LuminaryResult mesh_instance_get_default(MeshInstance* instance) {
  __CHECK_NULL_ARGUMENT(instance);

  memset(instance, 0, sizeof(MeshInstance));

  instance->rotation.w = 1.0f;
  instance->scale.x    = 1.0f;
  instance->scale.y    = 1.0f;
  instance->scale.z    = 1.0f;
  instance->deleted    = false;

  return LUMINARY_SUCCESS;
}
