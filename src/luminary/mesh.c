#include "mesh.h"

#define _USE_MATH_DEFINES
#include <math.h>
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

  (*mesh)->bounding_box =
    (MeshBoundingBox) {.min = vec128_set(-FLT_MAX, -FLT_MAX, -FLT_MAX, 0.0f), .max = vec128_set(FLT_MAX, FLT_MAX, FLT_MAX, 0.0f)};

  return LUMINARY_SUCCESS;
}

LuminaryResult mesh_set_name(Mesh* mesh, const char* name) {
  __CHECK_NULL_ARGUMENT(mesh);
  __CHECK_NULL_ARGUMENT(name);

  const size_t string_length = strlen(name);

  if (mesh->name) {
    __FAILURE_HANDLE(host_free(&mesh->name));
  }

  __FAILURE_HANDLE(host_malloc(&mesh->name, string_length + 1));

  memcpy(mesh->name, name, string_length);
  mesh->name[string_length] = '\0';

  return LUMINARY_SUCCESS;
}

LuminaryResult mesh_destroy(Mesh** mesh) {
  __CHECK_NULL_ARGUMENT(mesh);

  __FAILURE_HANDLE(host_free(&(*mesh)->triangles));

  if ((*mesh)->name) {
    __FAILURE_HANDLE(host_free(&(*mesh)->name));
  }

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

  instance->active  = true;
  instance->id      = instance_id_counter++;
  instance->scale.x = 1.0f;
  instance->scale.y = 1.0f;
  instance->scale.z = 1.0f;

  return LUMINARY_SUCCESS;
}

#define __INSTANCE_DIRTY(var)     \
  {                               \
    if (input->var != old->var) { \
      *dirty = true;              \
      return LUMINARY_SUCCESS;    \
    }                             \
  }

LuminaryResult mesh_instance_check_for_dirty(const MeshInstance* input, const MeshInstance* old, bool* dirty) {
  __CHECK_NULL_ARGUMENT(input);
  __CHECK_NULL_ARGUMENT(old);
  __CHECK_NULL_ARGUMENT(dirty);

  *dirty = false;

  __INSTANCE_DIRTY(active);

  if (input->active) {
    __INSTANCE_DIRTY(translation.x);
    __INSTANCE_DIRTY(translation.y);
    __INSTANCE_DIRTY(translation.z);

    __INSTANCE_DIRTY(scale.x);
    __INSTANCE_DIRTY(scale.y);
    __INSTANCE_DIRTY(scale.z);

    __INSTANCE_DIRTY(rotation.x);
    __INSTANCE_DIRTY(rotation.y);
    __INSTANCE_DIRTY(rotation.z);

    __INSTANCE_DIRTY(mesh_id);
  }

  return LUMINARY_SUCCESS;
}

LuminaryResult mesh_instance_from_public_api_instance(MeshInstance* mesh_instance, const LuminaryInstance* instance) {
  __CHECK_NULL_ARGUMENT(mesh_instance);
  __CHECK_NULL_ARGUMENT(instance);

  mesh_instance->id          = instance->id;
  mesh_instance->mesh_id     = instance->mesh_id;
  mesh_instance->translation = instance->position;
  mesh_instance->scale       = instance->scale;
  mesh_instance->rotation    = instance->rotation;
  mesh_instance->active      = true;

  return LUMINARY_SUCCESS;
}

LuminaryResult mesh_instance_to_public_api_instance(LuminaryInstance* instance, const MeshInstance* mesh_instance) {
  __CHECK_NULL_ARGUMENT(instance);
  __CHECK_NULL_ARGUMENT(mesh_instance);

  const LuminaryInstance converted_instance = (LuminaryInstance) {.id       = mesh_instance->id,
                                                                  .mesh_id  = mesh_instance->mesh_id,
                                                                  .position = mesh_instance->translation,
                                                                  .scale    = mesh_instance->scale,
                                                                  .rotation = mesh_instance->rotation};

  memcpy(instance, &converted_instance, sizeof(LuminaryInstance));

  return LUMINARY_SUCCESS;
}
