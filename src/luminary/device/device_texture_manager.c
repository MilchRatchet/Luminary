#include "device_texture_manager.h"

#include "device.h"
#include "internal_error.h"

LuminaryResult device_texture_manager_create(DeviceTextureManager** manager) {
  __CHECK_NULL_ARGUMENT(manager);

  __FAILURE_HANDLE(host_malloc(manager, sizeof(DeviceTextureManager)));
  memset(*manager, 0, sizeof(DeviceTextureManager));

  __FAILURE_HANDLE(array_create(&(*manager)->textures, sizeof(DeviceTexture*), 16));

  return LUMINARY_SUCCESS;
}

DEVICE_CTX_FUNC LuminaryResult device_texture_manager_add(
  DeviceTextureManager* manager, Device* device, const Texture** textures, uint32_t num_textures, bool* buffers_have_changed) {
  __CHECK_NULL_ARGUMENT(manager);
  __CHECK_NULL_ARGUMENT(device);
  __CHECK_NULL_ARGUMENT(textures);

  *buffers_have_changed = false;

  if (num_textures == 0)
    return LUMINARY_SUCCESS;

  for (uint32_t texture_id = 0; texture_id < num_textures; texture_id++) {
    DeviceTexture* device_texture;
    __FAILURE_HANDLE(device_texture_create(&device_texture, textures[texture_id], device, device->stream_main));

    __FAILURE_HANDLE(array_push(&manager->textures, &device_texture));
  }

  uint32_t texture_object_count;
  __FAILURE_HANDLE(array_get_num_elements(manager->textures, &texture_object_count));

  if (manager->texture_objs)
    __FAILURE_HANDLE(device_free(&manager->texture_objs));

  __FAILURE_HANDLE(device_malloc(&manager->texture_objs, sizeof(DeviceTextureObject) * texture_object_count));

  DeviceTextureObject* buffer;
  __FAILURE_HANDLE(device_staging_manager_register_direct_access(
    device->staging_manager, (void*) manager->texture_objs, 0, sizeof(DeviceTextureObject) * texture_object_count, (void**) &buffer));

  for (uint32_t texture_object_id = 0; texture_object_id < texture_object_count; texture_object_id++) {
    __FAILURE_HANDLE(device_struct_texture_object_convert(manager->textures[texture_object_id], buffer + texture_object_id));
  }

  *buffers_have_changed = true;

  return LUMINARY_SUCCESS;
}

DEVICE_CTX_FUNC LuminaryResult device_texture_manager_get_ptrs(DeviceTextureManager* manager, DeviceTextureManagerPtrs* ptrs) {
  __CHECK_NULL_ARGUMENT(manager);
  __CHECK_NULL_ARGUMENT(ptrs);

  ptrs->textures = DEVICE_CUPTR(manager->texture_objs);

  return LUMINARY_SUCCESS;
}

DEVICE_CTX_FUNC LuminaryResult device_texture_manager_destroy(DeviceTextureManager** manager) {
  __CHECK_NULL_ARGUMENT(manager);

  uint32_t num_textures;
  __FAILURE_HANDLE(array_get_num_elements((*manager)->textures, &num_textures));

  for (uint32_t texture_id = 0; texture_id < num_textures; texture_id++) {
    __FAILURE_HANDLE(device_texture_destroy(&(*manager)->textures[texture_id]));
  }

  __FAILURE_HANDLE(array_destroy(&(*manager)->textures));

  if ((*manager)->texture_objs)
    __FAILURE_HANDLE(device_free(&(*manager)->texture_objs));

  __FAILURE_HANDLE(host_free(manager));

  return LUMINARY_SUCCESS;
}
