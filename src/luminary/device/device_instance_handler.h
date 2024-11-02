#ifndef LUMINARY_DEVICE_INSTANCE_HANDLER_H
#define LUMINARY_DEVICE_INSTANCE_HANDLER_H

#include "device_utils.h"

struct DeviceInstanceHandler {
  uint32_t num_instancelets;
  ARRAYPTR uint32_t** instance_maps;
  ARRAY bool* instancelet_active;
  ARRAY DeviceInstancelet* instancelets;
  ARRAY DeviceTransform* transforms;
} typedef DeviceInstanceHandler;

LuminaryResult device_instance_handler_create(DeviceInstanceHandler** device_instance_handler);
LuminaryResult device_instance_handler_update(
  DeviceInstanceHandler* device_instance_handler, MeshInstanceUpdate* update, ARRAYPTR uint32_t** instancelets_dirty,
  const ARRAY DeviceMesh** device_meshes);
LuminaryResult device_instance_handler_destroy(DeviceInstanceHandler** device_instance_handler);

#endif /* LUMINARY_DEVICE_INSTANCE_HANDLER_H */
