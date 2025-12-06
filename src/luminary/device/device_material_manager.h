#ifndef LUMINARY_DEVICE_MATERIAL_MANAGER_H
#define LUMINARY_DEVICE_MATERIAL_MANAGER_H

#include "device_utils.h"

struct Device typedef Device;

struct MaterialProcessedUpdate {
  uint32_t material_id;
  DeviceMaterialCompressed material;
} typedef MaterialProcessedUpdate;

struct MaterialManager {
  uint32_t num_materials;
  DeviceMaterialCompressed* materials;
  ARRAY MaterialProcessedUpdate* cached_updates;
} typedef MaterialManager;

LuminaryResult material_manager_create(MaterialManager** manager);
LuminaryResult material_manager_add_updates(MaterialManager* manager, const ARRAY MaterialUpdate* updates);
LuminaryResult material_manager_clear_updates(MaterialManager* manager);
LuminaryResult material_manager_destroy(MaterialManager** manager);

struct DeviceMaterialManagerPtrs {
  CUdeviceptr materials;
} typedef DeviceMaterialManagerPtrs;

struct DeviceMaterialManager {
  uint32_t allocated_num_materials;
  DEVICE DeviceMaterialCompressed* materials;
} typedef DeviceMaterialManager;

LuminaryResult device_material_manager_create(DeviceMaterialManager** manager);
DEVICE_CTX_FUNC LuminaryResult device_material_manager_update(
  DeviceMaterialManager* manager, Device* device, const MaterialManager* shared_manager, bool* buffers_have_changed);
DEVICE_CTX_FUNC LuminaryResult device_material_manager_get_ptrs(DeviceMaterialManager* manager, DeviceMaterialManagerPtrs* ptrs);
DEVICE_CTX_FUNC LuminaryResult device_material_manager_destroy(DeviceMaterialManager** manager);

#endif /* LUMINARY_DEVICE_MATERIAL_MANAGER_H */
