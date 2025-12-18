#ifndef LUMINARY_DEVICE_MESH_INSTANCE_MANAGER_H
#define LUMINARY_DEVICE_MESH_INSTANCE_MANAGER_H

#include "device_utils.h"

struct Device typedef Device;
struct OptixBVH typedef OptixBVH;
struct OptixBVHInstanceCache typedef OptixBVHInstanceCache;

struct MeshInstanceProcessedUpdate {
  uint32_t instance_id;
  MeshInstance instance;
  DeviceTransform transform;
  uint32_t mesh_id;
} typedef MeshInstanceProcessedUpdate;

struct MeshInstanceManager {
  uint32_t num_instances;
  DeviceTransform* instance_transforms;
  uint32_t* instance_mesh_ids;
  ARRAY MeshInstanceProcessedUpdate* cached_updates;
} typedef MeshInstanceManager;

LuminaryResult mesh_instance_manager_create(MeshInstanceManager** manager);
LuminaryResult mesh_instance_manager_add_updates(MeshInstanceManager* manager, const ARRAY MeshInstanceUpdate* updates);
LuminaryResult mesh_instance_manager_clear_updates(MeshInstanceManager* manager);
LuminaryResult mesh_instance_manager_destroy(MeshInstanceManager** manager);

struct DeviceMeshInstanceManagerPtrs {
  CUdeviceptr vertices;
  CUdeviceptr texture_triangles;
  CUdeviceptr instance_transforms;
  CUdeviceptr instance_mesh_ids;
  OptixTraversableHandle bvh;
  OptixTraversableHandle bvh_shadow;
} typedef DeviceMeshInstanceManagerPtrs;

struct DeviceMeshInstanceManager {
  uint32_t allocated_num_meshes;
  ARRAY DeviceMesh** meshes;
  uint32_t allocated_num_instances;
  DEVICE DeviceTriangleVertex** vertices;
  DEVICE DeviceTriangleTexture** texture_triangles;
  DEVICE DeviceTransform* instance_transforms;
  DEVICE uint32_t* instance_mesh_ids;
  OptixBVHInstanceCache* optix_instance_cache;
  OptixBVH* bvh;
} typedef DeviceMeshInstanceManager;

LuminaryResult device_mesh_instance_manager_create(DeviceMeshInstanceManager** manager);
DEVICE_CTX_FUNC LuminaryResult
  device_mesh_instance_manager_add_mesh(DeviceMeshInstanceManager* manager, Device* device, const Mesh* mesh, bool* buffers_have_changed);
DEVICE_CTX_FUNC LuminaryResult device_mesh_instance_manager_update(
  DeviceMeshInstanceManager* manager, Device* device, const MeshInstanceManager* shared_manager, bool* buffers_have_changed);
DEVICE_CTX_FUNC LuminaryResult
  device_mesh_instance_manager_get_ptrs(DeviceMeshInstanceManager* manager, DeviceMeshInstanceManagerPtrs* ptrs);
DEVICE_CTX_FUNC LuminaryResult device_mesh_instance_manager_destroy(DeviceMeshInstanceManager** manager);

#endif /* LUMINARY_DEVICE_MESH_INSTANCE_MANAGER_H */
