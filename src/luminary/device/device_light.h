#ifndef LUMINARY_LIGHT_H
#define LUMINARY_LIGHT_H

#include "device_memory.h"
#include "device_utils.h"
#include "hashmap.h"
#include "host_intrinsics.h"
#include "mesh.h"

struct Device typedef Device;
struct OptixBVH typedef OptixBVH;

struct LightTreeBVHTriangle {
  Vec128 vertex;
  Vec128 vertex1;
  Vec128 vertex2;
} typedef LightTreeBVHTriangle;
LUM_STATIC_SIZE_ASSERT(LightTreeBVHTriangle, 3 * sizeof(Vec128));

struct LightTreeFragment {
  Vec128 high;
  Vec128 low;
  Vec128 middle;
  Vec128 average_direction;  // Normal, but flipped to be in Z+ orientation and scaled by 0.5
  Vec128 v0;                 // TODO: These are only used for spatial variance, if this is a performance issue, rethink it
  Vec128 v1;
  Vec128 v2;
  uint32_t instance_id;
  uint32_t material_slot_id;
  uint32_t material_tri_id;
  float power;
  uint32_t instance_cache_tri_id;
  float intensity;
  uint32_t padding1;
  uint32_t padding2;
} typedef LightTreeFragment;
LUM_STATIC_SIZE_ASSERT(LightTreeFragment, 0x90);

struct LightTreeCacheTriangle {
  uint32_t tri_id;
  Vec128 vertex;
  Vec128 vertex1;
  Vec128 vertex2;
  Vec128 cross;
  float average_intensity;
  float importance_normalization;
} typedef LightTreeCacheTriangle;

struct LightTreeCacheMesh {
  bool is_dirty;
  uint32_t instance_count;
  bool has_emission;
  ARRAY uint16_t* materials;
  ARRAY LightTreeCacheTriangle * ARRAY * material_triangles;
} typedef LightTreeCacheMesh;

struct LightTreeCacheInstance {
  bool active;
  uint32_t mesh_id;
  Quaternion rotation;
  vec3 scale;
  vec3 translation;
  bool is_dirty;
  ARRAY LightTreeFragment* fragments;        /* Computed during build step */
  ARRAY LightTreeBVHTriangle* bvh_triangles; /* Computed during build step */
} typedef LightTreeCacheInstance;

struct LightTreeCacheMaterial {
  bool is_dirty;
  bool has_emission;
  uint64_t emission_texture_hash;
  bool has_textured_emission;
  bool needs_reintegration;
  float constant_emission_intensity;
} typedef LightTreeCacheMaterial;

struct LightTreeCache {
  bool is_dirty;
  ARRAY LightTreeCacheMesh* meshes;
  ARRAY LightTreeCacheInstance* instances;
  ARRAY LightTreeCacheMaterial* materials;
} typedef LightTreeCache;

struct LightTreeIntegratorTask {
  uint32_t mesh_id;
  uint32_t material_slot_id;
  uint32_t material_slot_tri_id;
  uint32_t triangle_id;
} typedef LightTreeIntegratorTask;

struct LightTreeIntegrator {
  ARRAY LightTreeIntegratorTask* tasks;
  uint32_t* mesh_ids;
  uint32_t* triangle_ids;
  float* intensities;
  DEVICE uint32_t* device_mesh_ids;
  DEVICE uint32_t* device_triangle_ids;
  DEVICE float* device_intensities;
  uint32_t allocated_tasks;
} typedef LightTreeIntegrator;

struct LightTree {
  uint32_t build_id;
  LightTreeCache cache;
  LightTreeIntegrator integrator;
  void* root_data;
  size_t root_size;
  void* nodes_data;
  size_t nodes_size;
  void* tri_handle_map_data;
  void* bvh_vertex_buffer_data;
  uint32_t light_count;
} typedef LightTree;

LuminaryResult light_tree_create(LightTree** tree);

/*
 * This requires that all meshes, materials and textures are already synced with the device.
 */
LuminaryResult light_tree_update_cache_mesh(LightTree* tree, const Mesh* mesh);
LuminaryResult light_tree_update_cache_instance(LightTree* tree, const MeshInstanceUpdate* instance_update);
LuminaryResult light_tree_update_cache_material(LightTree* tree, const MaterialUpdate* material_update);

DEVICE_CTX_FUNC LuminaryResult light_tree_build(LightTree* tree, Device* device);
DEVICE_CTX_FUNC LuminaryResult light_tree_unload_integrator(LightTree* tree);

LuminaryResult light_tree_destroy(LightTree** tree);

struct DeviceLightTreePtrs {
  CUdeviceptr root;
  CUdeviceptr nodes;
  CUdeviceptr tri_handle_map;
  OptixTraversableHandle bvh;
} typedef DeviceLightTreePtrs;

struct DeviceLightTree {
  uint32_t build_id;
  size_t allocated_root_size;
  size_t allocated_nodes_size;
  uint32_t allocated_light_count;
  DEVICE DeviceLightTreeRootHeader* root;
  DEVICE DeviceLightTreeNode* nodes;
  DEVICE TriangleHandle* tri_handle_map;
  DEVICE LightTreeBVHTriangle* bvh_vertex_buffer;
  OptixBVH* bvh;
} typedef DeviceLightTree;

LuminaryResult device_light_tree_create(DeviceLightTree** tree);
DEVICE_CTX_FUNC LuminaryResult
  device_light_tree_update(DeviceLightTree* tree, Device* device, const LightTree* shared_tree, bool* buffers_have_changed);
DEVICE_CTX_FUNC LuminaryResult device_light_tree_get_ptrs(DeviceLightTree* tree, DeviceLightTreePtrs* ptrs);
DEVICE_CTX_FUNC LuminaryResult device_light_tree_destroy(DeviceLightTree** tree);

#endif /* LUMINARY_LIGHT_H */
