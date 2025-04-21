#ifndef LUMINARY_LIGHT_H
#define LUMINARY_LIGHT_H

#include "device_memory.h"
#include "device_utils.h"
#include "hashmap.h"
#include "host_intrinsics.h"
#include "mesh.h"

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
  DeviceLightMicroTriangleImportance microtriangle_importance;
  float importance_normalization;
} typedef LightTreeCacheTriangle;

struct LightTreeCacheMesh {
  bool is_dirty;
  uint32_t instance_count;
  bool has_emission;
  ARRAY uint16_t* materials;
  ARRAY LightTreeCacheTriangle* ARRAY* material_triangles;
} typedef LightTreeCacheMesh;

struct LightTreeBVHTriangle {
  Vec128 vertex;
  Vec128 vertex1;
  Vec128 vertex2;
} typedef LightTreeBVHTriangle;
LUM_STATIC_SIZE_ASSERT(LightTreeBVHTriangle, 3 * sizeof(Vec128));

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
  uint8_t* microtriangle_importance;
  float* importance_normalization;
  float* intensities;
  DEVICE uint32_t* device_mesh_ids;
  DEVICE uint32_t* device_triangle_ids;
  DEVICE float* device_microtriangle_importance;
  DEVICE float* device_importance_normalization;
  DEVICE float* device_intensities;
  uint32_t allocated_tasks;
} typedef LightTreeIntegrator;

struct LightTree {
  uint32_t build_id;
  LightTreeCache cache;
  LightTreeIntegrator integrator;
  void* nodes_data;
  size_t nodes_size;
  void* paths_data;
  size_t paths_size;
  void* tri_handle_map_data;
  size_t tri_handle_map_size;
  void* leaves_data;
  size_t leaves_size;
  void* importance_normalization_data;
  size_t importance_normalization_size;
  void* microtriangle_data;
  size_t microtriangle_size;
  void* bvh_vertex_buffer_data;
  uint32_t light_count;
} typedef LightTree;

struct Device typedef Device;

LuminaryResult light_tree_create(LightTree** tree);

/*
 * This requires that all meshes, materials and textures are already synced with the device.
 */
LuminaryResult light_tree_update_cache_mesh(LightTree* tree, const Mesh* mesh);
LuminaryResult light_tree_update_cache_instance(LightTree* tree, const MeshInstance* instance);
LuminaryResult light_tree_update_cache_material(LightTree* tree, const Material* material);

DEVICE_CTX_FUNC LuminaryResult light_tree_build(LightTree* tree, Device* device);

LuminaryResult light_tree_destroy(LightTree** tree);

#endif /* LUMINARY_LIGHT_H */
