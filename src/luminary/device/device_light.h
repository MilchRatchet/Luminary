#ifndef LUMINARY_LIGHT_H
#define LUMINARY_LIGHT_H

#include "device_memory.h"
#include "device_utils.h"
#include "host_intrinsics.h"
#include "mesh.h"

struct LightTreeFragment {
  Vec128 high;
  Vec128 low;
  Vec128 middle;
  uint32_t instance_id;
  uint32_t tri_id;
  float power;
  uint32_t padding;
} typedef LightTreeFragment;
LUM_STATIC_SIZE_ASSERT(LightTreeFragment, 0x40);

struct LightTreeCacheTriangle {
  uint64_t emission_texture_hash;
  bool has_textured_emission;
  float constant_emission_intensity;
  Vec128 cross_product;
  Vec128 vertex;
  Vec128 edge1;
  Vec128 edge2;
} typedef LightTreeCacheTriangle;

struct LightTreeCacheMesh {
  uint32_t instance_count;
  bool has_emission;
  ARRAY LightTreeCacheTriangle* triangles;
  bool is_dirty;
} typedef LightTreeCacheMesh;

struct LightTreeCacheInstance {
  bool active;
  uint32_t mesh_id;
  Quaternion rotation;
  vec3 scale;
  vec3 translation;
  bool is_dirty;
  ARRAY LightTreeFragment* fragments; /* Computed during build step */
} typedef LightTreeCacheInstance;

struct LightTreeCache {
  bool is_dirty;
  ARRAY LightTreeCacheMesh* meshes;
  ARRAY LightTreeCacheInstance* instances;
} typedef LightTreeCache;

struct LightTree {
  uint32_t build_id;
  LightTreeCache cache;
  void* nodes_data;
  size_t nodes_size;
  void* paths_data;
  size_t paths_size;
  void* light_instance_map_data;
  size_t light_instance_map_size;
  float bounding_sphere_size;
  vec3 bounding_sphere_center;
} typedef LightTree;

struct Device typedef Device;

LuminaryResult light_tree_create(LightTree** tree);

/*
 * This requires that all meshes, materials and textures are already synced with the device.
 */
DEVICE_CTX_FUNC LuminaryResult
  light_tree_update_cache_mesh(LightTree* tree, Device* device, const Mesh* mesh, const ARRAY Material* materials);

LuminaryResult light_tree_update_cache_instance(LightTree* tree, const MeshInstance* instance);

LuminaryResult light_tree_build(LightTree* tree);

LuminaryResult light_tree_destroy(LightTree** tree);

#endif /* LUMINARY_LIGHT_H */
