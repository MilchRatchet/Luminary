#ifndef LUMINARY_LIGHT_H
#define LUMINARY_LIGHT_H

#include "device_memory.h"
#include "device_utils.h"
#include "mesh.h"

struct LightTreeCacheTriangle {
  uint64_t emission_texture_hash;
  bool has_textured_emission;
  float constant_emission_intensity;
  vec3 cross_product;
} typedef LightTreeCacheTriangle;

struct LightTreeCacheMesh {
  uint32_t instance_count;
  ARRAY LightTreeCacheTriangle* triangles;
} typedef LightTreeCacheMesh;

struct LightTreeCacheInstance {
  bool active;
  uint32_t mesh_id;
  Quaternion rotation;
  vec3 scale;
  vec3 translation;
} typedef LightTreeCacheInstance;

struct LightTreeCache {
  bool has_changed;
  ARRAY LightTreeCacheMesh* meshes;
  ARRAY LightTreeCacheInstance* instances;
} typedef LightTreeCache;

struct LightTree {
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
