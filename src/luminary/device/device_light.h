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
  uint32_t instance_id;
  uint32_t tri_id;
  float power;
  uint32_t padding;
} typedef LightTreeFragment;
LUM_STATIC_SIZE_ASSERT(LightTreeFragment, 0x40);

struct LightTreeCacheTriangle {
  uint32_t tri_id;
  Vec128 vertex;
  Vec128 vertex1;
  Vec128 vertex2;
  Vec128 cross;
} typedef LightTreeCacheTriangle;

struct LightTreeCacheMesh {
  bool is_dirty;
  uint32_t instance_count;
  bool has_emission;
  ARRAY uint16_t* materials;
  ARRAY LightTreeCacheTriangle* ARRAY* triangles;
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

struct LightTreeCacheMaterial {
  bool is_dirty;
  bool has_emission;
  uint64_t emission_texture_hash;
  bool has_textured_emission;
  float constant_emission_intensity;
} typedef LightTreeCacheMaterial;

struct LightTreeCache {
  bool is_dirty;
  ARRAY LightTreeCacheMesh* meshes;
  ARRAY LightTreeCacheInstance* instances;
  ARRAY LightTreeCacheMaterial* materials;
} typedef LightTreeCache;

struct LightTree {
  uint32_t build_id;
  LightTreeCache cache;
  void* nodes_data;
  size_t nodes_size;
  void* paths_data;
  size_t paths_size;
  void* tri_handle_map_data;
  size_t tri_handle_map_size;
  HashMap* hash_map;
} typedef LightTree;

struct Device typedef Device;

LuminaryResult light_tree_create(LightTree** tree);

/*
 * This requires that all meshes, materials and textures are already synced with the device.
 */
LuminaryResult light_tree_update_cache_mesh(LightTree* tree, const Mesh* mesh);
LuminaryResult light_tree_update_cache_instance(LightTree* tree, const MeshInstance* instance);
LuminaryResult light_tree_update_cache_material(LightTree* tree, const Material* material);

LuminaryResult light_tree_build(LightTree* tree);

LuminaryResult light_tree_destroy(LightTree** tree);

#endif /* LUMINARY_LIGHT_H */
