#ifndef LUMINARY_LIGHT_H
#define LUMINARY_LIGHT_H

#include "device_memory.h"
#include "device_utils.h"
#include "mesh.h"

struct Device typedef Device;

// Light Tree construction flow:
// Mesh is dirty on Host => Host queues main device with light tree construction for mesh
// LightTree construction on device is done => queues signal work on host
// Host keeps track of in flight light tree constructions
// If counter reaches 0 inside the signal work, then queue build top level light tree
// Once top level light tree is done => queue upload light tree to all devices

struct LightTree {
  void* nodes_data;
  size_t nodes_size;
  void* paths_data;
  size_t paths_size;
  void* child_remapper_data;
  size_t child_remapper_size;
  float bounding_sphere_size;
  vec3 bounding_sphere_center;
  float normalized_power;
} typedef LightTree;

struct MeshletLightData {
  LightTree* light_tree;
  TriangleLight* lights;
} typedef MeshletLightData;

LuminaryResult light_load_bridge_lut(Device* device);
LuminaryResult light_tree_create(Meshlet* meshlet, Device* device, ARRAY const Material** materials);
LuminaryResult light_tree_create_toplevel(
  LightTree** tree, ARRAY const MeshInstance** instances, ARRAY const Mesh** meshes, ARRAY const Material** materials);
LuminaryResult light_tree_destroy(Meshlet* meshlet);
LuminaryResult light_tree_destroy_toplevel(LightTree** tree);

#endif /* LUMINARY_LIGHT_H */
