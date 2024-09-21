#ifndef LUMINARY_LIGHT_H
#define LUMINARY_LIGHT_H

#include "mesh.h"
#include "utils.h"

// Light Tree construction flow:
// Mesh is dirty on Host => Host queues main device with light tree construction for mesh
// LightTree construction on device is done => queues signal work on host
// Host keeps track of in flight light tree constructions
// If counter reaches 0 inside the signal work, then queue build top level light tree
// Once top level light tree is done => queue upload light tree to all devices

struct LightTree {
  void* data;
  size_t size;
} typedef LightTree;

struct LightData {
  LightTree* light_tree;
  TriangleLight* lights;
  uint32_t light_count;
} typedef LightData;

LuminaryResult light_tree_create(LightTree** tree, Mesh* mesh);

void lights_process(Scene* scene, int dmm_active);
void lights_load_bridge_lut();

#endif /* LUMINARY_LIGHT_H */
