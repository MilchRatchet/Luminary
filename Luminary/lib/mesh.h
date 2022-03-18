#ifndef MESH_H
#define MESH_H

#include <stdint.h>

#include "primitives.h"

struct UV {
  float u;
  float v;
} typedef UV;

struct Triangle {
  vec3 vertex;
  vec3 edge1;
  vec3 edge2;
  vec3 vertex_normal;
  vec3 edge1_normal;
  vec3 edge2_normal;
  UV vertex_texture;
  UV edge1_texture;
  UV edge2_texture;
  uint32_t object_maps;
  float padding1;
  float padding2;
  float padding3;
} typedef Triangle;

struct TraversalTriangle {
  vec3 vertex;
  vec3 edge1;
  vec3 edge2;
  vec3 face_normal;
} typedef TraversalTriangle;

#endif /* MESH_H */
