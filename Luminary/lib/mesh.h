#ifndef MESH_H
#define MESH_H

#include "primitives.h"
#include "stdint.h"

struct UV {
  float u;
  float v;
} typedef UV;

struct Triangle {
  vec3 vertex;
  vec3 edge1;
  vec3 edge2;
  UV vertex_texture;
  UV edge1_texture;
  UV edge2_texture;
  vec3 vertex_normal;
  vec3 edge1_normal;
  vec3 edge2_normal;
  uint16_t object_maps;
} typedef Triangle;

#endif /* MESH_H */
