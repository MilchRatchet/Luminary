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
  vec3 vertex_normal;
  vec3 edge1_normal;
  vec3 edge2_normal;
  UV vertex_texture;
  UV edge1_texture;
  UV edge2_texture;
  vec3 face_normal;
  uint32_t object_maps;
} typedef Triangle;

struct Traversal_Triangle {
  vec4 vertex;
  vec4 edge1;
  vec4 edge2;
} typedef Traversal_Triangle;

#endif /* MESH_H */
