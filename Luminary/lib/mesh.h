#ifndef MESH_H
#define MESH_H

#include "primitives.h"

struct UV {
  float u;
  float v;
} typedef UV;

struct Triangle {
  vec3 v1;
  vec3 v2;
  vec3 v3;
  UV vt1;
  UV vt2;
  UV vt3;
  vec3 vn1;
  vec3 vn2;
  vec3 vn3;
} typedef Triangle;

#endif /* MESH_H */
