#ifndef PRIMITIVES_H
#define PRIMITIVES_H

#include "image.h"

struct Sphere {
  float x;
  float y;
  float z;
  float radius;
  RGB8 color;
} typedef Sphere;

struct Cuboid {
  float x;
  float y;
  float z;
  float size_x;
  float size_y;
  float size_z;
  RGB8 color;
} typedef Cuboid;

#endif /* PRIMITIVES_H */
