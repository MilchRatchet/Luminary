#ifndef PRIMITIVES_H
#define PRIMITIVES_H

#include "image.h"

/*
 * IDs unique identify each object
 * 0 is reserved
 */

struct Sphere {
  unsigned int id;
  float x;
  float y;
  float z;
  float radius;
  RGB8 color;
} typedef Sphere;

struct Cuboid {
  unsigned int id;
  float x;
  float y;
  float z;
  float size_x;
  float size_y;
  float size_z;
  RGB8 color;
} typedef Cuboid;

struct Light {
  unsigned int id;
  float x;
  float y;
  float z;
  float intensity;
  RGB8 color;
} typedef Light;

#endif /* PRIMITIVES_H */
