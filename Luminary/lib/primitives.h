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
  RGBF color;
} typedef Sphere;

struct Cuboid {
  unsigned int id;
  float x;
  float y;
  float z;
  float size_x;
  float size_y;
  float size_z;
  RGBF color;
} typedef Cuboid;

struct Light {
  unsigned int id;
  float x;
  float y;
  float z;
  float intensity;
  RGBF color;
} typedef Light;

#endif /* PRIMITIVES_H */
