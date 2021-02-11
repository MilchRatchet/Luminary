#ifndef PRIMITIVES_H
#define PRIMITIVES_H

#include "image.h"

struct vec3 {
  float x;
  float y;
  float z;
} typedef vec3;

/*
 * IDs unique identify each object
 * 0 is reserved
 */

struct Sphere {
  unsigned int id;
  vec3 pos;
  float radius;
  float sign;
  RGBF color;
  float smoothness;
} typedef Sphere;

struct Cuboid {
  unsigned int id;
  vec3 pos;
  vec3 size;
  float sign;
  RGBF color;
  float smoothness;
} typedef Cuboid;

struct Light {
  unsigned int id;
  vec3 pos;
  float intensity;
  RGBF color;
} typedef Light;

#endif /* PRIMITIVES_H */
