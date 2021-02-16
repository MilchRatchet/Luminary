#ifndef PRIMITIVES_H
#define PRIMITIVES_H

#include "image.h"

struct vec3 {
  float x;
  float y;
  float z;
} typedef vec3;

struct Quaternion {
  float w;
  float x;
  float y;
  float z;
} typedef Quaternion;

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
  RGBF emission;
  float intensity;
  float smoothness;
} typedef Sphere;

struct Cuboid {
  unsigned int id;
  vec3 pos;
  vec3 size;
  float sign;
  RGBF color;
  RGBF emission;
  float intensity;
  float smoothness;
} typedef Cuboid;

#endif /* PRIMITIVES_H */
