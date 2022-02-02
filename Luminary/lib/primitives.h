#ifndef PRIMITIVES_H
#define PRIMITIVES_H

struct vec3 {
  float x;
  float y;
  float z;
} typedef vec3;

struct vec4 {
  float x;
  float y;
  float z;
  float w;
} typedef vec4;

struct Quaternion {
  float w;
  float x;
  float y;
  float z;
} typedef Quaternion;

struct Mat4x4 {
  float f11;
  float f12;
  float f13;
  float f14;
  float f21;
  float f22;
  float f23;
  float f24;
  float f31;
  float f32;
  float f33;
  float f34;
  float f41;
  float f42;
  float f43;
  float f44;
} typedef Mat4x4;

struct Mat3x3 {
  float f11;
  float f12;
  float f13;
  float f21;
  float f22;
  float f23;
  float f31;
  float f32;
  float f33;
} typedef Mat3x3;

#endif /* PRIMITIVES_H */
