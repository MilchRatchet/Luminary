#ifndef STRUCTS_H
#define STRUCTS_H

#include <assert.h>
#include <cuda_runtime_api.h>
#include <stdint.h>

/********************************************************
 * Vectors and matrices
 ********************************************************/

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

/********************************************************
 * Mesh
 ********************************************************/

struct Quad {
  vec3 vertex;
  vec3 edge1;
  vec3 edge2;
  vec3 normal;
} typedef Quad;

/********************************************************
 * Pixelformats
 ********************************************************/

struct RGB8 {
  uint8_t r;
  uint8_t g;
  uint8_t b;
} typedef RGB8;

struct RGBA8 {
  uint8_t r;
  uint8_t g;
  uint8_t b;
  uint8_t a;
} typedef RGBA8;

struct XRGB8 {
  uint8_t b;
  uint8_t g;
  uint8_t r;
  uint8_t ignore;
} typedef XRGB8;

struct RGB16 {
  uint16_t r;
  uint16_t g;
  uint16_t b;
} typedef RGB16;

struct RGBA16 {
  uint16_t r;
  uint16_t g;
  uint16_t b;
  uint16_t a;
} typedef RGBA16;

struct RGBF {
  float r;
  float g;
  float b;
} typedef RGBF;

#ifndef __cplusplus
struct RGBAhalf {
  uint16_t r;
  uint16_t g;
  uint16_t b;
  uint16_t a;
} typedef RGBAhalf;
#else
#include <cuda_fp16.h>

struct RGBAhalf {
  __half2 rg;
  __half2 ba;
} typedef RGBAhalf;
#endif

struct RGBAF {
  float r;
  float g;
  float b;
  float a;
} typedef RGBAF;

/********************************************************
 * Textures
 ********************************************************/

struct Material {
  float refraction_index;
  RGBAF albedo;
  RGBF emission;
  float metallic;
  float roughness;
  uint16_t albedo_tex;
  uint16_t luminance_tex;
  uint16_t material_tex;
  uint16_t normal_tex;
} typedef Material;

enum GBufferFlags {
  G_BUFFER_VOLUME_HIT           = 0b1,
  G_BUFFER_REFRACTION_IS_INSIDE = 0b10,
  G_BUFFER_COLORED_DIELECTRIC   = 0b100
} typedef GBufferFlags;

struct GBufferData {
  uint32_t hit_id;
  RGBAF albedo;
  RGBF emission;
  vec3 position;
  vec3 V;
  vec3 normal;
  float roughness;
  float metallic;
  uint32_t flags;
  /* IOR of medium in direction of V. */
  float ior_in;
  /* IOR of medium on the other side. */
  float ior_out;
} typedef GBufferData;

#endif /* STRUCTS_H */
