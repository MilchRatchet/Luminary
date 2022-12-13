#ifndef STRUCTS_H
#define STRUCTS_H

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

struct UV {
  float u;
  float v;
} typedef UV;

/********************************************************
 * Mesh
 ********************************************************/

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
  uint32_t object_maps;
  uint32_t light_id;
  float padding2;
  float padding3;
} typedef Triangle;

struct TraversalTriangle {
  vec3 vertex;
  vec3 edge1;
  vec3 edge2;
  uint32_t albedo_tex;
  float padding1;
  float padding2;
} typedef TraversalTriangle;

struct TriangleLight {
  vec3 vertex;
  vec3 edge1;
  vec3 edge2;
  uint32_t triangle_id;
  float padding1;
  float padding2;
} typedef TriangleLight;

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

struct TextureAssignment {
  uint16_t albedo_map;
  uint16_t illuminance_map;
  uint16_t material_map;
  uint16_t normal_map;
} typedef TextureAssignment;

struct TextureG {
  unsigned int width;
  unsigned int height;
  float* data;
} typedef TextureG;

enum TextureDataType { TexDataFP32 = 0, TexDataUINT8 = 1 } typedef TextureDataType;

struct TextureRGBA {
  unsigned int width;
  unsigned int height;
  unsigned int depth;
  unsigned int pitch;
  TextureDataType type;
  void* data;
  int gpu;
  int volume_tex;
} typedef TextureRGBA;

#endif /* STRUCTS_H */
