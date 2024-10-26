#ifndef WAVEFRONT_H
#define WAVEFRONT_H

#include <stddef.h>
#include <stdint.h>

#include "mesh.h"
#include "texture.h"
#include "utils.h"

struct WavefrontVertex {
  float x;
  float y;
  float z;
} typedef WavefrontVertex;

struct WavefrontNormal {
  float x;
  float y;
  float z;
} typedef WavefrontNormal;

struct WavefrontUV {
  float u;
  float v;
} typedef WavefrontUV;

struct WavefrontTriangle {
  int32_t v1;
  int32_t v2;
  int32_t v3;
  int32_t vt1;
  int32_t vt2;
  int32_t vt3;
  int32_t vn1;
  int32_t vn2;
  int32_t vn3;
  uint16_t object;
} typedef WavefrontTriangle;

struct WavefrontMaterial {
  size_t hash;
  RGBF diffuse_reflectivity;
  float dissolve;
  RGBF specular_reflectivity;
  float specular_exponent; /* [0, 1000] */
  RGBF emission;
  float refraction_index;
  uint16_t texture[4];
} typedef WavefrontMaterial;

enum WavefrontTextureInstanceType { WF_ALBEDO = 0, WF_LUMINANCE = 1, WF_MATERIAL = 2, WF_NORMAL = 3 } typedef WavefrontTextureInstanceType;

struct WavefrontTextureInstance {
  size_t hash;
  WavefrontTextureInstanceType type;
  uint16_t offset;
} typedef WavefrontTextureInstance;

enum WavefrontContentState {
  WAVEFRONT_CONTENT_STATE_READY_TO_READ    = 0,
  WAVEFRONT_CONTENT_STATE_READY_TO_CONVERT = 1,
  WAVEFRONT_CONTENT_STATE_FINISHED         = 2
} typedef WavefrontContentState;

struct WavefrontContent {
  WavefrontContentState state;
  ARRAY WavefrontVertex* vertices;
  ARRAY WavefrontNormal* normals;
  ARRAY WavefrontUV* uvs;
  ARRAY WavefrontTriangle* triangles;
  ARRAY WavefrontMaterial* materials;
  ARRAY Texture** maps[4];
  ARRAY WavefrontTextureInstance* textures;
} typedef WavefrontContent;

LuminaryResult wavefront_create(WavefrontContent** content);
LuminaryResult wavefront_read_file(WavefrontContent* content, Path* file);
LuminaryResult wavefront_convert_content(
  WavefrontContent* content, ARRAYPTR Mesh*** meshes, ARRAYPTR Texture*** textures, ARRAYPTR Material** materials,
  uint32_t material_offset);
LuminaryResult wavefront_destroy(WavefrontContent** content);

#endif /* WAVEFRONT_H */
