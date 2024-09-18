#ifndef WAVEFRONT_H
#define WAVEFRONT_H

#include <stddef.h>
#include <stdint.h>

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

struct WavefrontContent {
  ARRAY WavefrontVertex* vertices;
  ARRAY WavefrontNormal* normals;
  ARRAY WavefrontUV* uvs;
  ARRAY WavefrontTriangle* triangles;
  ARRAY WavefrontMaterial* materials;
  ARRAY Texture** maps[4];
  ARRAY WavefrontTextureInstance* textures;
} typedef WavefrontContent;

LuminaryResult wavefront_create(WavefrontContent** content);
LuminaryResult wavefront_read_file(WavefrontContent* _content, const char* filename);
LuminaryResult wavefront_generate_texture_assignments(const WavefrontContent* content, PackedMaterial** material);
LuminaryResult wavefront_convert_content(WavefrontContent* content, Triangle** triangles, TriangleGeomData* data, uint32_t* num_triangles);
LuminaryResult wavefront_destroy(WavefrontContent** content);

#endif /* WAVEFRONT_H */
