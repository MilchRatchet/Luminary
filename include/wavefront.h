#ifndef WAVEFRONT_H
#define WAVEFRONT_H

#include <stddef.h>
#include <stdint.h>

#include "structs.h"

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
  int v1;
  int v2;
  int v3;
  int vt1;
  int vt2;
  int vt3;
  int vn1;
  int vn2;
  int vn3;
  uint16_t object;
} typedef WavefrontTriangle;

struct WavefrontMaterial {
  size_t hash;
  uint16_t albedo_texture;
  uint16_t illuminance_texture;
  uint16_t material_texture;
  uint16_t normal_texture;
} typedef WavefrontMaterial;

enum WavefrontTextureInstanceType {
  WF_ALBEDO      = 0,
  WF_ILLUMINANCE = 1,
  WF_MATERIAL    = 2,
  WF_NORMAL      = 3
} typedef WavefrontTextureInstanceType;

struct WavefrontTextureInstance {
  size_t hash;
  WavefrontTextureInstanceType type;
  uint16_t offset;
} typedef WavefrontTextureInstance;

struct WavefrontTextureList {
  WavefrontTextureInstance* textures;
  uint32_t count;
  uint32_t length;
} typedef WavefrontTextureList;

struct WavefrontContent {
  WavefrontVertex* vertices;
  int vertices_length;
  WavefrontNormal* normals;
  int normals_length;
  WavefrontUV* uvs;
  int uvs_length;
  WavefrontTriangle* triangles;
  unsigned int triangles_length;
  WavefrontMaterial* materials;
  unsigned int materials_length;
  TextureRGBA* albedo_maps;
  unsigned int albedo_maps_length;
  TextureRGBA* illuminance_maps;
  unsigned int illuminance_maps_length;
  TextureRGBA* material_maps;
  unsigned int material_maps_length;
  TextureRGBA* normal_maps;
  unsigned int normal_maps_length;
  WavefrontTextureList* texture_list;
} typedef WavefrontContent;

void wavefront_init(WavefrontContent** content);
void wavefront_clear(WavefrontContent** content);
int wavefront_read_file(WavefrontContent* _content, const char* filename);
TextureAssignment* wavefront_generate_texture_assignments(WavefrontContent* content);
unsigned int wavefront_convert_content(WavefrontContent* content, Triangle** triangles, TriangleGeomData* data);

#endif /* WAVEFRONT_H */
