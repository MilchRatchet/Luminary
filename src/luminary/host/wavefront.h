#ifndef WAVEFRONT_H
#define WAVEFRONT_H

#include <stddef.h>
#include <stdint.h>

#include "dictionary.h"
#include "mesh.h"
#include "texture.h"
#include "utils.h"

typedef vec3 WavefrontVertex;
typedef vec3 WavefrontNormal;

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
  uint16_t material;
  uint16_t object;
} typedef WavefrontTriangle;

enum WavefrontTextureType {
  WF_ALBEDO    = 0,
  WF_LUMINANCE = 1,
  WF_ROUGHNESS = 2,
  WF_METALLIC  = 3,
  WF_NORMAL    = 4,
  WF_TEX_TYPE_COUNT
} typedef WavefrontTextureType;

struct WavefrontMaterial {
  size_t hash;
  RGBF diffuse_reflectivity;
  float dissolve;
  RGBF specular_reflectivity;
  float specular_exponent; /* [0, 1000] */
  RGBF emission;
  float refraction_index;
  uint16_t texture[WF_TEX_TYPE_COUNT];
} typedef WavefrontMaterial;

struct WavefrontTextureInstance {
  size_t hash;
  uint16_t texture_id;
} typedef WavefrontTextureInstance;

struct WavefrontArguments {
  bool legacy_smoothness;
  bool force_transparency_cutout;
  float emission_scale;
  bool force_bidirectional_emission;
} typedef WavefrontArguments;

enum WavefrontContentState {
  WAVEFRONT_CONTENT_STATE_READY_TO_READ    = 0,
  WAVEFRONT_CONTENT_STATE_READY_TO_CONVERT = 1,
  WAVEFRONT_CONTENT_STATE_FINISHED         = 2
} typedef WavefrontContentState;

struct WavefrontContent {
  WavefrontArguments args;
  WavefrontContentState state;
  ARRAY WavefrontVertex* vertices;
  ARRAY WavefrontNormal* normals;
  ARRAY WavefrontUV* uvs;
  ARRAY WavefrontTriangle* triangles;
  ARRAY WavefrontMaterial* materials;
  ARRAY Texture** textures;
  ARRAY WavefrontTextureInstance* texture_instances;
  ARRAY char** object_names;
} typedef WavefrontContent;

LuminaryResult wavefront_create(WavefrontContent** content, WavefrontArguments args);
LuminaryResult wavefront_read_file(WavefrontContent* content, Path* file, Queue* queue);
LuminaryResult wavefront_convert_content(
  WavefrontContent* content, ARRAYPTR Mesh*** meshes, ARRAYPTR Texture*** textures, ARRAYPTR Material** materials, uint32_t material_offset,
  Dictionary* mesh_name_dict);
LuminaryResult wavefront_destroy(WavefrontContent** content);

LuminaryResult wavefront_arguments_get_default(WavefrontArguments* arguments);

#endif /* WAVEFRONT_H */
