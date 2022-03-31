#ifndef WAVEFRONT_H
#define WAVEFRONT_H

#include <stddef.h>
#include <stdint.h>

#include "mesh.h"
#include "texture.h"

struct Wavefront_Vertex {
  float x;
  float y;
  float z;
} typedef Wavefront_Vertex;

struct Wavefront_Normal {
  float x;
  float y;
  float z;
} typedef Wavefront_Normal;

struct Wavefront_UV {
  float u;
  float v;
} typedef Wavefront_UV;

struct Wavefront_Triangle {
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
} typedef Wavefront_Triangle;

struct Wavefront_Material {
  size_t hash;
  uint16_t albedo_texture;
  uint16_t illuminance_texture;
  uint16_t material_texture;
} typedef Wavefront_Material;

enum Wavefront_TextureInstanceType { WF_ALBEDO = 0, WF_ILLUMINANCE = 1, WF_MATERIAL = 2 } typedef Wavefront_TextureInstanceType;

struct Wavefront_TextureInstance {
  size_t hash;
  Wavefront_TextureInstanceType type;
  uint16_t offset;
} typedef Wavefront_TextureInstance;

struct Wavefront_TextureList {
  Wavefront_TextureInstance* textures;
  uint32_t count;
  uint32_t length;
} typedef Wavefront_TextureList;

struct Wavefront_Content {
  Wavefront_Vertex* vertices;
  int vertices_length;
  Wavefront_Normal* normals;
  int normals_length;
  Wavefront_UV* uvs;
  int uvs_length;
  Wavefront_Triangle* triangles;
  unsigned int triangles_length;
  Wavefront_Material* materials;
  unsigned int materials_length;
  TextureRGBA* albedo_maps;
  unsigned int albedo_maps_length;
  TextureRGBA* illuminance_maps;
  unsigned int illuminance_maps_length;
  TextureRGBA* material_maps;
  unsigned int material_maps_length;
  Wavefront_TextureList* texture_list;
} typedef Wavefront_Content;

Wavefront_Content create_wavefront_content();
void free_wavefront_content(Wavefront_Content content);
int read_wavefront_file(const char* filename, Wavefront_Content* io_content);
TextureAssignment* get_texture_assignments(Wavefront_Content content);
unsigned int convert_wavefront_content(Triangle** triangles, Wavefront_Content content);

#endif /* WAVEFRONT_H */
