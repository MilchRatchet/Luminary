#ifndef WAVEFRONT_H
#define WAVEFRONT_H

#include "mesh.h"
#include "texture.h"
#include <stdint.h>

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
  unsigned int v1;
  unsigned int v2;
  unsigned int v3;
  unsigned int vt1;
  unsigned int vt2;
  unsigned int vt3;
  unsigned int vn1;
  unsigned int vn2;
  unsigned int vn3;
  uint16_t object;
} typedef Wavefront_Triangle;

struct Wavefront_Material {
  size_t hash;
  uint16_t albedo_texture;
  uint16_t illuminance_texture;
  uint16_t material_texture;
} typedef Wavefront_Material;

struct Wavefront_Content {
  Wavefront_Vertex* vertices;
  unsigned int vertices_length;
  Wavefront_Normal* normals;
  unsigned int normals_length;
  Wavefront_UV* uvs;
  unsigned int uvs_length;
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
} typedef Wavefront_Content;

Wavefront_Content create_wavefront_content();
void free_wavefront_content(Wavefront_Content content);
int read_wavefront_file(const char* filename, Wavefront_Content* io_content);
texture_assignment* get_texture_assignments(Wavefront_Content content);
unsigned int convert_wavefront_content(Triangle** triangles, Wavefront_Content content);

#endif /* WAVEFRONT_H */
