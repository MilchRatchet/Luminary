#ifndef WAVEFRONT_H
#define WAVEFRONT_H

#include "mesh.h"
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

struct Wavefront_Mesh {
  Wavefront_Vertex* vertices;
  unsigned int vertices_length;
  Wavefront_Normal* normals;
  unsigned int normals_length;
  Wavefront_UV* uvs;
  unsigned int uvs_length;
  Wavefront_Triangle* triangles;
  unsigned int triangles_length;
} typedef Wavefront_Mesh;

int read_mesh_from_file(const char* name, Wavefront_Mesh** meshes, const int previous_length);
unsigned int convert_wavefront_mesh(Triangle** triangles, Wavefront_Mesh* meshes, const int length);

#endif /* WAVEFRONT_H */
