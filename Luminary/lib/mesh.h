#ifndef MESH_H
#define MESH_H

struct Vertex {
  float x;
  float y;
  float z;
} typedef Vertex;

struct Normal {
  float x;
  float y;
  float z;
} typedef Normal;

struct UV {
  float u;
  float v;
} typedef UV;

struct Triangle {
  unsigned int v1;
  unsigned int v2;
  unsigned int v3;
  unsigned int vt1;
  unsigned int vt2;
  unsigned int vt3;
  unsigned int vn1;
  unsigned int vn2;
  unsigned int vn3;
} typedef Triangle;

struct Mesh {
  Vertex* vertices;
  unsigned int vertices_length;
  Normal* normals;
  unsigned int normals_length;
  UV* uvs;
  unsigned int uvs_length;
  Triangle* triangles;
  unsigned int triangles_length;
} typedef Mesh;

#endif /* MESH_H */
