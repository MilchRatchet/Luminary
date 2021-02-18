#include "mesh.h"
#include <immintrin.h>
#include <stdlib.h>
#include <stdio.h>

#define LINE_SIZE 4096

static unsigned int read_vertices(FILE* file, Vertex** vertices, char* line) {
  unsigned int length = 1024;

  *vertices = (Vertex*) malloc(sizeof(Vertex) * length);

  unsigned int ptr = 0;

  while (line[0] == 'v' && line[1] == ' ') {
    Vertex v;

    sscanf(line, "%*c %f %f %f", v.x, v.y, v.z);

    (*vertices)[ptr++] = v;

    if (ptr == length) {
      length *= 2;
      *vertices = (Vertex*) realloc(*vertices, sizeof(Vertex) * length);
    }

    fgets(line, LINE_SIZE, file);

    if (feof(file))
      return 1;
  }

  length = ptr;

  *vertices = (Vertex*) realloc(*vertices, sizeof(Vertex) * length);

  return length;
}

static int read_uv(FILE* file, UV** uvs, char* line, unsigned int length) {
  *uvs = (UV*) malloc(sizeof(UV) * length);

  unsigned int ptr = 0;

  while (line[0] == 'v' && line[1] == 't') {
    if (ptr == length)
      return 1;

    UV uv;

    sscanf(line, "%*c %f %f", uv.u, uv.v);

    (*uvs)[ptr++] = uv;

    fgets(line, LINE_SIZE, file);

    if (feof(file))
      return 1;
  }

  return 0;
}

static int read_normal(FILE* file, Normal** normals, char* line, unsigned int length) {
  *normals = (Normal*) malloc(sizeof(Normal) * length);

  unsigned int ptr = 0;

  while (line[0] == 'v' && line[1] == 'n') {
    if (ptr == length)
      return 1;

    Normal normal;

    sscanf(line, "%*c %f %f %f", normal.x, normal.y, normal.z);

    (*normals)[ptr++] = normal;

    fgets(line, LINE_SIZE, file);

    if (feof(file))
      return 1;
  }

  return 0;
}

static unsigned int read_triangles(FILE* file, Triangle** triangles, char* line) {
  unsigned int length = 1024;

  *triangles = (Triangle*) malloc(sizeof(Triangle) * length);

  unsigned int ptr = 0;

  __m256i addend = _mm256_set1_epi32(1);

  while (line[0] == 'v' && line[1] == 'n') {
    if (ptr == length)
      return 1;

    Triangle face;

    sscanf(
      line, "%*c %lu/%lu/%lu %lu/%lu/%lu %lu/%lu/%lu", face.v1, face.vt1, face.vn1, face.v2,
      face.vt2, face.vn2, face.v3, face.vt3, face.vn3);

    (*triangles)[ptr].vn3 = face.vn3 - 1;

    __m256i entries = _mm256_loadu_si256(&face);

    entries = _mm256_sub_epi32(entries, addend);

    _mm256_storeu_si256(&((*triangles)[ptr]), entries);

    ptr++;

    if (ptr == length) {
      length *= 2;
      *triangles = (Triangle*) realloc(*triangles, sizeof(Triangle) * length);
    }

    fgets(line, LINE_SIZE, file);

    if (feof(file))
      return -1;
  }

  length = ptr;

  *triangles = (Triangle*) realloc(*triangles, sizeof(Triangle) * length);

  return length;
}

static int read_mesh(FILE* file, Mesh* mesh, char* line) {
  while (line[0] != 'v') {
    fgets(line, LINE_SIZE, file);
    if (feof(file) || line[0] == 'o') {
      return 1;
    }
  }

  Vertex* vertices;
  mesh->vertices_length = read_vertices(file, &vertices, line);
  mesh->vertices        = vertices;

  UV* uvs;
  if (read_uv(file, &uvs, line, mesh->vertices_length))
    return 1;

  mesh->uvs_length = mesh->vertices_length;
  mesh->uvs        = uvs;

  Normal* normals;
  if (read_uv(file, &normals, line, mesh->vertices_length))
    return 1;

  mesh->normals_length = mesh->vertices_length;
  mesh->normals        = normals;

  while (line[0] != 'f') {
    fgets(line, LINE_SIZE, file);
    if (feof(file) || line[0] == 'o') {
      return 1;
    }
  }

  Triangle* triangles;
  mesh->triangles_length = read_triangles(file, &vertices, line);
  mesh->triangles        = triangles;

  return 0;
}

int read_mesh_from_file(char* name, Mesh** meshes) {
  FILE* file = fopen(name, 'r');

  if (!file) {
    return 1;
  }

  int mesh_count = 8;

  *meshes = (Mesh*) malloc(sizeof(Mesh) * mesh_count);

  int mesh_ptr = 0;

  char* line = (char*) malloc(LINE_SIZE);

  fgets(line, LINE_SIZE, file);
  while (!feof(file)) {
    if (line[0] == 'o') {
      if (!read_mesh(file, (*meshes) + mesh_ptr, line)) {
        free(*meshes);
        return 1;
      }
      mesh_ptr++;
      if (mesh_ptr == mesh_count) {
        mesh_count *= 2;
        *meshes = (Triangle*) realloc(*meshes, sizeof(Triangle) * mesh_count);
      }
    }
    else {
      fgets(line, LINE_SIZE, file);
    }
  }

  mesh_count = mesh_ptr;

  *meshes = (Triangle*) realloc(*meshes, sizeof(Triangle) * mesh_count);

  return 0;
}
