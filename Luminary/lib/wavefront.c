#include "wavefront.h"
#include <immintrin.h>
#include <stdlib.h>
#include <stdio.h>

#define LINE_SIZE 4096

static unsigned int read_vertices(FILE* file, Wavefront_Vertex** vertices, char* line) {
  unsigned int length = 1024;

  *vertices = (Wavefront_Vertex*) malloc(sizeof(Wavefront_Vertex) * length);

  unsigned int ptr = 0;

  while (line[0] == 'v' && line[1] == ' ') {
    Wavefront_Vertex v;

    sscanf(line, "%*c %f %f %f\n", &v.x, &v.y, &v.z);

    (*vertices)[ptr++] = v;

    if (ptr == length) {
      length *= 2;
      *vertices = (Wavefront_Vertex*) realloc(*vertices, sizeof(Wavefront_Vertex) * length);
    }

    fgets(line, LINE_SIZE, file);

    if (feof(file))
      break;
  }

  length = ptr;

  *vertices = (Wavefront_Vertex*) realloc(*vertices, sizeof(Wavefront_Vertex) * length);

  return length;
}

static unsigned int read_uv(FILE* file, Wavefront_UV** uvs, char* line) {
  unsigned int length = 1024;

  *uvs = (Wavefront_UV*) malloc(sizeof(Wavefront_UV) * length);

  unsigned int ptr = 0;

  while (line[0] == 'v' && line[1] == 't') {
    Wavefront_UV uv;

    sscanf(line, "%*2c %f %f", &uv.u, &uv.v);

    (*uvs)[ptr++] = uv;

    if (ptr == length) {
      length *= 2;
      *uvs = (Wavefront_UV*) realloc(*uvs, sizeof(Wavefront_UV) * length);
    }

    fgets(line, LINE_SIZE, file);

    if (feof(file))
      break;
  }

  length = ptr;

  *uvs = (Wavefront_UV*) realloc(*uvs, sizeof(Wavefront_UV) * length);

  return length;
}

static unsigned int read_normal(FILE* file, Wavefront_Normal** normals, char* line) {
  unsigned int length = 1024;

  *normals = (Wavefront_Normal*) malloc(sizeof(Wavefront_Normal) * length);

  unsigned int ptr = 0;

  while (line[0] == 'v' && line[1] == 'n') {
    Wavefront_Normal normal;

    sscanf(line, "%*2c %f %f %f", &normal.x, &normal.y, &normal.z);

    (*normals)[ptr++] = normal;

    if (ptr == length) {
      length *= 2;
      *normals = (Wavefront_Normal*) realloc(*normals, sizeof(Wavefront_Normal) * length);
    }

    fgets(line, LINE_SIZE, file);

    if (feof(file))
      break;
  }

  length = ptr;

  *normals = (Wavefront_Normal*) realloc(*normals, sizeof(Wavefront_Normal) * length);

  return length;
}

static unsigned int read_triangles(FILE* file, Wavefront_Triangle** triangles, char* line) {
  unsigned int length = 1024;

  *triangles = (Wavefront_Triangle*) malloc(sizeof(Wavefront_Triangle) * length);

  unsigned int ptr = 0;

  __m256i addend = _mm256_set1_epi32(1);

  while (line[0] == 'f' || line[0] == 's') {
    // ignore smoothing groups
    if (line[0] != 's') {
      Wavefront_Triangle face;

      sscanf(
        line, "%*c %lu/%lu/%lu %lu/%lu/%lu %lu/%lu/%lu", &face.v1, &face.vt1, &face.vn1, &face.v2,
        &face.vt2, &face.vn2, &face.v3, &face.vt3, &face.vn3);

      (*triangles)[ptr].vn3 = face.vn3 - 1;

      __m256i entries = _mm256_loadu_si256(&face);

      entries = _mm256_sub_epi32(entries, addend);

      _mm256_storeu_si256(&((*triangles)[ptr]), entries);

      ptr++;

      if (ptr == length) {
        length *= 2;
        *triangles = (Wavefront_Triangle*) realloc(*triangles, sizeof(Wavefront_Triangle) * length);
      }
    }
    fgets(line, LINE_SIZE, file);

    if (feof(file))
      break;
  }

  length = ptr;

  *triangles = (Wavefront_Triangle*) realloc(*triangles, sizeof(Wavefront_Triangle) * length);

  return length;
}

static int read_mesh(FILE* file, Wavefront_Mesh* mesh, char* line) {
  while (line[0] != 'v') {
    fgets(line, LINE_SIZE, file);
    if (feof(file) || line[0] == 'o') {
      return 1;
    }
  }

  Wavefront_Vertex* vertices;
  mesh->vertices_length = read_vertices(file, &vertices, line);
  mesh->vertices        = vertices;

  Wavefront_UV* uvs;
  mesh->uvs_length = read_uv(file, &uvs, line);
  mesh->uvs        = uvs;

  Wavefront_Normal* normals;
  mesh->normals_length = read_normal(file, &normals, line);
  mesh->normals        = normals;

  while (line[0] != 'f') {
    fgets(line, LINE_SIZE, file);
    if (feof(file) || line[0] == 'o') {
      return 1;
    }
  }

  Wavefront_Triangle* triangles;
  mesh->triangles_length = read_triangles(file, &triangles, line);
  mesh->triangles        = triangles;

  return 0;
}

int read_mesh_from_file(char* name, Wavefront_Mesh** meshes) {
  FILE* file = fopen(name, "r");

  if (!file) {
    return -1;
  }

  int mesh_count = 8;

  *meshes = (Wavefront_Mesh*) malloc(sizeof(Wavefront_Mesh) * mesh_count);

  int mesh_ptr = 0;

  char* line = (char*) malloc(LINE_SIZE);

  line[LINE_SIZE - 1] = '\0';

  fgets(line, LINE_SIZE, file);

  while (!feof(file)) {
    if (line[0] == 'o') {
      if (read_mesh(file, (*meshes) + mesh_ptr, line)) {
        free(*meshes);
        return -1;
      }
      mesh_ptr++;
      if (mesh_ptr == mesh_count) {
        mesh_count *= 2;
        *meshes = (Wavefront_Mesh*) realloc(*meshes, sizeof(Wavefront_Mesh) * mesh_count);
      }
    }
    else {
      fgets(line, LINE_SIZE, file);
    }
  }

  mesh_count = mesh_ptr;

  *meshes = (Wavefront_Mesh*) realloc(*meshes, sizeof(Wavefront_Mesh) * mesh_count);

  free(line);

  return mesh_count;
}

unsigned int convert_wavefront_mesh(Triangle** triangles, Wavefront_Mesh* meshes, int length) {
  unsigned int count = 0;
  for (int i = 0; i < length; i++) {
    count += meshes[i].triangles_length;
  }
  *triangles       = (Triangle*) malloc(sizeof(Triangle) * count);
  unsigned int ptr = 0;

  for (int i = 0; i < length; i++) {
    Wavefront_Mesh mesh = meshes[i];
    for (int j = 0; j < mesh.triangles_length; j++) {
      Wavefront_Triangle t = mesh.triangles[j];
      Triangle triangle;

      Wavefront_Vertex v = mesh.vertices[t.v1 % mesh.vertices_length];

      triangle.v1.x = v.x;
      triangle.v1.y = v.y;
      triangle.v1.z = v.z;

      v = mesh.vertices[t.v2 % mesh.vertices_length];

      triangle.v2.x = v.x;
      triangle.v2.y = v.y;
      triangle.v2.z = v.z;

      v = mesh.vertices[t.v3 % mesh.vertices_length];

      triangle.v3.x = v.x;
      triangle.v3.y = v.y;
      triangle.v3.z = v.z;

      Wavefront_UV uv = mesh.uvs[t.vt1 % mesh.uvs_length];

      triangle.vt1.u = uv.u;
      triangle.vt1.v = uv.v;

      uv = mesh.uvs[t.vt2 % mesh.uvs_length];

      triangle.vt2.u = uv.u;
      triangle.vt2.v = uv.v;

      uv = mesh.uvs[t.vt3 % mesh.uvs_length];

      triangle.vt3.u = uv.u;
      triangle.vt3.v = uv.v;

      Wavefront_Normal n = mesh.normals[t.vn1 % mesh.normals_length];

      triangle.vn1.x = n.x;
      triangle.vn1.y = n.y;
      triangle.vn1.z = n.z;

      n = mesh.normals[t.vn2 % mesh.normals_length];

      triangle.vn2.x = n.x;
      triangle.vn2.y = n.y;
      triangle.vn2.z = n.z;

      n = mesh.normals[t.vn3 % mesh.normals_length];

      triangle.vn3.x = n.x;
      triangle.vn3.y = n.y;
      triangle.vn3.z = n.z;

      (*triangles)[ptr++] = triangle;
    }
  }
  return count;
}
