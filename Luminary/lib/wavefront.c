#include "wavefront.h"
#include <immintrin.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define LINE_SIZE 4096

static int read_mesh(FILE* file, Wavefront_Mesh* mesh, char* line) {
  fgets(line, LINE_SIZE, file);
  if (feof(file))
    return -1;

  mesh->vertices_length = 1024;
  mesh->vertices = (Wavefront_Vertex*) malloc(sizeof(Wavefront_Vertex) * mesh->vertices_length);
  unsigned int vertex_ptr = 0;

  mesh->uvs_length    = 1024;
  mesh->uvs           = (Wavefront_UV*) malloc(sizeof(Wavefront_UV) * mesh->uvs_length);
  unsigned int uv_ptr = 0;

  mesh->normals_length = 1024;
  mesh->normals = (Wavefront_Normal*) malloc(sizeof(Wavefront_Normal) * mesh->normals_length);
  unsigned int normal_ptr = 0;

  mesh->triangles_length = 1024;
  mesh->triangles =
    (Wavefront_Triangle*) malloc(sizeof(Wavefront_Triangle) * mesh->triangles_length);
  unsigned int triangle_ptr = 0;

  while (line[0] != 'o') {
    if (line[0] == 'v' && line[1] == ' ') {
      Wavefront_Vertex v;

      sscanf(line, "%*c %f %f %f\n", &v.x, &v.y, &v.z);

      mesh->vertices[vertex_ptr++] = v;

      if (vertex_ptr == mesh->vertices_length) {
        mesh->vertices_length *= 2;
        mesh->vertices = (Wavefront_Vertex*) realloc(
          mesh->vertices, sizeof(Wavefront_Vertex) * mesh->vertices_length);
      }
    }
    else if (line[0] == 'v' && line[1] == 't') {
      Wavefront_UV uv;

      sscanf(line, "%*2c %f %f", &uv.u, &uv.v);

      mesh->uvs[uv_ptr++] = uv;

      if (uv_ptr == mesh->uvs_length) {
        mesh->uvs_length *= 2;
        mesh->uvs = (Wavefront_UV*) realloc(mesh->uvs, sizeof(Wavefront_UV) * mesh->uvs_length);
      }
    }
    else if (line[0] == 'v' && line[1] == 'n') {
      Wavefront_Normal normal;

      sscanf(line, "%*2c %f %f %f", &normal.x, &normal.y, &normal.z);

      mesh->normals[normal_ptr++] = normal;

      if (normal_ptr == mesh->normals_length) {
        mesh->normals_length *= 2;
        mesh->normals = (Wavefront_Normal*) realloc(
          mesh->normals, sizeof(Wavefront_Normal) * mesh->normals_length);
      }
    }
    else if (line[0] == 'f') {
      Wavefront_Triangle face;

      sscanf(
        line, "%*c %u/%u/%u %u/%u/%u %u/%u/%u", &face.v1, &face.vt1, &face.vn1, &face.v2, &face.vt2,
        &face.vn2, &face.v3, &face.vt3, &face.vn3);

      mesh->triangles[triangle_ptr++] = face;

      if (triangle_ptr == mesh->triangles_length) {
        mesh->triangles_length *= 2;
        mesh->triangles = (Wavefront_Triangle*) realloc(
          mesh->triangles, sizeof(Wavefront_Triangle) * mesh->triangles_length);
      }
    }

    fgets(line, LINE_SIZE, file);

    if (feof(file))
      break;
  }

  mesh->vertices_length = vertex_ptr;
  mesh->vertices =
    (Wavefront_Vertex*) realloc(mesh->vertices, sizeof(Wavefront_Vertex) * mesh->vertices_length);

  mesh->uvs_length = uv_ptr;
  mesh->uvs        = (Wavefront_UV*) realloc(mesh->uvs, sizeof(Wavefront_UV) * mesh->uvs_length);

  mesh->normals_length = normal_ptr;
  mesh->normals =
    (Wavefront_Normal*) realloc(mesh->normals, sizeof(Wavefront_Normal) * mesh->normals_length);

  mesh->triangles_length = triangle_ptr;
  mesh->triangles        = (Wavefront_Triangle*) realloc(
    mesh->triangles, sizeof(Wavefront_Triangle) * mesh->triangles_length);

  return 0;
}

int read_mesh_from_file(const char* name, Wavefront_Mesh** meshes, const int previous_length) {
  FILE* file = fopen(name, "r");

  if (!file) {
    return -1;
  }

  int mesh_count = 8 + previous_length;

  if (previous_length) {
    *meshes = (Wavefront_Mesh*) realloc(*meshes, sizeof(Wavefront_Mesh) * mesh_count);
  }
  else {
    *meshes = (Wavefront_Mesh*) malloc(sizeof(Wavefront_Mesh) * mesh_count);
  }

  int mesh_ptr = previous_length;

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

unsigned int convert_wavefront_mesh(
  Triangle** triangles, Wavefront_Mesh* meshes, const int length) {
  unsigned int count = 0;
  for (int i = 0; i < length; i++) {
    count += meshes[i].triangles_length;
  }
  *triangles       = (Triangle*) malloc(sizeof(Triangle) * count);
  unsigned int ptr = 0;

  int vertex_offset = 0;
  int uv_offset     = 0;
  int normal_offset = 0;

  for (int i = 0; i < length; i++) {
    Wavefront_Mesh mesh = meshes[i];
    for (int j = 0; j < mesh.triangles_length; j++) {
      Wavefront_Triangle t = mesh.triangles[j];
      Triangle triangle;

      Wavefront_Vertex v = mesh.vertices[t.v1 - 1 - vertex_offset];

      triangle.v1.x = v.x;
      triangle.v1.y = v.y;
      triangle.v1.z = v.z;

      v = mesh.vertices[t.v2 - 1 - vertex_offset];

      triangle.v2.x = v.x;
      triangle.v2.y = v.y;
      triangle.v2.z = v.z;

      v = mesh.vertices[t.v3 - 1 - vertex_offset];

      triangle.v3.x = v.x;
      triangle.v3.y = v.y;
      triangle.v3.z = v.z;

      Wavefront_UV uv = mesh.uvs[t.vt1 - 1 - uv_offset];

      triangle.vt1.u = uv.u;
      triangle.vt1.v = uv.v;

      uv = mesh.uvs[t.vt2 - 1 - uv_offset];

      triangle.vt2.u = uv.u;
      triangle.vt2.v = uv.v;

      uv = mesh.uvs[t.vt3 - 1 - uv_offset];

      triangle.vt3.u = uv.u;
      triangle.vt3.v = uv.v;

      Wavefront_Normal n = mesh.normals[t.vn1 - 1 - normal_offset];

      float n_length = 1.0f / sqrt(n.x * n.x + n.y * n.y + n.z * n.z);

      triangle.vn1.x = n.x * n_length;
      triangle.vn1.y = n.y * n_length;
      triangle.vn1.z = n.z * n_length;

      n = mesh.normals[t.vn2 - 1 - normal_offset];

      n_length = 1.0f / sqrt(n.x * n.x + n.y * n.y + n.z * n.z);

      triangle.vn2.x = n.x * n_length;
      triangle.vn2.y = n.y * n_length;
      triangle.vn2.z = n.z * n_length;

      n = mesh.normals[t.vn3 - 1 - normal_offset];

      n_length = 1.0f / sqrt(n.x * n.x + n.y * n.y + n.z * n.z);

      triangle.vn3.x = n.x * n_length;
      triangle.vn3.y = n.y * n_length;
      triangle.vn3.z = n.z * n_length;

      (*triangles)[ptr++] = triangle;
    }
    vertex_offset += mesh.vertices_length;
    uv_offset += mesh.uvs_length;
    normal_offset += mesh.normals_length;
  }
  return count;
}
