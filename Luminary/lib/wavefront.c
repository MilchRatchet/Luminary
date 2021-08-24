#include "wavefront.h"

#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "bench.h"
#include "error.h"
#include "png.h"
#include "texture.h"
#include "utils.h"

#define LINE_SIZE 4096

Wavefront_Content create_wavefront_content() {
  Wavefront_Content content;
  content.vertices         = malloc(sizeof(Wavefront_Vertex) * 1);
  content.vertices_length  = 0;
  content.normals          = malloc(sizeof(Wavefront_Normal) * 1);
  content.normals_length   = 0;
  content.uvs              = malloc(sizeof(Wavefront_UV) * 1);
  content.uvs_length       = 0;
  content.triangles        = malloc(sizeof(Wavefront_Triangle) * 1);
  content.triangles_length = 0;
  content.materials        = malloc(sizeof(Wavefront_Material) * 1);
  content.materials_length = 1;

  content.materials[0].hash                = 0;
  content.materials[0].albedo_texture      = 0;
  content.materials[0].illuminance_texture = 0;
  content.materials[0].material_texture    = 0;

  content.albedo_maps             = (TextureRGBA*) malloc(sizeof(TextureRGBA) * 1);
  content.albedo_maps_length      = 1;
  content.illuminance_maps        = (TextureRGBA*) malloc(sizeof(TextureRGBA) * 1);
  content.illuminance_maps_length = 1;
  content.material_maps           = (TextureRGBA*) malloc(sizeof(TextureRGBA) * 1);
  content.material_maps_length    = 1;

  content.albedo_maps[0].width          = 1;
  content.albedo_maps[0].height         = 1;
  content.albedo_maps[0].data           = (RGBAF*) malloc(sizeof(RGBAF));
  content.albedo_maps[0].data[0].r      = 0.9f;
  content.albedo_maps[0].data[0].g      = 0.9f;
  content.albedo_maps[0].data[0].b      = 0.9f;
  content.albedo_maps[0].data[0].a      = 1.0f;
  content.illuminance_maps[0].width     = 1;
  content.illuminance_maps[0].height    = 1;
  content.illuminance_maps[0].data      = (RGBAF*) malloc(sizeof(RGBAF));
  content.illuminance_maps[0].data[0].r = 0.0f;
  content.illuminance_maps[0].data[0].g = 0.0f;
  content.illuminance_maps[0].data[0].b = 0.0f;
  content.illuminance_maps[0].data[0].a = 0.0f;
  content.material_maps[0].width        = 1;
  content.material_maps[0].height       = 1;
  content.material_maps[0].data         = (RGBAF*) malloc(sizeof(RGBAF));
  content.material_maps[0].data[0].r    = 0.2f;
  content.material_maps[0].data[0].g    = 0.0f;
  content.material_maps[0].data[0].b    = 1.0f / 255.0f;
  content.material_maps[0].data[0].a    = 0.0f;

  return content;
}

void free_wavefront_content(Wavefront_Content content) {
  free(content.vertices);
  free(content.normals);
  free(content.uvs);
  free(content.triangles);
  free(content.materials);

  for (int i = 0; i < content.albedo_maps_length; i++) {
    free(content.albedo_maps[i].data);
  }
  for (int i = 0; i < content.illuminance_maps_length; i++) {
    free(content.illuminance_maps[i].data);
  }
  for (int i = 0; i < content.material_maps_length; i++) {
    free(content.material_maps[i].data);
  }

  free(content.albedo_maps);
  free(content.illuminance_maps);
  free(content.material_maps);
}

static size_t hash_djb2(unsigned char* str) {
  size_t hash = 5381;
  int c;

  while ((c = *(str++))) {
    hash = ((hash << 5) + hash) + c;
  }

  return hash;
}

static void read_materials_file(const char* filename, Wavefront_Content* io_content) {
  FILE* file;
  fopen_s(&file, filename, "r");

  if (!file) {
    print_error("Could not read material file!");
    return;
  }

  Wavefront_Content content = *io_content;

  unsigned int materials_count        = content.materials_length;
  unsigned int albedo_maps_count      = content.albedo_maps_length;
  unsigned int illuminance_maps_count = content.illuminance_maps_length;
  unsigned int material_maps_count    = content.material_maps_length;

  char* line = malloc(LINE_SIZE);
  char* path = malloc(LINE_SIZE);

  while (!feof(file)) {
    fgets(line, LINE_SIZE, file);

    if (line[0] == 'n' && line[1] == 'e' && line[2] == 'w' && line[3] == 'm' && line[4] == 't' && line[5] == 'l') {
      ensure_capacity(content.materials, materials_count, content.materials_length, sizeof(Wavefront_Material));
      sscanf_s(line, "%*s %s\n", path, LINE_SIZE);
      size_t hash = hash_djb2((unsigned char*) path);

      content.materials[materials_count].hash                = hash;
      content.materials[materials_count].albedo_texture      = 0;
      content.materials[materials_count].illuminance_texture = 0;
      content.materials[materials_count].material_texture    = 0;
      materials_count++;
    }
    else if (line[0] == 'm' && line[1] == 'a' && line[2] == 'p' && line[3] == '_') {
      if (line[4] == 'K' && line[5] == 'd') {
        ensure_capacity(content.albedo_maps, albedo_maps_count, content.albedo_maps_length, sizeof(TextureRGBA));
        sscanf_s(line, "%*s %s\n", path, LINE_SIZE);
        content.albedo_maps[albedo_maps_count]                = load_texture_from_png(path);
        content.materials[materials_count - 1].albedo_texture = albedo_maps_count;
        albedo_maps_count++;
      }
      else if (line[4] == 'K' && line[5] == 'e') {
        ensure_capacity(content.illuminance_maps, illuminance_maps_count, content.illuminance_maps_length, sizeof(TextureRGBA));
        sscanf_s(line, "%*s %s\n", path, LINE_SIZE);
        content.illuminance_maps[illuminance_maps_count]           = load_texture_from_png(path);
        content.materials[materials_count - 1].illuminance_texture = illuminance_maps_count;
        illuminance_maps_count++;
      }
      else if (line[4] == 'N' && line[5] == 's') {
        ensure_capacity(content.material_maps, material_maps_count, content.material_maps_length, sizeof(TextureRGBA));
        sscanf_s(line, "%*s %s\n", path, LINE_SIZE);
        content.material_maps[material_maps_count]              = load_texture_from_png(path);
        content.materials[materials_count - 1].material_texture = material_maps_count;
        material_maps_count++;
      }
    }
  }

  content.materials_length        = materials_count;
  content.materials               = safe_realloc(content.materials, sizeof(Wavefront_Material) * content.materials_length);
  content.albedo_maps_length      = albedo_maps_count;
  content.albedo_maps             = safe_realloc(content.albedo_maps, sizeof(TextureRGBA) * content.albedo_maps_length);
  content.illuminance_maps_length = illuminance_maps_count;
  content.illuminance_maps        = safe_realloc(content.illuminance_maps, sizeof(TextureRGBA) * content.illuminance_maps_length);
  content.material_maps_length    = material_maps_count;
  content.material_maps           = safe_realloc(content.material_maps, sizeof(TextureRGBA) * content.material_maps_length);

  *io_content = content;

  fclose(file);
}

int read_wavefront_file(const char* filename, Wavefront_Content* io_content) {
  bench_tic();
  FILE* file;
  fopen_s(&file, filename, "r");

  if (!file) {
    print_error("File could not be opened!");
    return -1;
  }

  Wavefront_Content content = *io_content;

  const unsigned int vertices_offset = content.vertices_length;
  const unsigned int normals_offset  = content.normals_length;
  const unsigned int uvs_offset      = content.uvs_length;

  unsigned int triangles_count = content.triangles_length;
  unsigned int vertices_count  = content.vertices_length;
  unsigned int normals_count   = content.normals_length;
  unsigned int uvs_count       = content.uvs_length;
  unsigned int materials_count = content.materials_length;

  size_t* loaded_mtls             = malloc(sizeof(size_t) * 16);
  unsigned int loaded_mtls_count  = 0;
  unsigned int loaded_mtls_length = 16;

  uint16_t current_material = 0;

  char* line = malloc(LINE_SIZE);
  char* path = malloc(LINE_SIZE);

  while (!feof(file)) {
    fgets(line, LINE_SIZE, file);

    if (line[0] == 'v' && line[1] == ' ') {
      ensure_capacity(content.vertices, vertices_count, content.vertices_length, sizeof(Wavefront_Vertex));
      Wavefront_Vertex v;
      sscanf_s(line, "%*c %f %f %f\n", &v.x, &v.y, &v.z);
      content.vertices[vertices_count] = v;
      vertices_count++;
    }
    else if (line[0] == 'v' && line[1] == 'n') {
      ensure_capacity(content.normals, normals_count, content.normals_length, sizeof(Wavefront_Normal));
      Wavefront_Normal n;
      sscanf_s(line, "%*2c %f %f %f\n", &n.x, &n.y, &n.z);
      content.normals[normals_count] = n;
      normals_count++;
    }
    else if (line[0] == 'v' && line[1] == 't') {
      ensure_capacity(content.uvs, uvs_count, content.uvs_length, sizeof(Wavefront_UV));
      Wavefront_UV uv;
      sscanf_s(line, "%*2c %f %f\n", &uv.u, &uv.v);
      content.uvs[uvs_count] = uv;
      uvs_count++;
    }
    else if (line[0] == 'f') {
      ensure_capacity(content.triangles, triangles_count, content.triangles_length, sizeof(Wavefront_Triangle));
      Wavefront_Triangle face;
      sscanf_s(
        line, "%*c %u/%u/%u %u/%u/%u %u/%u/%u", &face.v1, &face.vt1, &face.vn1, &face.v2, &face.vt2, &face.vn2, &face.v3, &face.vt3,
        &face.vn3);
      face.object = current_material;

      face.v1 += vertices_offset;
      face.v2 += vertices_offset;
      face.v3 += vertices_offset;
      face.vn1 += normals_offset;
      face.vn2 += normals_offset;
      face.vn3 += normals_offset;
      face.vt1 += uvs_offset;
      face.vt2 += uvs_offset;
      face.vt3 += uvs_offset;

      content.triangles[triangles_count] = face;
      triangles_count++;
    }
    else if (line[0] == 'm' && line[1] == 't' && line[2] == 'l' && line[3] == 'l' && line[4] == 'i' && line[5] == 'b') {
      sscanf_s(line, "%*s %s\n", path, LINE_SIZE);
      size_t hash = hash_djb2((unsigned char*) path);

      int already_loaded = 0;

      for (unsigned int i = 0; i < loaded_mtls_count; i++) {
        if (loaded_mtls[i] == hash)
          already_loaded = 1;
      }

      if (!already_loaded) {
        ensure_capacity(loaded_mtls, loaded_mtls_count, loaded_mtls_length, sizeof(size_t));
        loaded_mtls[loaded_mtls_count++] = hash;
        read_materials_file(path, &content);
        materials_count = content.materials_length;
      }
    }
    else if (line[0] == 'u' && line[1] == 's' && line[2] == 'e' && line[3] == 'm' && line[4] == 't' && line[5] == 'l') {
      sscanf_s(line, "%*s %s\n", path, LINE_SIZE);
      size_t hash      = hash_djb2((unsigned char*) path);
      current_material = 0;
      for (int i = 1; i < materials_count; i++) {
        if (content.materials[i].hash == hash) {
          current_material = i;
          break;
        }
      }
    }
  }

  content.vertices_length  = vertices_count;
  content.vertices         = safe_realloc(content.vertices, sizeof(Wavefront_Vertex) * content.vertices_length);
  content.normals_length   = normals_count;
  content.normals          = safe_realloc(content.normals, sizeof(Wavefront_Normal) * content.normals_length);
  content.uvs_length       = uvs_count;
  content.uvs              = safe_realloc(content.uvs, sizeof(Wavefront_UV) * content.uvs_length);
  content.triangles_length = triangles_count;
  content.triangles        = safe_realloc(content.triangles, sizeof(Wavefront_Triangle) * content.triangles_length);

  *io_content = content;

  free(loaded_mtls);

  bench_toc("Reading *.obj File");

  return 0;
}

texture_assignment* get_texture_assignments(Wavefront_Content content) {
  texture_assignment* texture_assignments = malloc(sizeof(texture_assignment) * content.materials_length);

  for (int i = 0; i < content.materials_length; i++) {
    texture_assignment assignment;
    assignment.albedo_map      = content.materials[i].albedo_texture;
    assignment.illuminance_map = content.materials[i].illuminance_texture;
    assignment.material_map    = content.materials[i].material_texture;

    texture_assignments[i] = assignment;
  }

  return texture_assignments;
}

unsigned int convert_wavefront_content(Triangle** triangles, Wavefront_Content content) {
  bench_tic();

  unsigned int count = content.triangles_length;

  *triangles       = (Triangle*) malloc(sizeof(Triangle) * count);
  unsigned int ptr = 0;

  for (int j = 0; j < count; j++) {
    Wavefront_Triangle t = content.triangles[j];
    Triangle triangle;

    Wavefront_Vertex v;

    if (t.v1 > content.vertices_length) {
      continue;
    }
    else {
      v = content.vertices[t.v1 - 1];
    }

    triangle.vertex.x = v.x;
    triangle.vertex.y = v.y;
    triangle.vertex.z = v.z;

    if (t.v2 > content.vertices_length) {
      continue;
    }
    else {
      v = content.vertices[t.v2 - 1];
    }

    triangle.edge1.x = v.x - triangle.vertex.x;
    triangle.edge1.y = v.y - triangle.vertex.y;
    triangle.edge1.z = v.z - triangle.vertex.z;

    if (t.v3 > content.vertices_length) {
      continue;
    }
    else {
      v = content.vertices[t.v3 - 1];
    }

    triangle.edge2.x = v.x - triangle.vertex.x;
    triangle.edge2.y = v.y - triangle.vertex.y;
    triangle.edge2.z = v.z - triangle.vertex.z;

    Wavefront_UV uv;

    if (t.vt1 > content.uvs_length) {
      uv.u = 0.0f;
      uv.v = 0.0f;
    }
    else {
      uv = content.uvs[t.vt1 - 1];
    }

    triangle.vertex_texture.u = uv.u;
    triangle.vertex_texture.v = uv.v;

    if (t.vt2 > content.uvs_length) {
      uv.u = 0.0f;
      uv.v = 0.0f;
    }
    else {
      uv = content.uvs[t.vt2 - 1];
    }

    triangle.edge1_texture.u = uv.u - triangle.vertex_texture.u;
    triangle.edge1_texture.v = uv.v - triangle.vertex_texture.v;

    if (t.vt3 > content.uvs_length) {
      uv.u = 0.0f;
      uv.v = 0.0f;
    }
    else {
      uv = content.uvs[t.vt3 - 1];
    }

    triangle.edge2_texture.u = uv.u - triangle.vertex_texture.u;
    triangle.edge2_texture.v = uv.v - triangle.vertex_texture.v;

    Wavefront_Normal n;

    if (t.vn1 > content.normals_length) {
      n.x = 0.0f;
      n.y = 0.0f;
      n.z = 0.0f;
    }
    else {
      n = content.normals[t.vn1 - 1];

      const float n_length = 1.0f / sqrtf(n.x * n.x + n.y * n.y + n.z * n.z);

      if (isnan(n_length) || isinf(n_length)) {
        n.x = 0.0f;
        n.y = 0.0f;
        n.z = 0.0f;
      }
      else {
        n.x *= n_length;
        n.y *= n_length;
        n.z *= n_length;
      }
    }

    triangle.vertex_normal.x = n.x;
    triangle.vertex_normal.y = n.y;
    triangle.vertex_normal.z = n.z;

    if (t.vn2 > content.normals_length) {
      n.x = 0.0f;
      n.y = 0.0f;
      n.z = 0.0f;
    }
    else {
      n = content.normals[t.vn2 - 1];

      const float n_length = 1.0f / sqrtf(n.x * n.x + n.y * n.y + n.z * n.z);

      if (isnan(n_length) || isinf(n_length)) {
        n.x = 0.0f;
        n.y = 0.0f;
        n.z = 0.0f;
      }
      else {
        n.x *= n_length;
        n.y *= n_length;
        n.z *= n_length;
      }
    }

    triangle.edge1_normal.x = n.x - triangle.vertex_normal.x;
    triangle.edge1_normal.y = n.y - triangle.vertex_normal.y;
    triangle.edge1_normal.z = n.z - triangle.vertex_normal.z;

    if (t.vn3 > content.normals_length) {
      n.x = 0.0f;
      n.y = 0.0f;
      n.z = 0.0f;
    }
    else {
      n = content.normals[t.vn3 - 1];

      const float n_length = 1.0f / sqrtf(n.x * n.x + n.y * n.y + n.z * n.z);

      if (isnan(n_length) || isinf(n_length)) {
        n.x = 0.0f;
        n.y = 0.0f;
        n.z = 0.0f;
      }
      else {
        n.x *= n_length;
        n.y *= n_length;
        n.z *= n_length;
      }
    }

    triangle.edge2_normal.x = n.x - triangle.vertex_normal.x;
    triangle.edge2_normal.y = n.y - triangle.vertex_normal.y;
    triangle.edge2_normal.z = n.z - triangle.vertex_normal.z;

    triangle.object_maps = t.object;

    (*triangles)[ptr++] = triangle;
  }

  bench_toc("Converting Mesh");

  return ptr;
}
