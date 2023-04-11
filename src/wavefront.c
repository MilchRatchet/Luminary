#include "wavefront.h"

#include <assert.h>
#include <float.h>
#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bench.h"
#include "log.h"
#include "png.h"
#include "structs.h"
#include "utils.h"

#define LINE_SIZE 4096
#define READ_BUFFER_SIZE 262144  // 256kb

void wavefront_init(WavefrontContent** content) {
  *content = malloc(sizeof(WavefrontContent));

  (*content)->vertices         = malloc(sizeof(WavefrontVertex) * 1);
  (*content)->vertices_length  = 0;
  (*content)->normals          = malloc(sizeof(WavefrontNormal) * 1);
  (*content)->normals_length   = 0;
  (*content)->uvs              = malloc(sizeof(WavefrontUV) * 1);
  (*content)->uvs_length       = 0;
  (*content)->triangles        = malloc(sizeof(WavefrontTriangle) * 1);
  (*content)->triangles_length = 0;
  (*content)->materials        = malloc(sizeof(WavefrontMaterial) * 1);
  (*content)->materials_length = 1;

  (*content)->materials[0].hash                = 0;
  (*content)->materials[0].albedo_texture      = 0;
  (*content)->materials[0].illuminance_texture = 0;
  (*content)->materials[0].material_texture    = 0;
  (*content)->materials[0].normal_texture      = 0;

  (*content)->albedo_maps             = (TextureRGBA*) malloc(sizeof(TextureRGBA) * 1);
  (*content)->albedo_maps_length      = 0;
  (*content)->illuminance_maps        = (TextureRGBA*) malloc(sizeof(TextureRGBA) * 1);
  (*content)->illuminance_maps_length = 0;
  (*content)->material_maps           = (TextureRGBA*) malloc(sizeof(TextureRGBA) * 1);
  (*content)->material_maps_length    = 0;
  (*content)->normal_maps             = (TextureRGBA*) malloc(sizeof(TextureRGBA) * 1);
  (*content)->normal_maps_length      = 0;

  (*content)->texture_list           = malloc(sizeof(WavefrontTextureList));
  (*content)->texture_list->textures = malloc(sizeof(WavefrontTextureInstance) * 16);
  (*content)->texture_list->count    = 0;
  (*content)->texture_list->length   = 16;
}

void wavefront_clear(WavefrontContent** content) {
  if (!content) {
    error_message("WavefrontContent is NULL.");
    return;
  }

  free((*content)->vertices);
  free((*content)->normals);
  free((*content)->uvs);
  free((*content)->triangles);
  free((*content)->materials);

  for (unsigned int i = 0; i < (*content)->albedo_maps_length; i++) {
    free((*content)->albedo_maps[i].data);
  }
  for (unsigned int i = 0; i < (*content)->illuminance_maps_length; i++) {
    free((*content)->illuminance_maps[i].data);
  }
  for (unsigned int i = 0; i < (*content)->material_maps_length; i++) {
    free((*content)->material_maps[i].data);
  }
  for (unsigned int i = 0; i < (*content)->normal_maps_length; i++) {
    free((*content)->normal_maps[i].data);
  }

  free((*content)->albedo_maps);
  free((*content)->illuminance_maps);
  free((*content)->material_maps);
  free((*content)->normal_maps);
  free((*content)->texture_list->textures);
  free((*content)->texture_list);

  free(*content);
}

static size_t hash_djb2(unsigned char* str) {
  size_t hash = 5381;
  int c;

  while ((c = *(str++))) {
    hash = ((hash << 5) + hash) + c;
  }

  return hash;
}

/*
 * @result Index of texture if texture is already present, else TEXTURE_NONE
 */
static uint16_t find_texture(WavefrontTextureList* textures, uint32_t hash, WavefrontTextureInstanceType type) {
  for (uint32_t i = 0; i < textures->count; i++) {
    WavefrontTextureInstance tex = textures->textures[i];
    if (tex.hash == hash && tex.type == type)
      return tex.offset;
  }

  return TEXTURE_NONE;
}

static void add_texture(WavefrontTextureList* textures, uint32_t hash, WavefrontTextureInstanceType type, uint16_t offset) {
  if (textures->count == textures->length) {
    textures->length   = textures->length * 2;
    textures->textures = safe_realloc(textures->textures, sizeof(WavefrontTextureInstance) * textures->length);
  }

  WavefrontTextureInstance tex = {.hash = hash, .offset = offset, .type = type};

  textures->textures[textures->count++] = tex;
}

static void read_materials_file(WavefrontContent* _content, const char* filename) {
  log_message("Reading *.mtl file (%s)", filename);
  FILE* file = fopen(filename, "r");

  if (!file) {
    error_message("Could not read material file!");
    return;
  }

  WavefrontContent content = *_content;

  unsigned int materials_count        = content.materials_length;
  unsigned int albedo_maps_count      = content.albedo_maps_length;
  unsigned int illuminance_maps_count = content.illuminance_maps_length;
  unsigned int material_maps_count    = content.material_maps_length;
  unsigned int normal_maps_count      = content.normal_maps_length;

  char* line = malloc(LINE_SIZE);
  char* path = malloc(LINE_SIZE);

  while (!feof(file)) {
    fgets(line, LINE_SIZE, file);

    if (line[0] == 'n' && line[1] == 'e' && line[2] == 'w' && line[3] == 'm' && line[4] == 't' && line[5] == 'l') {
      ensure_capacity(content.materials, materials_count, content.materials_length, sizeof(WavefrontMaterial));
      sscanf(line, "%*s %s\n", path);
      size_t hash = hash_djb2((unsigned char*) path);

      content.materials[materials_count].hash                = hash;
      content.materials[materials_count].albedo_texture      = TEXTURE_NONE;
      content.materials[materials_count].illuminance_texture = TEXTURE_NONE;
      content.materials[materials_count].material_texture    = TEXTURE_NONE;
      content.materials[materials_count].normal_texture      = TEXTURE_NONE;
      materials_count++;
    }
    else if (line[0] == 'm' && line[1] == 'a' && line[2] == 'p' && line[3] == '_') {
      if (line[4] == 'K' && line[5] == 'd') {
        ensure_capacity(content.albedo_maps, albedo_maps_count, content.albedo_maps_length, sizeof(TextureRGBA));
        sscanf(line, "%*s %[^\n]\n", path);
        const size_t hash   = hash_djb2((unsigned char*) path);
        uint16_t texture_id = find_texture(content.texture_list, hash, WF_ALBEDO);

        if (texture_id == TEXTURE_NONE) {
          texture_id = albedo_maps_count++;
          add_texture(content.texture_list, hash, WF_ALBEDO, texture_id);
          content.albedo_maps[texture_id] = load_texture_from_png(path);
        }

        content.materials[materials_count - 1].albedo_texture = texture_id;
      }
      else if (line[4] == 'K' && line[5] == 'e') {
        ensure_capacity(content.illuminance_maps, illuminance_maps_count, content.illuminance_maps_length, sizeof(TextureRGBA));
        sscanf(line, "%*s %[^\n]\n", path);
        const size_t hash   = hash_djb2((unsigned char*) path);
        uint16_t texture_id = find_texture(content.texture_list, hash, WF_ILLUMINANCE);

        if (texture_id == TEXTURE_NONE) {
          texture_id = illuminance_maps_count++;
          add_texture(content.texture_list, hash, WF_ILLUMINANCE, texture_id);
          content.illuminance_maps[texture_id] = load_texture_from_png(path);
        }

        content.materials[materials_count - 1].illuminance_texture = texture_id;
      }
      else if (line[4] == 'N' && line[5] == 's') {
        ensure_capacity(content.material_maps, material_maps_count, content.material_maps_length, sizeof(TextureRGBA));
        sscanf(line, "%*s %[^\n]\n", path);
        const size_t hash   = hash_djb2((unsigned char*) path);
        uint16_t texture_id = find_texture(content.texture_list, hash, WF_MATERIAL);

        if (texture_id == TEXTURE_NONE) {
          texture_id = material_maps_count++;
          add_texture(content.texture_list, hash, WF_MATERIAL, texture_id);
          content.material_maps[texture_id] = load_texture_from_png(path);
        }

        content.materials[materials_count - 1].material_texture = texture_id;
      }
      else if (line[4] == 'B' || line[4] == 'b') {
        ensure_capacity(content.normal_maps, normal_maps_count, content.normal_maps_length, sizeof(TextureRGBA));

        // This breaks if the path has a space in it
        // Blender uses the option -bm 1.0 to specify that the normals are scaled by 1.0
        // Hence the output is map_Bump -bm 1.0 [Path]
        const char* normal_map_path = strrchr(line, ' ') + 1;
        sscanf(normal_map_path, "%[^\n]\n", path);

        const size_t hash   = hash_djb2((unsigned char*) path);
        uint16_t texture_id = find_texture(content.texture_list, hash, WF_NORMAL);

        if (texture_id == TEXTURE_NONE) {
          texture_id = normal_maps_count++;
          add_texture(content.texture_list, hash, WF_NORMAL, texture_id);
          content.normal_maps[texture_id] = load_texture_from_png(path);
        }

        content.materials[materials_count - 1].normal_texture = texture_id;
      }
    }
  }

  content.materials_length        = materials_count;
  content.materials               = safe_realloc(content.materials, sizeof(WavefrontMaterial) * content.materials_length);
  content.albedo_maps_length      = albedo_maps_count;
  content.albedo_maps             = safe_realloc(content.albedo_maps, sizeof(TextureRGBA) * content.albedo_maps_length);
  content.illuminance_maps_length = illuminance_maps_count;
  content.illuminance_maps        = safe_realloc(content.illuminance_maps, sizeof(TextureRGBA) * content.illuminance_maps_length);
  content.material_maps_length    = material_maps_count;
  content.material_maps           = safe_realloc(content.material_maps, sizeof(TextureRGBA) * content.material_maps_length);
  content.normal_maps_length      = normal_maps_count;
  content.normal_maps             = safe_realloc(content.normal_maps, sizeof(TextureRGBA) * content.normal_maps_length);

  *_content = content;

  fclose(file);

  log_message(
    "Material counts: %d (%d %d %d %d)", materials_count, albedo_maps_count, illuminance_maps_count, material_maps_count,
    normal_maps_count);
}

/*
 * Reads a line str of n floating point numbers and writes them into dst.
 * @param str String containing the floating point numbers.
 * @param n Number of floating point numbers.
 * @param dst Array the floating point numbers are written to.
 * @result Returns the number of floating point numbers written.
 */
static int read_float_line(const char* str, const int n, float* dst) {
  const char* rstr = str;
  for (int i = 0; i < n; i++) {
    char* new_rstr;
    dst[i] = strtof(rstr, &new_rstr);

    if (!new_rstr)
      return i + 1;

    rstr = (const char*) new_rstr;
  }

  return n;
}

/*
 * Reads face indices from a string. Note that quads have to be triangles fans.
 * @param str String containing the indices in Wavefront format.
 * @param face1 Pointer to Triangle which gets filled by the data.
 * @param face2 Pointer to Triangle which gets filled by the data.
 * @result Returns the number of triangles parsed.
 */
static int read_face(const char* str, WavefrontTriangle* face1, WavefrontTriangle* face2) {
  int ptr = 0;

  const unsigned int num_check = 0b00110000;
  const unsigned int num_mask  = 0b11110000;

  int data[12];
  unsigned int data_ptr = 0;

  int sign = 1;

  char c = str[ptr++];

  // Find first number
  while ((c & num_mask) != num_check && c != '-') {
    if (c == '\0' || c == '\r' || c == '\n')
      break;
    c = str[ptr++];
  }

  while (c != '\0' && c != '\r' && c != '\n') {
    if (c == '-') {
      sign = -1;
    }

    int value = 0;

    while ((c & num_mask) == num_check) {
      value = value * 10 + (int) (c & 0b00001111);

      c = str[ptr++];
    }

    if (c == '/' || c == ' ' || c == '\0' || c == '\r' || c == '\n') {
      data[data_ptr++] = value * sign;

      sign = 1;
    }

    if (c == '\0' || c == '\r' || c == '\n')
      break;

    c = str[ptr++];
  }

  int tris = 0;

  switch (data_ptr) {
    case 3:  // Triangle, Only v
    {
      memset(face1, 0, sizeof(WavefrontTriangle));
      face1->v1 = data[0];
      face1->v2 = data[1];
      face1->v3 = data[2];
      tris      = 1;
    } break;
    case 4:  // Quad, Only v
    {
      memset(face1, 0, sizeof(WavefrontTriangle));
      memset(face2, 0, sizeof(WavefrontTriangle));
      face1->v1 = data[0];
      face1->v2 = data[1];
      face1->v3 = data[2];
      face2->v1 = data[0];
      face2->v2 = data[2];
      face2->v3 = data[3];
      tris      = 2;
    } break;
    case 6:  // Triangle, Only v and vt
    {
      memset(face1, 0, sizeof(WavefrontTriangle));
      face1->v1  = data[0];
      face1->vt1 = data[1];
      face1->v2  = data[2];
      face1->vt2 = data[3];
      face1->v3  = data[4];
      face1->vt3 = data[5];
      tris       = 1;
    } break;
    case 8:  // Quad, Only v and vt
    {
      memset(face1, 0, sizeof(WavefrontTriangle));
      memset(face2, 0, sizeof(WavefrontTriangle));
      face1->v1  = data[0];
      face1->vt1 = data[1];
      face1->v2  = data[2];
      face1->vt2 = data[3];
      face1->v3  = data[4];
      face1->vt3 = data[5];
      face2->v1  = data[0];
      face2->vt1 = data[1];
      face2->v2  = data[4];
      face2->vt2 = data[5];
      face2->v3  = data[6];
      face2->vt3 = data[7];
      tris       = 2;
    } break;
    case 9:  // Triangle
    {
      face1->v1     = data[0];
      face1->vt1    = data[1];
      face1->vn1    = data[2];
      face1->v2     = data[3];
      face1->vt2    = data[4];
      face1->vn2    = data[5];
      face1->v3     = data[6];
      face1->vt3    = data[7];
      face1->vn3    = data[8];
      face1->object = 0;
      tris          = 1;
    } break;
    case 12:  // Quad
    {
      face1->v1     = data[0];
      face1->vt1    = data[1];
      face1->vn1    = data[2];
      face1->v2     = data[3];
      face1->vt2    = data[4];
      face1->vn2    = data[5];
      face1->v3     = data[6];
      face1->vt3    = data[7];
      face1->vn3    = data[8];
      face1->object = 0;
      face2->v1     = data[0];
      face2->vt1    = data[1];
      face2->vn1    = data[2];
      face2->v2     = data[6];
      face2->vt2    = data[7];
      face2->vn2    = data[8];
      face2->v3     = data[9];
      face2->vt3    = data[10];
      face2->vn3    = data[11];
      face2->object = 0;
      tris          = 2;
    } break;
    default: {
      error_message("A face is of unsupported format. %s\n", str);
      tris = 0;
    } break;
  }

  return tris;
}

int wavefront_read_file(WavefrontContent* _content, const char* filename) {
  log_message("Reading *.obj file (%s)", filename);
  bench_tic();
  FILE* file = fopen(filename, "r");

  if (!file) {
    error_message("File could not be opened!");
    return -1;
  }

  WavefrontContent content = *_content;

  const unsigned int vertices_offset = content.vertices_length;
  const unsigned int normals_offset  = content.normals_length;
  const unsigned int uvs_offset      = content.uvs_length;

  unsigned int triangles_count = content.triangles_length;
  int vertices_count           = content.vertices_length;
  int normals_count            = content.normals_length;
  int uvs_count                = content.uvs_length;
  unsigned int materials_count = content.materials_length;

  size_t* loaded_mtls             = malloc(sizeof(size_t) * 16);
  unsigned int loaded_mtls_count  = 0;
  unsigned int loaded_mtls_length = 16;

  uint16_t current_material = 0;

  char* path        = malloc(LINE_SIZE);
  char* read_buffer = malloc(READ_BUFFER_SIZE + LINE_SIZE);

  memset(read_buffer + READ_BUFFER_SIZE, 0, LINE_SIZE);

  size_t offset = 0;

  while (!feof(file)) {
    fread(read_buffer + offset, 1, READ_BUFFER_SIZE - offset, file);

    char* line = read_buffer;
    char* eol;

    while ((eol = strchr(line, '\n'))) {
      *eol = '\0';
      if (line[0] == 'v' && line[1] == ' ') {
        ensure_capacity(content.vertices, vertices_count, content.vertices_length, sizeof(WavefrontVertex));
        WavefrontVertex v;
        read_float_line(line + 2, 3, &v.x);
        content.vertices[vertices_count] = v;
        vertices_count++;
      }
      else if (line[0] == 'v' && line[1] == 'n') {
        ensure_capacity(content.normals, normals_count, content.normals_length, sizeof(WavefrontNormal));
        WavefrontNormal n;
        read_float_line(line + 3, 3, &n.x);
        content.normals[normals_count] = n;
        normals_count++;
      }
      else if (line[0] == 'v' && line[1] == 't') {
        ensure_capacity(content.uvs, uvs_count, content.uvs_length, sizeof(WavefrontUV));
        WavefrontUV uv;
        read_float_line(line + 3, 2, &uv.u);
        content.uvs[uvs_count] = uv;
        uvs_count++;
      }
      else if (line[0] == 'f') {
        ensure_capacity(content.triangles, triangles_count, content.triangles_length, sizeof(WavefrontTriangle));
        WavefrontTriangle face1;
        WavefrontTriangle face2;
        const int returned_faces = read_face(line, &face1, &face2);

        if (returned_faces >= 1) {
          face1.object = current_material;
          face1.v1 += vertices_offset;
          face1.v2 += vertices_offset;
          face1.v3 += vertices_offset;
          face1.vn1 += normals_offset;
          face1.vn2 += normals_offset;
          face1.vn3 += normals_offset;
          face1.vt1 += uvs_offset;
          face1.vt2 += uvs_offset;
          face1.vt3 += uvs_offset;

          content.triangles[triangles_count++] = face1;
        }

        if (returned_faces >= 2) {
          face2.object = current_material;
          face2.v1 += vertices_offset;
          face2.v2 += vertices_offset;
          face2.v3 += vertices_offset;
          face2.vn1 += normals_offset;
          face2.vn2 += normals_offset;
          face2.vn3 += normals_offset;
          face2.vt1 += uvs_offset;
          face2.vt2 += uvs_offset;
          face2.vt3 += uvs_offset;

          content.triangles[triangles_count++] = face2;
        }
      }
      else if (line[0] == 'm' && line[1] == 't' && line[2] == 'l' && line[3] == 'l' && line[4] == 'i' && line[5] == 'b') {
        sscanf(line, "%*s %[^\n]", path);
        size_t hash = hash_djb2((unsigned char*) path);

        int already_loaded = 0;

        for (unsigned int i = 0; i < loaded_mtls_count; i++) {
          if (loaded_mtls[i] == hash)
            already_loaded = 1;
        }

        if (!already_loaded) {
          ensure_capacity(loaded_mtls, loaded_mtls_count, loaded_mtls_length, sizeof(size_t));
          loaded_mtls[loaded_mtls_count++] = hash;
          read_materials_file(&content, path);
          materials_count = content.materials_length;
        }
      }
      else if (line[0] == 'u' && line[1] == 's' && line[2] == 'e' && line[3] == 'm' && line[4] == 't' && line[5] == 'l') {
        sscanf(line, "%*s %[^\n]", path);
        size_t hash      = hash_djb2((unsigned char*) path);
        current_material = 0;
        for (unsigned int i = 1; i < materials_count; i++) {
          if (content.materials[i].hash == hash) {
            current_material = i;
            break;
          }
        }
      }

      line = eol + 1;
    }

    offset = strlen(line);
    memcpy(read_buffer, line, offset);
  }

  content.vertices_length  = vertices_count;
  content.vertices         = safe_realloc(content.vertices, sizeof(WavefrontVertex) * content.vertices_length);
  content.normals_length   = normals_count;
  content.normals          = safe_realloc(content.normals, sizeof(WavefrontNormal) * content.normals_length);
  content.uvs_length       = uvs_count;
  content.uvs              = safe_realloc(content.uvs, sizeof(WavefrontUV) * content.uvs_length);
  content.triangles_length = triangles_count;
  content.triangles        = safe_realloc(content.triangles, sizeof(WavefrontTriangle) * content.triangles_length);

  *_content = content;

  free(loaded_mtls);
  free(path);
  free(read_buffer);

  bench_toc("Reading *.obj File");

  log_message("Mesh: Verts: %d Tris: %d", vertices_count, triangles_count);

  return 0;
}

TextureAssignment* wavefront_generate_texture_assignments(WavefrontContent* content) {
  TextureAssignment* texture_assignments = malloc(sizeof(TextureAssignment) * content->materials_length);

  for (unsigned int i = 0; i < content->materials_length; i++) {
    TextureAssignment assignment;
    assignment.albedo_map      = content->materials[i].albedo_texture;
    assignment.illuminance_map = content->materials[i].illuminance_texture;
    assignment.material_map    = content->materials[i].material_texture;
    assignment.normal_map      = content->materials[i].normal_texture;

    texture_assignments[i] = assignment;
  }

  return texture_assignments;
}

unsigned int wavefront_convert_content(WavefrontContent* content, Triangle** triangles, TriangleGeomData* data) {
  bench_tic();

  static_assert(sizeof(WavefrontVertex) == 3 * sizeof(float), "Wavefront Vertex must be a struct of 3 floats!.");

  unsigned int count = content->triangles_length;

  *triangles = (Triangle*) malloc(sizeof(Triangle) * count);

  data->index_buffer  = (uint32_t*) malloc(sizeof(uint32_t) * 4 * count);
  data->vertex_buffer = (float*) malloc(sizeof(float) * 4 * content->vertices_length);
  data->vertex_count  = content->vertices_length;

  for (int j = 0; j < content->vertices_length; j++) {
    data->vertex_buffer[j * 4 + 0] = content->vertices[j].x;
    data->vertex_buffer[j * 4 + 1] = content->vertices[j].y;
    data->vertex_buffer[j * 4 + 2] = content->vertices[j].z;
  }

  unsigned int ptr                 = 0;
  unsigned int index_triplet_count = 0;

  for (unsigned int j = 0; j < count; j++) {
    WavefrontTriangle t = content->triangles[j];
    Triangle triangle;

    WavefrontVertex v;

    t.v1 += (t.v1 < 0) ? content->vertices_length + 1 : 0;

    if (t.v1 > content->vertices_length) {
      continue;
    }
    else {
      v = content->vertices[t.v1 - 1];
    }

    triangle.vertex.x = v.x;
    triangle.vertex.y = v.y;
    triangle.vertex.z = v.z;

    t.v2 += (t.v2 < 0) ? content->vertices_length + 1 : 0;

    if (t.v2 > content->vertices_length) {
      continue;
    }
    else {
      v = content->vertices[t.v2 - 1];
    }

    triangle.edge1.x = v.x - triangle.vertex.x;
    triangle.edge1.y = v.y - triangle.vertex.y;
    triangle.edge1.z = v.z - triangle.vertex.z;

    t.v3 += (t.v3 < 0) ? content->vertices_length + 1 : 0;

    if (t.v3 > content->vertices_length) {
      continue;
    }
    else {
      v = content->vertices[t.v3 - 1];
    }

    triangle.edge2.x = v.x - triangle.vertex.x;
    triangle.edge2.y = v.y - triangle.vertex.y;
    triangle.edge2.z = v.z - triangle.vertex.z;

    if (
      fabsf(triangle.edge1.x) < FLT_EPSILON && fabsf(triangle.edge1.y) < FLT_EPSILON && fabsf(triangle.edge1.z) < FLT_EPSILON
      && fabsf(triangle.edge2.x) < FLT_EPSILON && fabsf(triangle.edge2.y) < FLT_EPSILON && fabsf(triangle.edge2.z) < FLT_EPSILON) {
      continue;
    }

    data->index_buffer[index_triplet_count * 4 + 0] = t.v1;
    data->index_buffer[index_triplet_count * 4 + 1] = t.v2;
    data->index_buffer[index_triplet_count * 4 + 2] = t.v3;

    WavefrontUV uv;

    t.vt1 += (t.vt1 < 0) ? content->uvs_length + 1 : 0;

    if (t.vt1 > content->uvs_length || t.vt1 == 0) {
      uv.u = 0.0f;
      uv.v = 0.0f;
    }
    else {
      uv = content->uvs[t.vt1 - 1];
    }

    triangle.vertex_texture.u = uv.u;
    triangle.vertex_texture.v = uv.v;

    t.vt2 += (t.vt2 < 0) ? content->uvs_length + 1 : 0;

    if (t.vt2 > content->uvs_length || t.vt2 == 0) {
      uv.u = 0.0f;
      uv.v = 0.0f;
    }
    else {
      uv = content->uvs[t.vt2 - 1];
    }

    triangle.edge1_texture.u = uv.u - triangle.vertex_texture.u;
    triangle.edge1_texture.v = uv.v - triangle.vertex_texture.v;

    t.vt3 += (t.vt3 < 0) ? content->uvs_length + 1 : 0;

    if (t.vt3 > content->uvs_length || t.vt3 == 0) {
      uv.u = 0.0f;
      uv.v = 0.0f;
    }
    else {
      uv = content->uvs[t.vt3 - 1];
    }

    triangle.edge2_texture.u = uv.u - triangle.vertex_texture.u;
    triangle.edge2_texture.v = uv.v - triangle.vertex_texture.v;

    WavefrontNormal n;

    t.vn1 += (t.vn1 < 0) ? content->normals_length + 1 : 0;

    if (t.vn1 > content->normals_length || t.vn1 == 0) {
      n.x = 0.0f;
      n.y = 0.0f;
      n.z = 0.0f;
    }
    else {
      n = content->normals[t.vn1 - 1];

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

    t.vn2 += (t.vn2 < 0) ? content->normals_length + 1 : 0;

    if (t.vn2 > content->normals_length || t.vn2 == 0) {
      n.x = 0.0f;
      n.y = 0.0f;
      n.z = 0.0f;
    }
    else {
      n = content->normals[t.vn2 - 1];

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

    t.vn3 += (t.vn3 < 0) ? content->normals_length + 1 : 0;

    if (t.vn3 > content->normals_length || t.vn3 == 0) {
      n.x = 0.0f;
      n.y = 0.0f;
      n.z = 0.0f;
    }
    else {
      n = content->normals[t.vn3 - 1];

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
    triangle.light_id    = LIGHT_ID_NONE;

    (*triangles)[ptr++] = triangle;
    index_triplet_count++;
  }

  *triangles = (Triangle*) realloc(*triangles, sizeof(Triangle) * ptr);

  data->index_buffer   = (uint32_t*) realloc(data->index_buffer, sizeof(uint32_t) * 4 * index_triplet_count);
  data->index_count    = 3 * index_triplet_count;
  data->triangle_count = index_triplet_count;

  bench_toc("Converting Mesh");

  return ptr;
}
