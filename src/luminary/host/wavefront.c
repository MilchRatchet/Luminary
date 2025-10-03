#include "wavefront.h"

#include <assert.h>
#include <float.h>
#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "host_intrinsics.h"
#include "internal_error.h"
#include "internal_path.h"
#include "material.h"
#include "png.h"
#include "utils.h"

//
// Implementation of mtl parser based on http://paulbourke.net/dataformats/mtl/.
//

#define LINE_SIZE 256 * 1024               // 256 KB
#define READ_BUFFER_SIZE 16 * 1024 * 1024  // 16 MB

static WavefrontMaterial _wavefront_get_default_material() {
  WavefrontMaterial default_material;

  default_material.hash                    = 0;
  default_material.diffuse_reflectivity.r  = 0.9f;
  default_material.diffuse_reflectivity.g  = 0.9f;
  default_material.diffuse_reflectivity.b  = 0.9f;
  default_material.dissolve                = 1.0f;
  default_material.specular_reflectivity.r = 0.0f;
  default_material.specular_reflectivity.g = 0.0f;
  default_material.specular_reflectivity.b = 0.0f;
  default_material.specular_exponent       = 300.0f;
  default_material.emission.r              = 0.0f;
  default_material.emission.g              = 0.0f;
  default_material.emission.b              = 0.0f;
  default_material.refraction_index        = 1.0f;
  default_material.texture[WF_ALBEDO]      = TEXTURE_NONE;
  default_material.texture[WF_LUMINANCE]   = TEXTURE_NONE;
  default_material.texture[WF_ROUGHNESS]   = TEXTURE_NONE;
  default_material.texture[WF_METALLIC]    = TEXTURE_NONE;
  default_material.texture[WF_NORMAL]      = TEXTURE_NONE;

  return default_material;
}

LuminaryResult wavefront_create(WavefrontContent** content, WavefrontArguments args) {
  __CHECK_NULL_ARGUMENT(content);

  __FAILURE_HANDLE(host_malloc(content, sizeof(WavefrontContent)));

  (*content)->state = WAVEFRONT_CONTENT_STATE_READY_TO_READ;

  (*content)->args = args;

  __FAILURE_HANDLE(array_create(&(*content)->vertices, sizeof(WavefrontVertex), 16));
  __FAILURE_HANDLE(array_create(&(*content)->normals, sizeof(WavefrontNormal), 16));
  __FAILURE_HANDLE(array_create(&(*content)->uvs, sizeof(WavefrontUV), 16));

  __FAILURE_HANDLE(array_create(&(*content)->triangles, sizeof(WavefrontTriangle), 16));
  __FAILURE_HANDLE(array_create(&(*content)->materials, sizeof(WavefrontMaterial), 16));

  WavefrontMaterial default_material = _wavefront_get_default_material();

  __FAILURE_HANDLE(array_push(&(*content)->materials, &default_material));

  __FAILURE_HANDLE(array_create(&(*content)->textures, sizeof(Texture*), 16));
  __FAILURE_HANDLE(array_create(&(*content)->texture_instances, sizeof(WavefrontTextureInstance), 16));

  __FAILURE_HANDLE(array_create(&(*content)->object_names, sizeof(char*), 16));

  return LUMINARY_SUCCESS;
}

LuminaryResult wavefront_destroy(WavefrontContent** content) {
  __CHECK_NULL_ARGUMENT(content);

  __FAILURE_HANDLE(array_destroy(&(*content)->vertices));
  __FAILURE_HANDLE(array_destroy(&(*content)->normals));
  __FAILURE_HANDLE(array_destroy(&(*content)->uvs));
  __FAILURE_HANDLE(array_destroy(&(*content)->triangles));
  __FAILURE_HANDLE(array_destroy(&(*content)->materials));

  uint32_t num_textures;
  __FAILURE_HANDLE(array_get_num_elements((*content)->textures, &num_textures));

  for (uint32_t texture_id = 0; texture_id < num_textures; texture_id++) {
    __FAILURE_HANDLE(texture_destroy(&(*content)->textures[texture_id]));
  }

  __FAILURE_HANDLE(array_destroy(&(*content)->textures));
  __FAILURE_HANDLE(array_destroy(&(*content)->texture_instances));

  uint32_t num_objects;
  __FAILURE_HANDLE(array_get_num_elements((*content)->object_names, &num_objects));

  for (uint32_t object_id = 0; object_id < num_objects; object_id++) {
    __FAILURE_HANDLE(host_free(&(*content)->object_names[object_id]));
  }

  __FAILURE_HANDLE(array_destroy(&(*content)->object_names));

  __FAILURE_HANDLE(host_free(content));

  return LUMINARY_SUCCESS;
}

/*
 * Reads a line str of n floating point numbers and writes them into dst.
 * @param str String containing the floating point numbers.
 * @param n Number of floating point numbers.
 * @param dst Array the floating point numbers are written to.
 * @result Returns the number of floating point numbers written.
 */
static uint32_t read_float_line(const char* str, const uint32_t n, float* dst) {
  const char* rstr = str;
  for (uint32_t i = 0; i < n; i++) {
    char* new_rstr;
    dst[i] = strtof(rstr, &new_rstr);

    if (!new_rstr)
      return i + 1;

    rstr = (const char*) new_rstr;
  }

  return n;
}

static size_t hash_djb2(unsigned char* str) {
  size_t hash = 5381;
  size_t c;

  while ((c = *(str++))) {
    hash = ((hash << 5) + hash) + c;
  }

  return hash;
}

/*
 * @result Index of texture if texture is already present, else TEXTURE_NONE
 */
static uint16_t find_texture(const WavefrontContent* content, uint32_t hash) {
  uint32_t texture_count;
  __FAILURE_HANDLE(array_get_num_elements(content->texture_instances, &texture_count));

  for (uint32_t tex_id = 0; tex_id < texture_count; tex_id++) {
    WavefrontTextureInstance tex = content->texture_instances[tex_id];
    if (tex.hash == hash)
      return tex.texture_id;
  }

  return TEXTURE_NONE;
}

static LuminaryResult _wavefront_parse_map(
  WavefrontContent* content, Path* mtl_file_path, Queue* queue, const char* line, const size_t line_len) {
  if (line_len < 8) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Line is too short to be a valid map_ line.");
  }

  if (strncmp(line, "map_", 4)) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Line is not a map_ line.");
  }

  uint32_t path_offset = 7;

  // Determine type of map
  WavefrontTextureType type;
  if (!strncmp(line + 4, "Kd", 2)) {
    type = WF_ALBEDO;
  }
  else if (!strncmp(line + 4, "Ke", 2)) {
    type = WF_LUMINANCE;
  }
  else if (!strncmp(line + 4, "Ns", 2)) {
    type = WF_ROUGHNESS;
  }
  else if (!strncmp(line + 4, "refl", 4)) {
    type        = WF_METALLIC;
    path_offset = 9;
  }
  else if (!strncmp(line + 4, "Bump", 4)) {
    type        = WF_NORMAL;
    path_offset = 9;
  }
  else {
    // Not a supported type.
    return LUMINARY_SUCCESS;
  }

  // Find path
  const char* path = line + path_offset;

  // The next character is a valid memory address because at least a \0 must follow.
  if (path[0] == '-') {
    // We now parse all commands until we find a word that is not a command argument and does not start with a -.

    // Valid address, same argument.
    const char* command = path + 1;
    bool is_command     = true;

    while (is_command) {
      // This tells us how many arguments will follow this command, this is how often we have to skip words.
      uint32_t num_args = 0;

      if (command[0] == 'o' || command[0] == 's') {
        num_args = 3;  // o or s
      }
      else if (command[0] == 't') {
        if (command[1] == ' ') {
          num_args = 3;  // t
        }
        else {
          num_args = 1;  // texres
        }
      }
      else if (command[0] == 'm') {
        num_args = 2;  // mm
      }
      else if (command[0] == 'c' || command[0] == 'b') {
        num_args = 1;  // cc or clamp or blendu or blendv or bm
      }

      for (uint32_t arg = 0; arg <= num_args; arg++) {
        command = strchr(command, ' ');

        if (command == (char*) 0) {
          __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Something went wrong parsing the following line in an *.mtl file: %s", line);
        }

        is_command = (command[0] == '-');
        command++;
      }
    }

    path = command;
  }

  const size_t hash   = hash_djb2((unsigned char*) path);
  uint16_t texture_id = find_texture(content, hash);

  if (texture_id == TEXTURE_NONE) {
    uint32_t new_texture_id;
    __FAILURE_HANDLE(array_get_num_elements(content->textures, &new_texture_id));

    if (new_texture_id > 0xFFFFu) {
      __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Exceeded limit of 65535 textures.");
    }

    texture_id = (uint16_t) new_texture_id;

    WavefrontTextureInstance texture_instance;
    texture_instance.hash       = hash;
    texture_instance.texture_id = texture_id;

    const char* tex_file_path;
    __FAILURE_HANDLE(path_apply(mtl_file_path, path, &tex_file_path));

    Texture* tex;
    __FAILURE_HANDLE(texture_create(&tex));

    // Scene textures require mipmapping
    tex->mipmap = TEXTURE_MIPMAP_MODE_GENERATE;

    __FAILURE_HANDLE(texture_load_async(tex, queue, tex_file_path));

    __FAILURE_HANDLE(array_push(&content->textures, &tex));
    __FAILURE_HANDLE(array_push(&content->texture_instances, &texture_instance));
  }

  uint32_t current_material_ptr;
  __FAILURE_HANDLE(array_get_num_elements(content->materials, &current_material_ptr));
  current_material_ptr--;

  content->materials[current_material_ptr].texture[type] = texture_id;

  return LUMINARY_SUCCESS;
}

static LuminaryResult read_materials_file(WavefrontContent* content, Path* mtl_file_path, Queue* queue) {
  __CHECK_NULL_ARGUMENT(content);
  __CHECK_NULL_ARGUMENT(mtl_file_path);

  const char* mtl_file_path_string;
  __FAILURE_HANDLE(path_apply(mtl_file_path, (const char*) 0, &mtl_file_path_string));

  log_message("Reading *.mtl file (%s)", mtl_file_path_string);
  FILE* file = fopen(mtl_file_path_string, "r");

  if (!file) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Failed to open file *.mtl file (%s)", mtl_file_path_string);
  }

  // Invalidate file path string.
  mtl_file_path_string = (const char*) 0;

  char* line;
  __FAILURE_HANDLE(host_malloc(&line, LINE_SIZE));

  uint32_t current_material_ptr;
  __FAILURE_HANDLE(array_get_num_elements(content->materials, &current_material_ptr));

  // PTR to the last element in the array.
  current_material_ptr--;

  while (!feof(file)) {
    fgets(line, LINE_SIZE, file);

    size_t line_len = strlen(line);

    if (!line_len)
      continue;

    // Get rid of the newline character. This makes things easier because any file path will be the end of the line.
    if (line[line_len - 1] == '\n')
      line[line_len - 1] = '\0';

    if (line[0] == 'n' && line[1] == 'e' && line[2] == 'w' && line[3] == 'm' && line[4] == 't' && line[5] == 'l') {
      char* name  = line + 7;
      size_t hash = hash_djb2((unsigned char*) name);

      WavefrontMaterial mat = _wavefront_get_default_material();
      mat.hash              = hash;

      __FAILURE_HANDLE(array_push(&content->materials, &mat));

      current_material_ptr++;
    }
    else if (line[0] == 'K' && line[1] == 'd') {
      char* value = line + 3;

      float diffuse_reflectivity[3];
      if (read_float_line(value, 3, diffuse_reflectivity) == 3) {
        content->materials[current_material_ptr].diffuse_reflectivity.r = diffuse_reflectivity[0];
        content->materials[current_material_ptr].diffuse_reflectivity.g = diffuse_reflectivity[1];
        content->materials[current_material_ptr].diffuse_reflectivity.b = diffuse_reflectivity[2];
      }
      else {
        warn_message("Expected three values in diffuse reflectivity in *.mtl file but didn't find three numbers. Line: %s.", line);
      }
    }
    else if (line[0] == 'd') {
      char* value = line + 2;

      float dissolve;
      if (read_float_line(value, 1, &dissolve)) {
        content->materials[current_material_ptr].dissolve = dissolve;
      }
      else {
        warn_message("Expected dissolve in *.mtl file but didn't find a number. Line: %s.", line);
      }
    }
    else if (line[0] == 'K' && line[1] == 's') {
      char* value = line + 3;

      float specular_reflectivity[3];
      if (read_float_line(value, 3, specular_reflectivity) == 3) {
        content->materials[current_material_ptr].specular_reflectivity.r = specular_reflectivity[0];
        content->materials[current_material_ptr].specular_reflectivity.g = specular_reflectivity[1];
        content->materials[current_material_ptr].specular_reflectivity.b = specular_reflectivity[2];
      }
      else {
        warn_message("Expected three values in specular reflectivity in *.mtl file but didn't find three numbers. Line: %s.", line);
      }
    }
    else if (line[0] == 'N' && line[1] == 's') {
      char* value = line + 3;

      float specular_exponent;
      if (read_float_line(value, 1, &specular_exponent)) {
        content->materials[current_material_ptr].specular_exponent = specular_exponent;
      }
      else {
        warn_message("Expected specular_exponent in *.mtl file but didn't find a number. Line: %s.", line);
      }
    }
    else if (line[0] == 'K' && line[1] == 'e') {
      char* value = line + 3;

      float emission[3];
      if (read_float_line(value, 3, emission) == 3) {
        content->materials[current_material_ptr].emission.r = emission[0] * content->args.emission_scale;
        content->materials[current_material_ptr].emission.g = emission[1] * content->args.emission_scale;
        content->materials[current_material_ptr].emission.b = emission[2] * content->args.emission_scale;
      }
      else {
        warn_message("Expected three values in emission in *.mtl file but didn't find three numbers. Line: %s.", line);
      }
    }
    else if (line[0] == 'N' && line[1] == 'i') {
      char* value = line + 3;

      float refraction_index;
      if (read_float_line(value, 1, &refraction_index)) {
        content->materials[current_material_ptr].refraction_index = refraction_index;
      }
      else {
        warn_message("Expected refraction index in *.mtl file but didn't find a number. Line: %s.", line);
      }
    }
    else if (line[0] == 'm' && line[1] == 'a') {
      __FAILURE_HANDLE(_wavefront_parse_map(content, mtl_file_path, queue, line, line_len));
    }
  }

  __FAILURE_HANDLE(host_free(&line));

  fclose(file);

  return LUMINARY_SUCCESS;
}

/*
 * Reads face indices from a string. Note that quads have to be triangles fans.
 * @param str String containing the indices in Wavefront format.
 * @param face1 Pointer to Triangle which gets filled by the data.
 * @param face2 Pointer to Triangle which gets filled by the data.
 * @result Returns the number of triangles parsed.
 */
static uint32_t read_face(const char* str, WavefrontTriangle* face1, WavefrontTriangle* face2) {
  uint32_t ptr = 0;

  const uint32_t num_check = 0b00110000;
  const uint32_t num_mask  = 0b11110000;

  uint32_t data[12];
  uint32_t data_ptr = 0;

  int32_t sign = 1;

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

    int32_t value = 0;

    while ((c & num_mask) == num_check) {
      value = value * 10 + (int32_t) (c & 0b00001111);

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

  uint32_t tris = 0;

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
      face1->v1  = data[0];
      face1->vt1 = data[1];
      face1->vn1 = data[2];
      face1->v2  = data[3];
      face1->vt2 = data[4];
      face1->vn2 = data[5];
      face1->v3  = data[6];
      face1->vt3 = data[7];
      face1->vn3 = data[8];
      tris       = 1;
    } break;
    case 12:  // Quad
    {
      face1->v1  = data[0];
      face1->vt1 = data[1];
      face1->vn1 = data[2];
      face1->v2  = data[3];
      face1->vt2 = data[4];
      face1->vn2 = data[5];
      face1->v3  = data[6];
      face1->vt3 = data[7];
      face1->vn3 = data[8];
      face2->v1  = data[0];
      face2->vt1 = data[1];
      face2->vn1 = data[2];
      face2->v2  = data[6];
      face2->vt2 = data[7];
      face2->vn2 = data[8];
      face2->v3  = data[9];
      face2->vt3 = data[10];
      face2->vn3 = data[11];
      tris       = 2;
    } break;
    default: {
      error_message("A face is of unsupported format. %s\n", str);
      tris = 0;
    } break;
  }

  return tris;
}

LuminaryResult wavefront_read_file(WavefrontContent* content, Path* wavefront_file_path, Queue* queue) {
  __CHECK_NULL_ARGUMENT(content);
  __CHECK_NULL_ARGUMENT(wavefront_file_path);

  if (content->state != WAVEFRONT_CONTENT_STATE_READY_TO_READ) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Wavefront content was in an illegal state.");
  }

  content->state = WAVEFRONT_CONTENT_STATE_READY_TO_CONVERT;

  const char* file_path_string;
  __FAILURE_HANDLE(path_apply(wavefront_file_path, (const char*) 0, &file_path_string));

  log_message("Reading *.obj file (%s)", file_path_string);
  FILE* file = fopen(file_path_string, "r");

  if (!file) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "File %s could not be opened!", file_path_string);
  }

  // Invalidate this string as we will not need it again.
  file_path_string = (const char*) 0;

  uint32_t vertices_offset;
  __FAILURE_HANDLE(array_get_num_elements(content->vertices, &vertices_offset));

  uint32_t normals_offset;
  __FAILURE_HANDLE(array_get_num_elements(content->normals, &normals_offset));

  uint32_t uvs_offset;
  __FAILURE_HANDLE(array_get_num_elements(content->uvs, &uvs_offset));

  ARRAY size_t* loaded_mtls;
  __FAILURE_HANDLE(array_create(&loaded_mtls, sizeof(size_t), 16));

  uint16_t current_material = 0;

  char* path;
  __FAILURE_HANDLE(host_malloc(&path, LINE_SIZE));

  char* read_buffer;
  __FAILURE_HANDLE(host_malloc(&read_buffer, READ_BUFFER_SIZE));

  char* read_buffer_swap;
  __FAILURE_HANDLE(host_malloc(&read_buffer_swap, READ_BUFFER_SIZE));

  // NULL terminate the buffers.
  read_buffer[READ_BUFFER_SIZE - 1]      = '\0';
  read_buffer_swap[READ_BUFFER_SIZE - 1] = '\0';

  uint16_t current_object = UINT16_MAX;

  size_t offset = 0;

  while (!feof(file)) {
    // We only read up to the second last byte in the dst buffer to keep the
    // buffer NULL terminated.
    fread(read_buffer + offset, 1, READ_BUFFER_SIZE - offset - 1, file);

    char* line = read_buffer;
    char* eol;

    while ((eol = strchr(line, '\n'))) {
      *eol = '\0';
      if (line[0] == 'v' && line[1] == ' ') {
        WavefrontVertex v;
        read_float_line(line + 2, 3, &v.x);

        __FAILURE_HANDLE(array_push(&content->vertices, &v));
      }
      else if (line[0] == 'v' && line[1] == 'n') {
        WavefrontNormal n;
        read_float_line(line + 3, 3, &n.x);

        __FAILURE_HANDLE(array_push(&content->normals, &n));
      }
      else if (line[0] == 'v' && line[1] == 't') {
        WavefrontUV uv;
        read_float_line(line + 3, 2, &uv.u);

        __FAILURE_HANDLE(array_push(&content->uvs, &uv));
      }
      else if (line[0] == 'f') {
        WavefrontTriangle face1;
        WavefrontTriangle face2;
        const uint32_t returned_faces = read_face(line, &face1, &face2);

        if (returned_faces >= 1) {
          face1.v1 += vertices_offset;
          face1.v2 += vertices_offset;
          face1.v3 += vertices_offset;
          face1.vn1 += normals_offset;
          face1.vn2 += normals_offset;
          face1.vn3 += normals_offset;
          face1.vt1 += uvs_offset;
          face1.vt2 += uvs_offset;
          face1.vt3 += uvs_offset;
          face1.material = current_material;
          face1.object   = current_object;

          __FAILURE_HANDLE(array_push(&content->triangles, &face1));
        }

        if (returned_faces >= 2) {
          face2.v1 += vertices_offset;
          face2.v2 += vertices_offset;
          face2.v3 += vertices_offset;
          face2.vn1 += normals_offset;
          face2.vn2 += normals_offset;
          face2.vn3 += normals_offset;
          face2.vt1 += uvs_offset;
          face2.vt2 += uvs_offset;
          face2.vt3 += uvs_offset;
          face2.material = current_material;
          face2.object   = current_object;

          __FAILURE_HANDLE(array_push(&content->triangles, &face2));
        }
      }
      else if (line[0] == 'o') {
        sscanf(line, "%*s %[^\n]", path);

        const size_t string_len = strlen(path);

        char* object_name;
        __FAILURE_HANDLE(host_malloc(&object_name, string_len + 1));

        memcpy(object_name, path, string_len);
        object_name[string_len] = '\0';

        __FAILURE_HANDLE(array_push(&content->object_names, &object_name));

        current_object++;
      }
      else if (line[0] == 'm' && line[1] == 't' && line[2] == 'l' && line[3] == 'l' && line[4] == 'i' && line[5] == 'b') {
        sscanf(line, "%*s %[^\n]", path);
        const size_t hash = hash_djb2((unsigned char*) path);

        bool already_loaded = false;

        uint32_t loaded_mtls_count;
        __FAILURE_HANDLE(array_get_num_elements(loaded_mtls, &loaded_mtls_count));

        for (uint32_t mtl_id = 0; mtl_id < loaded_mtls_count; mtl_id++) {
          if (loaded_mtls[mtl_id] == hash)
            already_loaded = true;
        }

        if (!already_loaded) {
          __FAILURE_HANDLE(array_push(&loaded_mtls, &hash));

          Path* mtl_file_path;
          __FAILURE_HANDLE(path_extend(&mtl_file_path, wavefront_file_path, path));

          __FAILURE_HANDLE(read_materials_file(content, mtl_file_path, queue));

          __FAILURE_HANDLE(luminary_path_destroy(&mtl_file_path));
        }
      }
      else if (line[0] == 'u' && line[1] == 's' && line[2] == 'e' && line[3] == 'm' && line[4] == 't' && line[5] == 'l') {
        sscanf(line, "%*s %[^\n]", path);
        size_t hash      = hash_djb2((unsigned char*) path);
        current_material = 0;

        uint32_t material_count;
        __FAILURE_HANDLE(array_get_num_elements(content->materials, &material_count));

        for (uint32_t material_id = 1; material_id < material_count; material_id++) {
          if (content->materials[material_id].hash == hash) {
            current_material = material_id;
            break;
          }
        }
      }

      line = eol + 1;
    }

    offset = strlen(line);

    memcpy(read_buffer_swap, line, offset);
    memcpy(read_buffer, read_buffer_swap, offset);
  }

  __FAILURE_HANDLE(array_destroy(&loaded_mtls));
  __FAILURE_HANDLE(host_free(&path));
  __FAILURE_HANDLE(host_free(&read_buffer));
  __FAILURE_HANDLE(host_free(&read_buffer_swap));

  return LUMINARY_SUCCESS;
}

static LuminaryResult _wavefront_convert_materials(WavefrontContent* content, ARRAYPTR Material** materials, ARRAYPTR Texture*** textures) {
  __CHECK_NULL_ARGUMENT(content);
  __CHECK_NULL_ARGUMENT(materials);

  uint32_t material_count;
  __FAILURE_HANDLE(array_get_num_elements(content->materials, &material_count));

  uint32_t texture_offset;
  __FAILURE_HANDLE(array_get_num_elements(*textures, &texture_offset));

  uint32_t texture_count;
  __FAILURE_HANDLE(array_get_num_elements(content->texture_instances, &texture_count));

  for (uint32_t tex_id = 0; tex_id < texture_count; tex_id++) {
    const WavefrontTextureInstance instance = content->texture_instances[tex_id];
    const Texture* tex                      = content->textures[instance.texture_id];

    __FAILURE_HANDLE(array_push(textures, &tex));
  }

  __FAILURE_HANDLE(array_resize(&content->texture_instances, 0));
  __FAILURE_HANDLE(array_resize(&content->textures, 0));

  uint32_t material_id_offset;
  __FAILURE_HANDLE(array_get_num_elements(*materials, &material_id_offset));

  for (uint32_t mat_id = 0; mat_id < material_count; mat_id++) {
    const WavefrontMaterial wavefront_mat = content->materials[mat_id];

    const bool has_albedo_tex    = (wavefront_mat.texture[WF_ALBEDO] != TEXTURE_NONE);
    const bool has_luminance_tex = (wavefront_mat.texture[WF_LUMINANCE] != TEXTURE_NONE);
    const bool has_roughness_tex = (wavefront_mat.texture[WF_ROUGHNESS] != TEXTURE_NONE);
    const bool has_metallic_tex  = (wavefront_mat.texture[WF_METALLIC] != TEXTURE_NONE);
    const bool has_normal_tex    = (wavefront_mat.texture[WF_NORMAL] != TEXTURE_NONE);
    const bool has_emission = (wavefront_mat.emission.r > 0.0f) || (wavefront_mat.emission.g > 0.0f) || (wavefront_mat.emission.b > 0.0f);

    Material mat;
    __FAILURE_HANDLE(material_get_default(&mat));

    mat.id                       = material_id_offset + mat_id;
    mat.base_substrate           = LUMINARY_MATERIAL_BASE_SUBSTRATE_OPAQUE;
    mat.albedo.r                 = wavefront_mat.diffuse_reflectivity.r;
    mat.albedo.g                 = wavefront_mat.diffuse_reflectivity.g;
    mat.albedo.b                 = wavefront_mat.diffuse_reflectivity.b;
    mat.albedo.a                 = wavefront_mat.dissolve;
    mat.emission                 = wavefront_mat.emission;
    mat.emission_scale           = content->args.emission_scale;
    mat.refraction_index         = wavefront_mat.refraction_index;
    mat.roughness                = 1.0f - wavefront_mat.specular_exponent / 1000.0f;
    mat.roughness_clamp          = 0.25f;
    mat.roughness_as_smoothness  = content->args.legacy_smoothness;
    mat.emission_active          = has_luminance_tex || has_emission;
    mat.thin_walled              = false;
    mat.normal_map_is_compressed = true;
    mat.metallic                 = wavefront_mat.specular_reflectivity.r > 0.5f;
    mat.albedo_tex               = has_albedo_tex ? texture_offset + wavefront_mat.texture[WF_ALBEDO] : TEXTURE_NONE;
    mat.luminance_tex            = has_luminance_tex ? texture_offset + wavefront_mat.texture[WF_LUMINANCE] : TEXTURE_NONE;
    mat.roughness_tex            = has_roughness_tex ? texture_offset + wavefront_mat.texture[WF_ROUGHNESS] : TEXTURE_NONE;
    mat.metallic_tex             = has_metallic_tex ? texture_offset + wavefront_mat.texture[WF_METALLIC] : TEXTURE_NONE;
    mat.normal_tex               = has_normal_tex ? texture_offset + wavefront_mat.texture[WF_NORMAL] : TEXTURE_NONE;

    __FAILURE_HANDLE(array_push(materials, &mat));
  }

  return LUMINARY_SUCCESS;
}

static_assert(sizeof(WavefrontVertex) == 3 * sizeof(float), "Wavefront Vertex must be a struct of 3 floats!.");

LuminaryResult wavefront_convert_content(
  WavefrontContent* content, ARRAYPTR Mesh*** meshes, ARRAYPTR Texture*** textures, ARRAYPTR Material** materials,
  uint32_t material_offset) {
  __CHECK_NULL_ARGUMENT(content);
  __CHECK_NULL_ARGUMENT(meshes);
  __CHECK_NULL_ARGUMENT(textures);
  __CHECK_NULL_ARGUMENT(materials);

  if (content->state != WAVEFRONT_CONTENT_STATE_READY_TO_CONVERT) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Wavefront content was in an illegal state.");
  }

  content->state = WAVEFRONT_CONTENT_STATE_FINISHED;

  uint32_t num_objects;
  __FAILURE_HANDLE(array_get_num_elements(content->object_names, &num_objects));

  if (num_objects == 0) {
    warn_message("Wavefront file contained no objects.");
    return LUMINARY_SUCCESS;
  }

  uint32_t triangle_count;
  __FAILURE_HANDLE(array_get_num_elements(content->triangles, &triangle_count));

  uint32_t vertex_count;
  __FAILURE_HANDLE(array_get_num_elements(content->vertices, &vertex_count));

  uint32_t uv_count;
  __FAILURE_HANDLE(array_get_num_elements(content->uvs, &uv_count));

  uint32_t normal_count;
  __FAILURE_HANDLE(array_get_num_elements(content->normals, &normal_count));

  __FAILURE_HANDLE(_wavefront_convert_materials(content, materials, textures));

  Mesh* mesh;
  __FAILURE_HANDLE(mesh_create(&mesh));

  __FAILURE_HANDLE(mesh_set_name(mesh, content->object_names[0]));

  __FAILURE_HANDLE(host_malloc(&mesh->data.index_buffer, sizeof(uint32_t) * 4 * triangle_count));
  __FAILURE_HANDLE(host_malloc(&mesh->data.vertex_buffer, sizeof(float) * 4 * vertex_count));

  mesh->data.vertex_count = vertex_count;

  Vec128 min_bound = vec128_set_1(FLT_MAX);
  Vec128 max_bound = vec128_set_1(-FLT_MAX);

  for (uint32_t vertex_id = 0; vertex_id < vertex_count; vertex_id++) {
    WavefrontVertex vertex = content->vertices[vertex_id];

    mesh->data.vertex_buffer[vertex_id * 4 + 0] = vertex.x;
    mesh->data.vertex_buffer[vertex_id * 4 + 1] = vertex.y;
    mesh->data.vertex_buffer[vertex_id * 4 + 2] = vertex.z;
    mesh->data.vertex_buffer[vertex_id * 4 + 3] = 0.0f;

    Vec128 v = vec128_set(vertex.x, vertex.y, vertex.z, 0.0f);

    min_bound = vec128_min(min_bound, v);
    max_bound = vec128_min(max_bound, v);
  }

  mesh->bounding_box = (MeshBoundingBox) {.min = min_bound, .max = max_bound};

  __FAILURE_HANDLE(host_malloc(&mesh->triangles, sizeof(Triangle) * triangle_count));

  uint32_t ptr = 0;

  for (uint32_t tri_id = 0; tri_id < triangle_count; tri_id++) {
    WavefrontTriangle t = content->triangles[tri_id];
    Triangle triangle;

    WavefrontVertex v;

    const uint32_t v1_ptr = (t.v1 > 0) ? t.v1 - 1 : t.v1 + vertex_count;

    if (v1_ptr >= vertex_count) {
      continue;
    }
    else {
      v = content->vertices[v1_ptr];
    }

    triangle.vertex.x = v.x;
    triangle.vertex.y = v.y;
    triangle.vertex.z = v.z;

    const uint32_t v2_ptr = (t.v2 > 0) ? t.v2 - 1 : t.v2 + vertex_count;

    if (v2_ptr >= vertex_count) {
      continue;
    }
    else {
      v = content->vertices[v2_ptr];
    }

    triangle.edge1.x = v.x - triangle.vertex.x;
    triangle.edge1.y = v.y - triangle.vertex.y;
    triangle.edge1.z = v.z - triangle.vertex.z;

    const uint32_t v3_ptr = (t.v3 > 0) ? t.v3 - 1 : t.v3 + vertex_count;

    if (v3_ptr >= vertex_count) {
      continue;
    }
    else {
      v = content->vertices[v3_ptr];
    }

    triangle.edge2.x = v.x - triangle.vertex.x;
    triangle.edge2.y = v.y - triangle.vertex.y;
    triangle.edge2.z = v.z - triangle.vertex.z;

    if (
      fabsf(triangle.edge1.x) < FLT_EPSILON && fabsf(triangle.edge1.y) < FLT_EPSILON && fabsf(triangle.edge1.z) < FLT_EPSILON
      && fabsf(triangle.edge2.x) < FLT_EPSILON && fabsf(triangle.edge2.y) < FLT_EPSILON && fabsf(triangle.edge2.z) < FLT_EPSILON) {
      continue;
    }

    mesh->data.index_buffer[ptr * 4 + 0] = t.v1 - 1;
    mesh->data.index_buffer[ptr * 4 + 1] = t.v2 - 1;
    mesh->data.index_buffer[ptr * 4 + 2] = t.v3 - 1;

    // Some OBJs have no normals. We compute the face normal here and use that instead.
    // TODO: Implement smooth normals generation.
    vec3 face_n = (vec3) {.x = triangle.edge1.y * triangle.edge2.z - triangle.edge1.z * triangle.edge2.y,
                          .y = triangle.edge1.z * triangle.edge2.x - triangle.edge1.x * triangle.edge2.z,
                          .z = triangle.edge1.x * triangle.edge2.y - triangle.edge1.y * triangle.edge2.x};

    const float face_n_length = 1.0f / sqrtf(face_n.x * face_n.x + face_n.y * face_n.y + face_n.z * face_n.z);

    if (isnan(face_n_length) == false && isinf(face_n_length) == false) {
      face_n.x *= face_n_length;
      face_n.y *= face_n_length;
      face_n.z *= face_n_length;
    }

    WavefrontUV uv;

    const uint32_t vt1_ptr = (t.vt1 > 0) ? t.vt1 - 1 : t.vt1 + uv_count;

    if (vt1_ptr >= uv_count) {
      uv.u = 0.0f;
      uv.v = 0.0f;
    }
    else {
      uv = content->uvs[vt1_ptr];
    }

    triangle.vertex_texture.u = uv.u;
    triangle.vertex_texture.v = uv.v;

    const uint32_t vt2_ptr = (t.vt2 > 0) ? t.vt2 - 1 : t.vt2 + uv_count;

    if (vt2_ptr >= uv_count) {
      uv.u = 0.0f;
      uv.v = 0.0f;
    }
    else {
      uv = content->uvs[vt2_ptr];
    }

    triangle.edge1_texture.u = uv.u - triangle.vertex_texture.u;
    triangle.edge1_texture.v = uv.v - triangle.vertex_texture.v;

    const uint32_t vt3_ptr = (t.vt3 > 0) ? t.vt3 - 1 : t.vt3 + uv_count;

    if (vt3_ptr >= uv_count) {
      uv.u = 0.0f;
      uv.v = 0.0f;
    }
    else {
      uv = content->uvs[vt3_ptr];
    }

    triangle.edge2_texture.u = uv.u - triangle.vertex_texture.u;
    triangle.edge2_texture.v = uv.v - triangle.vertex_texture.v;

    WavefrontNormal n;

    const uint32_t vn1_ptr = (t.vn1 > 0) ? t.vn1 - 1 : t.vn1 + normal_count;

    if (vn1_ptr >= normal_count) {
      n = face_n;
    }
    else {
      n = content->normals[vn1_ptr];

      const float n_length = 1.0f / sqrtf(n.x * n.x + n.y * n.y + n.z * n.z);

      if (isnan(n_length) || isinf(n_length)) {
        n = face_n;
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

    const uint32_t vn2_ptr = (t.vn2 > 0) ? t.vn2 - 1 : t.vn2 + normal_count;

    if (vn2_ptr >= normal_count) {
      n = face_n;
    }
    else {
      n = content->normals[vn2_ptr];

      const float n_length = 1.0f / sqrtf(n.x * n.x + n.y * n.y + n.z * n.z);

      if (isnan(n_length) || isinf(n_length)) {
        n = face_n;
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

    const uint32_t vn3_ptr = (t.vn3 > 0) ? t.vn3 - 1 : t.vn3 + normal_count;

    if (vn3_ptr >= normal_count) {
      n = face_n;
    }
    else {
      n = content->normals[vn3_ptr];

      const float n_length = 1.0f / sqrtf(n.x * n.x + n.y * n.y + n.z * n.z);

      if (isnan(n_length) || isinf(n_length)) {
        n = face_n;
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

    triangle.material_id = material_offset + t.material;

    mesh->triangles[ptr++] = triangle;
  }

  const uint32_t new_triangle_count = ptr;

  __FAILURE_HANDLE(host_realloc(&mesh->triangles, sizeof(Triangle) * new_triangle_count));

  __FAILURE_HANDLE(host_realloc(&mesh->data.index_buffer, sizeof(uint32_t) * 4 * new_triangle_count));
  mesh->data.index_count    = 3 * new_triangle_count;
  mesh->data.triangle_count = new_triangle_count;

  __FAILURE_HANDLE(array_push(meshes, &mesh));

  return LUMINARY_SUCCESS;
}

LuminaryResult wavefront_arguments_get_default(WavefrontArguments* arguments) {
  __CHECK_NULL_ARGUMENT(arguments);

  arguments->legacy_smoothness         = false;
  arguments->force_transparency_cutout = false;
  arguments->emission_scale            = 1.0f;

  return LUMINARY_SUCCESS;
}
