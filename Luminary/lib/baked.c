#include "baked.h"

#include <stdio.h>
#include <string.h>

#include "bench.h"
#include "buffer.h"
#include "config.h"
#include "error.h"
#include "raytrace.h"

#define head_size 0x68
#define magic 4919420911629456716ul
#define version LUMINARY_VERSION_HASH

/*
 * Format:
 *       | 0x00               | 0x08
 * ------+--------------------+----------------------
 *  0x00 | Magic              | Version
 *  0x10 | Instance           | Triangles
 *  0x20 | TraversalTriangles | Nodes
 *  0x30 | Lights             | TextureAssignments
 *  0x40 | TexAlb             | TexMat
 *  0x50 | TexIllum           | StringsCount
 *  0x60 | Strings            |
 * ------+--------------------+----------------------
 *
 * Texture Atlas Header:
 * 0x00 - Relative Offset (8 Bytes)
 * 0x08 - Width (4 Bytes)
 * 0x12 - Height (4 Bytes)
 *
 * Strings Header: (First string is output path, rest is mesh paths)
 * 0x00 - Relative Offset (8 Bytes)
 * 0x08 - Length (8 Bytes)
 *
 * First come all the headers, then the data.
 */

static TextureRGBA* load_textures(FILE* file, uint64_t count, uint64_t offset) {
  const uint64_t header_size = 16 * count;

  uint8_t* head = malloc(header_size);
  fseek(file, offset, SEEK_SET);
  fread_s(head, header_size, header_size, 1, file);

  TextureRGBA* textures = malloc(sizeof(TextureRGBA) * count);
  uint64_t data_offset;
  uint32_t width, height;
  uint64_t amount;
  uint64_t total_length = 0;

  for (uint64_t i = 0; i < count; i++) {
    memcpy(&data_offset, head + 16 * i, 8);
    memcpy(&width, head + 16 * i + 8, 4);
    memcpy(&height, head + 16 * i + 12, 4);

    textures[i].width  = width;
    textures[i].height = height;

    amount           = width * height;
    textures[i].data = (RGBAF*) (data_offset - header_size);
    total_length += amount;
  }

  RGBAF* data = malloc(sizeof(RGBAF) * total_length);

  fseek(file, offset + header_size, SEEK_SET);
  fread_s(data, sizeof(RGBAF) * total_length, sizeof(RGBAF) * total_length, 1, file);

  for (uint64_t i = 0; i < count; i++) {
    uint8_t* ptr = (uint8_t*) textures[i].data;
    ptr += (uint64_t) data;
    textures[i].data = (RGBAF*) ptr;
  }

  return textures;
}

static void free_textures(TextureRGBA* textures, uint64_t count) {
  free(textures[0].data);
  free(textures);
}

static char** load_strings(FILE* file, uint64_t count, uint64_t offset) {
  const uint64_t header_size = 16 * count;

  uint64_t* head = malloc(header_size);
  fseek(file, offset, SEEK_SET);
  fread_s(head, header_size, header_size, 1, file);

  char** strings = malloc(sizeof(char*) * count);

  uint64_t total_length = 0;

  for (uint64_t i = 0; i < count; i++) {
    strings[i] = (char*) (head[2 * i] - header_size);
    total_length += head[2 * i + 1];
  }

  char* data = malloc(total_length);
  fseek(file, offset + header_size, SEEK_SET);
  fread_s(data, total_length, total_length, 1, file);

  for (uint64_t i = 0; i < count; i++) {
    strings[i] += (uint64_t) data;
  }

  free(head);

  return strings;
}

RaytraceInstance* load_baked(const char* filename) {
  bench_tic();
  FILE* file;
  fopen_s(&file, filename, "rb");

  assert(file != 0, "Baked file could not be loaded!", 1);

  uint64_t* head = malloc(head_size);
  fread_s(head, head_size, head_size, 1, file);

  assert(head[0] == magic, "Specified file is not a Luminary Baked File!", 1);

  if (head[1] != version) {
    warn_message("Baked file is from a different version of Luminary.");
  }

  RaytraceInstance* instance = (RaytraceInstance*) malloc(sizeof(RaytraceInstance));

  fseek(file, head[2], SEEK_SET);
  fread_s(instance, sizeof(RaytraceInstance), sizeof(RaytraceInstance), 1, file);

  Scene scene = instance->scene_gpu;

  scene.triangles = malloc(sizeof(Triangle) * scene.triangles_length);
  fseek(file, head[3], SEEK_SET);
  fread_s(scene.triangles, sizeof(Triangle) * scene.triangles_length, sizeof(Triangle) * scene.triangles_length, 1, file);

  scene.traversal_triangles = malloc(sizeof(TraversalTriangle) * scene.triangles_length);
  fseek(file, head[4], SEEK_SET);
  fread_s(
    scene.traversal_triangles, sizeof(TraversalTriangle) * scene.triangles_length, sizeof(TraversalTriangle) * scene.triangles_length, 1,
    file);

  scene.nodes = malloc(sizeof(Node8) * scene.nodes_length);
  fseek(file, head[5], SEEK_SET);
  fread_s(scene.nodes, sizeof(Node8) * scene.nodes_length, sizeof(Node8) * scene.nodes_length, 1, file);

  scene.lights = malloc(sizeof(Light) * scene.lights_length);
  fseek(file, head[6], SEEK_SET);
  fread_s(scene.lights, sizeof(Light) * scene.lights_length, sizeof(Light) * scene.lights_length, 1, file);

  scene.texture_assignments = malloc(sizeof(TextureAssignment) * scene.materials_length);
  fseek(file, head[7], SEEK_SET);
  fread_s(
    scene.texture_assignments, sizeof(TextureAssignment) * scene.materials_length, sizeof(TextureAssignment) * scene.materials_length, 1,
    file);

  TextureRGBA* albedo_tex = load_textures(file, instance->albedo_atlas_length, head[8]);
  void* albedo_atlas      = initialize_textures(albedo_tex, instance->albedo_atlas_length);
  free_textures(albedo_tex, instance->albedo_atlas_length);
  TextureRGBA* illuminance_tex = load_textures(file, instance->illuminance_atlas_length, head[9]);
  void* illuminance_atlas      = initialize_textures(illuminance_tex, instance->illuminance_atlas_length);
  free_textures(illuminance_tex, instance->illuminance_atlas_length);
  TextureRGBA* material_tex = load_textures(file, instance->material_atlas_length, head[10]);
  void* material_atlas      = initialize_textures(material_tex, instance->material_atlas_length);
  free_textures(material_tex, instance->material_atlas_length);

  RaytraceInstance* final = init_raytracing(
    instance->settings, albedo_atlas, instance->albedo_atlas_length, illuminance_atlas, instance->illuminance_atlas_length, material_atlas,
    instance->material_atlas_length, scene, instance->default_material);

  uint64_t strings_count = head[11];
  char** strings         = load_strings(file, strings_count, head[12]);

  final->settings.output_path       = strings[0];
  final->settings.mesh_files        = strings + 1;
  final->settings.mesh_files_count  = strings_count - 1;
  final->settings.mesh_files_length = strings_count - 1;

  free(instance);
  free_scene(scene);
  free(head);

  bench_toc("Loading Luminary Baked File");

  return final;
}

#define CPU_PTR 0
#define GPU_PTR 1
#define TEX_PTR 2

static uint64_t write_data(FILE* file, uint64_t count, uint64_t size, void* data, int kind) {
  uint64_t address     = ftell(file);
  uint64_t data_length = count * size;
  if (kind == GPU_PTR) {
    data = memcpy_gpu_to_cpu(data, data_length);
  }
  else if (kind == TEX_PTR) {
    data = memcpy_texture_to_cpu(data, &data_length);
  }
  fwrite(data, data_length, 1, file);
  if (kind)
    free_host_memory(data);
  return address;
}

static uint64_t serialize_strings(RaytraceInstance* instance, void** ptr) {
  const uint64_t str_count   = 1 + instance->settings.mesh_files_count;
  const uint64_t header_size = 8 * 2 * str_count;

  uint64_t* header = malloc(header_size);

  uint64_t total_length = header_size;

  {
    const uint64_t output_len = strlen(instance->settings.output_path) + 1;
    header[1]                 = output_len;
    total_length += output_len;
  }

  for (uint64_t i = 0; i < instance->settings.mesh_files_count; i++) {
    const uint64_t mesh_len = strlen(instance->settings.mesh_files[i]) + 1;
    header[3 + 2 * i]       = mesh_len;
    total_length += mesh_len;
  }

  char* body = malloc(total_length);

  uint64_t offset = header_size;
  header[0]       = header_size;
  memcpy(body + offset, instance->settings.output_path, header[1]);
  offset += header[1];

  for (uint64_t i = 0; i < instance->settings.mesh_files_count; i++) {
    header[2 + 2 * i] = offset;
    memcpy(body + offset, instance->settings.mesh_files[i], header[3 + 2 * i]);
    offset += header[3 + 2 * i];
  }

  memcpy(body, header, header_size);

  free(header);

  *ptr = body;

  return total_length;
}

void serialize_baked(RaytraceInstance* instance) {
  bench_tic();
  FILE* file;
  fopen_s(&file, "generated.baked", "wb");

  fseek(file, head_size, SEEK_SET);

  uint64_t* head = (uint64_t*) malloc(head_size);

  head[0]  = magic;
  head[1]  = version;
  head[2]  = write_data(file, 1, sizeof(RaytraceInstance), instance, CPU_PTR);
  head[3]  = write_data(file, instance->scene_gpu.triangles_length, sizeof(Triangle), instance->scene_gpu.triangles, GPU_PTR);
  head[4]  = write_data(file, instance->scene_gpu.triangles_length, sizeof(TraversalTriangle), instance->scene_gpu.traversal_triangles, 1);
  head[5]  = write_data(file, instance->scene_gpu.nodes_length, sizeof(Node8), instance->scene_gpu.nodes, GPU_PTR);
  head[6]  = write_data(file, instance->scene_gpu.lights_length, sizeof(Light), instance->scene_gpu.lights, GPU_PTR);
  head[7]  = write_data(file, instance->scene_gpu.materials_length, sizeof(TextureAssignment), instance->scene_gpu.texture_assignments, 1);
  head[8]  = write_data(file, instance->albedo_atlas_length, 1, device_buffer_get_pointer(instance->albedo_atlas), TEX_PTR);
  head[9]  = write_data(file, instance->illuminance_atlas_length, 1, device_buffer_get_pointer(instance->illuminance_atlas), TEX_PTR);
  head[10] = write_data(file, instance->material_atlas_length, 1, device_buffer_get_pointer(instance->material_atlas), TEX_PTR);
  head[11] = 1 + instance->settings.mesh_files_count;

  void* strings;
  uint64_t strings_length = serialize_strings(instance, &strings);
  head[12]                = write_data(file, 1, strings_length, strings, CPU_PTR);
  free(strings);

  fseek(file, 0, SEEK_SET);

  fwrite(head, head_size, 1, file);

  free(head);
  fclose(file);

  bench_toc("Baked Luminary File");
}
