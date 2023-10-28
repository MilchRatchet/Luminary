#include "baked.h"

#include <stdio.h>
#include <string.h>

#include "bench.h"
#include "buffer.h"
#include "config.h"
#include "qoi.h"
#include "raytrace.h"
#include "stars.h"
#include "structs.h"
#include "texture.h"
#include "utils.h"

////////////////////////////////////////////////////////////////////
// Definition
////////////////////////////////////////////////////////////////////

#define head_size 0x70
#define magic 4919420911629456716ul
#define version LUMINARY_VERSION_HASH

/*
 * Format:
 *       | 0x00               | 0x08
 * ------+--------------------+----------------------
 *  0x00 | Magic              | Version
 *  0x10 | Instance           | Triangles
 *  0x20 | Nodes              | Lights
 *  0x30 | TextureAssignments | TexAlb
 *  0x40 | TexMat             | TexIllum
 *  0x50 | TexNormal          | NULL
 *  0x60 | StringsCount       | Strings
 * ------+--------------------+----------------------
 *
 * Texture Atlas Header (12 Bytes):
 * 0x00 - Size in bytes (4 bytes)
 * 0x04 - Width (4 bytes)
 * 0x08 - Height (4 bytes)
 *
 * Strings Header: (First string is output path, rest is mesh paths)
 * 0x00 - Relative Offset (8 Bytes)
 * 0x08 - Length (8 Bytes)
 *
 * First come all the headers, then the data.
 */

////////////////////////////////////////////////////////////////////
// GPU-CPU functions
////////////////////////////////////////////////////////////////////

static void* memcpy_gpu_to_cpu(void* gpu_ptr, size_t size) {
  void* cpu_ptr;
  gpuErrchk(cudaMallocHost((void**) &(cpu_ptr), size));
  gpuErrchk(cudaMemcpy(cpu_ptr, gpu_ptr, size, cudaMemcpyDeviceToHost));
  return cpu_ptr;
}

static void* memcpy_texture_to_cpu(void* textures_ptr, uint64_t* count) {
  const uint64_t tex_count           = *count;
  const uint64_t header_element_size = 12;
  const uint64_t header_size         = header_element_size * tex_count;

  cudaTextureObject_t* tex_objects;
  gpuErrchk(cudaMallocHost((void**) &(tex_objects), sizeof(cudaTextureObject_t) * tex_count));
  gpuErrchk(cudaMemcpy(tex_objects, textures_ptr, sizeof(cudaTextureObject_t) * tex_count, cudaMemcpyDeviceToHost));

  struct cudaResourceDesc resource;
  uint64_t buffer_size = header_size;

  for (uint64_t i = 0; i < tex_count; i++) {
    cudaGetTextureObjectResourceDesc(&resource, tex_objects[i]);
    uint64_t pitch  = (uint64_t) resource.res.pitch2D.pitchInBytes;
    uint64_t height = (uint64_t) resource.res.pitch2D.height;
    buffer_size += (pitch + 32) * height;
  }

  uint8_t* cpu_ptr;
  gpuErrchk(cudaMallocHost((void**) &(cpu_ptr), buffer_size));

  uint64_t offset = header_size;

  for (uint64_t i = 0; i < tex_count; i++) {
    cudaGetTextureObjectResourceDesc(&resource, tex_objects[i]);
    size_t pitch    = resource.res.pitch2D.pitchInBytes;
    uint32_t width  = (uint32_t) resource.res.pitch2D.width;
    uint32_t height = (uint32_t) resource.res.pitch2D.height;
    void* source    = resource.res.pitch2D.devPtr;
    gpuErrchk(cudaMemcpy(cpu_ptr + offset, source, pitch * height, cudaMemcpyDeviceToHost));

    TextureRGBA tex;
    texture_create(&tex, width, height, 1, pitch / sizeof(RGBA8), (void*) (cpu_ptr + offset), TexDataUINT8, TexStorageCPU);

    uint32_t encoded_size;

    void* encoded_data = qoi_encode_RGBA8(&tex, (int*) &encoded_size);

    memcpy(cpu_ptr + header_element_size * i + 0x00, &encoded_size, sizeof(uint32_t));
    memcpy(cpu_ptr + header_element_size * i + 0x04, &width, sizeof(uint32_t));
    memcpy(cpu_ptr + header_element_size * i + 0x08, &height, sizeof(uint32_t));

    memcpy(cpu_ptr + offset, encoded_data, encoded_size);
    offset += encoded_size;

    free(encoded_data);
  }

  buffer_size = offset;

  gpuErrchk(cudaFreeHost(tex_objects));

  *count = buffer_size;

  return cpu_ptr;
}

static void free_host_memory(void* ptr) {
  gpuErrchk(cudaFreeHost(ptr));
}

////////////////////////////////////////////////////////////////////
// Baked file format handlers
////////////////////////////////////////////////////////////////////

static TextureRGBA* load_textures(FILE* file, uint64_t count, uint64_t offset) {
  const uint32_t header_element_size = 12;
  const uint64_t header_size         = header_element_size * count;

  uint8_t* head = malloc(header_size);
  fseek(file, offset, SEEK_SET);
  fread(head, header_size, 1, file);

  TextureRGBA* textures = malloc(sizeof(TextureRGBA) * count);
  uint32_t width, height, data_size;
  uint64_t total_length = 0;

  for (uint64_t i = 0; i < count; i++) {
    memcpy(&data_size, head + header_element_size * i + 0x00, sizeof(uint32_t));
    memcpy(&width, head + header_element_size * i + 0x04, sizeof(uint32_t));
    memcpy(&height, head + header_element_size * i + 0x08, sizeof(uint32_t));

    texture_create(textures + i, width, height, 1, width, (void*) ((uint64_t) (data_size)), TexDataUINT8, TexStorageCPU);

    textures[i].data = (void*) ((uint64_t) (data_size));
    total_length += data_size;
  }

  uint8_t* encoded_data = malloc(total_length);

  fseek(file, offset + header_size, SEEK_SET);
  fread(encoded_data, total_length, 1, file);

  size_t encoded_offset = 0;

  for (uint64_t i = 0; i < count; i++) {
    const uint64_t size = (uint64_t) textures[i].data;
    TextureRGBA* tex    = qoi_decode_RGBA8((void*) (encoded_data + encoded_offset), size);
    textures[i].data    = tex->data;
    textures[i].pitch   = tex->width;
    free(tex);

    encoded_offset += size;
  }

  return textures;
}

static void free_textures(TextureRGBA* textures, uint64_t count) {
  if (count == 0)
    return;

  // All textures come from the same malloced buffer so only the first one needs to get freed
  free(textures[0].data);
  free(textures);
}

static char** load_strings(FILE* file, uint64_t count, uint64_t offset) {
  const uint64_t header_size = 16 * count;

  uint64_t* head = malloc(header_size);
  fseek(file, offset, SEEK_SET);
  fread(head, header_size, 1, file);

  char** strings = malloc(sizeof(char*) * count);

  uint64_t total_length = 0;

  for (uint64_t i = 0; i < count; i++) {
    strings[i] = (char*) (head[2 * i] - header_size);
    total_length += head[2 * i + 1];
  }

  char* data = malloc(total_length);
  fseek(file, offset + header_size, SEEK_SET);
  fread(data, total_length, 1, file);

  for (uint64_t i = 0; i < count; i++) {
    strings[i] += (uint64_t) data;
  }

  free(head);

  return strings;
}

static TraversalTriangle* construct_traversal_triangles(
  const Triangle* triangles, const unsigned int triangles_length, const PackedMaterial* materials) {
  TraversalTriangle* traversal_triangles = malloc(sizeof(TraversalTriangle) * triangles_length);

  for (unsigned int i = 0; i < triangles_length; i++) {
    const Triangle triangle   = triangles[i];
    const uint32_t albedo_tex = materials[triangle.material_id].albedo_map;
    TraversalTriangle tt      = {
           .vertex     = {.x = triangle.vertex.x, .y = triangle.vertex.y, .z = triangle.vertex.z},
           .edge1      = {.x = triangle.edge1.x, .y = triangle.edge1.y, .z = triangle.edge1.z},
           .edge2      = {.x = triangle.edge2.x, .y = triangle.edge2.y, .z = triangle.edge2.z},
           .albedo_tex = albedo_tex};
    traversal_triangles[i] = tt;
  }

  return traversal_triangles;
}

RaytraceInstance* load_baked(const char* filename) {
  bench_tic("Loading Luminary Baked File");
  FILE* file = fopen(filename, "rb");

  assert(file != 0, "Baked file could not be loaded!", 1);

  uint64_t* head = malloc(head_size);
  fread(head, head_size, 1, file);

  assert(head[0] == magic, "Specified file is not a Luminary Baked File!", 1);

  if (head[1] != version) {
    warn_message("Baked file is from a different version of Luminary.");
  }

  RaytraceInstance* instance = (RaytraceInstance*) malloc(sizeof(RaytraceInstance));

  fseek(file, head[2], SEEK_SET);
  fread(instance, sizeof(RaytraceInstance), 1, file);

  Scene* scene;
  scene_init(&scene);

  memcpy(scene, &instance->scene, sizeof(Scene));

  scene->triangles = malloc(sizeof(Triangle) * scene->triangle_data.triangle_count);
  fseek(file, head[3], SEEK_SET);
  fread(scene->triangles, sizeof(Triangle) * scene->triangle_data.triangle_count, 1, file);

  scene->triangle_lights = malloc(sizeof(TriangleLight) * scene->triangle_lights_count);
  fseek(file, head[5], SEEK_SET);
  fread(scene->triangle_lights, sizeof(TriangleLight) * scene->triangle_lights_count, 1, file);

  scene->materials = malloc(sizeof(PackedMaterial) * scene->materials_count);
  fseek(file, head[6], SEEK_SET);
  fread(scene->materials, sizeof(PackedMaterial) * scene->materials_count, 1, file);

  TextureRGBA* albedo_tex      = load_textures(file, instance->tex_atlas.albedo_length, head[7]);
  TextureRGBA* illuminance_tex = load_textures(file, instance->tex_atlas.illuminance_length, head[8]);
  TextureRGBA* material_tex    = load_textures(file, instance->tex_atlas.material_length, head[9]);
  TextureRGBA* normal_tex      = load_textures(file, instance->tex_atlas.normal_length, head[10]);

  TextureAtlas tex_atlas = {
    .albedo             = (DeviceBuffer*) 0,
    .albedo_length      = instance->tex_atlas.albedo_length,
    .illuminance        = (DeviceBuffer*) 0,
    .illuminance_length = instance->tex_atlas.illuminance_length,
    .material           = (DeviceBuffer*) 0,
    .material_length    = instance->tex_atlas.material_length,
    .normal             = (DeviceBuffer*) 0,
    .normal_length      = instance->tex_atlas.normal_length};

  texture_create_atlas(&tex_atlas.albedo, albedo_tex, instance->tex_atlas.albedo_length);
  texture_create_atlas(&tex_atlas.illuminance, illuminance_tex, instance->tex_atlas.illuminance_length);
  texture_create_atlas(&tex_atlas.material, material_tex, instance->tex_atlas.material_length);
  texture_create_atlas(&tex_atlas.normal, normal_tex, instance->tex_atlas.normal_length);

  free_textures(albedo_tex, instance->tex_atlas.albedo_length);
  free_textures(illuminance_tex, instance->tex_atlas.illuminance_length);
  free_textures(material_tex, instance->tex_atlas.material_length);
  free_textures(normal_tex, instance->tex_atlas.normal_length);

  // scene->traversal_triangles = construct_traversal_triangles(scene->triangles, scene->triangles_length, scene->materials);

  scene->sky.stars = (Star*) 0;

  RaytraceInstance* final;
  raytrace_init(&final, instance->settings, tex_atlas, scene);

  final->scene.sky.cloud.initialized = 0;

  uint64_t strings_count = head[12];
  char** strings         = load_strings(file, strings_count, head[13]);

  final->settings.output_path       = strings[0];
  final->settings.mesh_files        = strings + 1;
  final->settings.mesh_files_count  = strings_count - 1;
  final->settings.mesh_files_length = strings_count - 1;

  free(instance);
  scene_clear(&scene);
  free(head);
  fclose(file);

  bench_toc();

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

  for (int i = 0; i < instance->settings.mesh_files_count; i++) {
    const uint64_t mesh_len = strlen(instance->settings.mesh_files[i]) + 1;
    header[3 + 2 * i]       = mesh_len;
    total_length += mesh_len;
  }

  char* body = malloc(total_length);

  uint64_t offset = header_size;
  header[0]       = header_size;
  memcpy(body + offset, instance->settings.output_path, header[1]);
  offset += header[1];

  for (int i = 0; i < instance->settings.mesh_files_count; i++) {
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
  bench_tic("Baked Luminary File");
  FILE* file = fopen("generated.baked", "wb");

  if (!file) {
    error_message("Failed to write baked file. generated.baked could not be opened.");
    return;
  }

  fseek(file, head_size, SEEK_SET);

  uint64_t* head = (uint64_t*) malloc(head_size);

  head[0] = magic;
  head[1] = version;
  head[2] = write_data(file, 1, sizeof(RaytraceInstance), instance, CPU_PTR);
  head[3] = write_data(file, instance->scene.triangle_data.triangle_count, sizeof(Triangle), instance->scene.triangles, GPU_PTR);
  head[5] = write_data(file, instance->scene.triangle_lights_count, sizeof(TriangleLight), instance->scene.triangle_lights, GPU_PTR);
  head[6] = write_data(file, instance->scene.materials_count, sizeof(PackedMaterial), instance->scene.materials, 1);
  head[7] = write_data(file, instance->tex_atlas.albedo_length, 1, device_buffer_get_pointer(instance->tex_atlas.albedo), TEX_PTR);
  head[8] =
    write_data(file, instance->tex_atlas.illuminance_length, 1, device_buffer_get_pointer(instance->tex_atlas.illuminance), TEX_PTR);
  head[9]  = write_data(file, instance->tex_atlas.material_length, 1, device_buffer_get_pointer(instance->tex_atlas.material), TEX_PTR);
  head[10] = write_data(file, instance->tex_atlas.normal_length, 1, device_buffer_get_pointer(instance->tex_atlas.normal), TEX_PTR);
  head[12] = 1 + instance->settings.mesh_files_count;

  void* strings;
  uint64_t strings_length = serialize_strings(instance, &strings);
  head[13]                = write_data(file, 1, strings_length, strings, CPU_PTR);
  free(strings);

  fseek(file, 0, SEEK_SET);

  fwrite(head, head_size, 1, file);

  free(head);
  fclose(file);

  bench_toc();
}
