#include "baked.h"

#include <stdio.h>
#include <string.h>

#include "bench.h"
#include "raytrace.h"

#define head_size 0x80
#define magic 4919420911629456716ul
#define version 22092021  // we literally just use the date of the last change as the version

/*
 * Format:
 *       | 0x00               | 0x08
 * ------+--------------------+----------------------
 *  0x00 | Magic              | Version
 *  0x10 | Instance           | Triangles
 *  0x20 | TraversalTriangles | Nodes
 *  0x30 | Lights             | TextureAssignments
 *  0x40 | TexAlbLength       | TexAlb
 *  0x50 | TexMatLength       | TexMat
 *  0x60 | TexIllumLength     | TexIllum
 *  0x70 | StringsCount       | Strings
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

Scene load_baked(const char* filename, RaytraceInstance** instance);

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
  head[2]  = write_data(file, 1, sizeof(RaytraceInstance), instance, 0);
  head[3]  = write_data(file, instance->scene_gpu.triangles_length, sizeof(Triangle), instance->scene_gpu.triangles, 1);
  head[4]  = write_data(file, instance->scene_gpu.triangles_length, sizeof(TraversalTriangle), instance->scene_gpu.traversal_triangles, 1);
  head[5]  = write_data(file, instance->scene_gpu.nodes_length, sizeof(Node8), instance->scene_gpu.nodes, 1);
  head[6]  = write_data(file, instance->scene_gpu.lights_length, sizeof(Light), instance->scene_gpu.lights, 1);
  head[7]  = write_data(file, instance->scene_gpu.materials_length, sizeof(TextureAssignment), instance->scene_gpu.texture_assignments, 1);
  head[8]  = instance->albedo_atlas_length;
  head[9]  = write_data(file, head[8], 1, instance->albedo_atlas, TEX_PTR);
  head[10] = instance->material_atlas_length;
  head[11] = write_data(file, head[10], 1, instance->material_atlas, TEX_PTR);
  head[12] = instance->illuminance_atlas_length;
  head[13] = write_data(file, head[12], 1, instance->illuminance_atlas, TEX_PTR);
  head[14] = 1 + instance->settings.mesh_files_count;

  void* strings;
  uint64_t strings_length = serialize_strings(instance, &strings);
  head[15]                = write_data(file, 1, strings_length, strings, CPU_PTR);
  free(strings);

  fseek(file, 0, SEEK_SET);

  fwrite(head, head_size, 1, file);

  free(head);
  fclose(file);

  bench_toc("Baked Luminary File");
}
