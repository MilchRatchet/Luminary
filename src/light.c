#include "light.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "bench.h"
#include "buffer.h"
#include "ceb.h"
#include "device.h"
#include "structs.h"
#include "texture.h"
#include "utils.h"

static int min_3_float_to_int(float a, float b, float c) {
  int ai = (int) floorf(a);
  int bi = (int) floorf(b);
  int ci = (int) floorf(c);

  int minab = (ai < bi) ? ai : bi;

  return (minab < ci) ? minab : ci;
}

static int max_3_float_to_int(float a, float b, float c) {
  int ai = (int) floorf(a);
  int bi = (int) floorf(b);
  int ci = (int) floorf(c);

  int maxab = (ai < bi) ? bi : ai;

  return (maxab < ci) ? ci : maxab;
}

/*
 * Returns 1 if triangle texture is non zero at some point, 0 else.
 * If texture is not RGB8 or not present on CPU this will always return 1.
 */
static int light_triangle_texture_has_emission(Triangle triangle, TextureRGBA tex) {
  // TODO: Implement support for other types of textures.
  if (tex.type != TexDataUINT8) {
    warn_message("Texture is not 8 bits. Assume that this triangle is a light.");
    return 1;
  }

  if (tex.num_components != 4) {
    warn_message("Texture does not have 4 channels. Assume that this triangle is a light.");
    return 1;
  }

  if (tex.storage != TexStorageCPU) {
    warn_message("Texture is allocated on the GPU. Assume that this triangle is a light.");
    return 1;
  }

  UV v0 = {.u = triangle.vertex_texture.u, .v = triangle.vertex_texture.v};
  UV v1 = {.u = triangle.vertex_texture.u + triangle.edge1_texture.u, .v = triangle.vertex_texture.v + triangle.edge1_texture.v};
  UV v2 = {.u = triangle.vertex_texture.u + triangle.edge2_texture.u, .v = triangle.vertex_texture.v + triangle.edge2_texture.v};

  v0.v = 1.0f - v0.v;
  v1.v = 1.0f - v1.v;
  v2.v = 1.0f - v2.v;

  v0.u *= tex.width;
  v0.v *= tex.height;
  v1.u *= tex.width;
  v1.v *= tex.height;
  v2.u *= tex.width;
  v2.v *= tex.height;

  const int min_y = min_3_float_to_int(v0.v, v1.v, v2.v);
  const int max_y = max_3_float_to_int(v0.v, v1.v, v2.v) + 1;

  float m0 = (v0.u - v1.u) / (v0.v - v1.v);
  float m1 = (v1.u - v2.u) / (v1.v - v2.v);
  float m2 = (v2.u - v0.u) / (v2.v - v0.v);

  if (isinf(m0) || isnan(m0)) {
    m0 = tex.width;
  }

  if (isinf(m1) || isnan(m1)) {
    m1 = tex.width;
  }

  if (isinf(m2) || isnan(m2)) {
    m2 = tex.width;
  }

  const float a0 = v0.u - m0 * v0.v;
  const float a1 = v1.u - m1 * v1.v;
  const float a2 = v2.u - m2 * v2.v;

  float min_e_0, max_e_0;

  {
    const float e_0_0 = a0 + v0.v * m0;
    const float e_0_1 = a0 + v1.v * m0;
    min_e_0           = min(e_0_0, e_0_1);
    max_e_0           = max(e_0_0, e_0_1);
  }

  float min_e_1, max_e_1;

  {
    const float e_1_1 = a1 + v1.v * m1;
    const float e_1_2 = a1 + v2.v * m1;
    min_e_1           = min(e_1_1, e_1_2);
    max_e_1           = max(e_1_1, e_1_2);
  }

  float min_e_2, max_e_2;

  {
    const float e_2_2 = a2 + v2.v * m2;
    const float e_2_0 = a2 + v0.v * m2;
    min_e_2           = min(e_2_2, e_2_0);
    max_e_2           = max(e_2_2, e_2_0);
  }

  const uint32_t* ptr = (uint32_t*) tex.data;

  for (int j = min_y; j <= max_y; j++) {
    const int coordy = j % tex.height;
    const float v    = (float) j;
    const float e0   = max(min(a0 + v * m0, max_e_0), min_e_0);
    const float e1   = max(min(a1 + v * m1, max_e_1), min_e_1);
    const float e2   = max(min(a2 + v * m2, max_e_2), min_e_2);
    const int min_x  = min_3_float_to_int(e0, e1, e2);
    const int max_x  = max_3_float_to_int(e0, e1, e2) + 1;

    for (int i = min_x; i <= max_x; i++) {
      const int coordx = i % tex.width;

      const uint32_t col = ptr[coordx + coordy * tex.width];

      if (col & 0xffffff00)
        return 1;
    }
  }

  return 0;
}

void lights_process(Scene* scene, int dmm_active) {
  bench_tic("Processing Lights");

  ////////////////////////////////////////////////////////////////////
  // Iterate over all triangles and find all light candidates.
  ////////////////////////////////////////////////////////////////////

  uint32_t candidate_lights_length = 16;
  uint32_t candidate_lights_count  = 0;
  TriangleLight* candidate_lights  = (TriangleLight*) malloc(sizeof(TriangleLight) * candidate_lights_length);

  uint32_t lights_length = 16;
  uint32_t lights_count  = 0;
  TriangleLight* lights  = (TriangleLight*) malloc(sizeof(TriangleLight) * candidate_lights_length);

  for (uint32_t i = 0; i < scene->triangle_data.triangle_count; i++) {
    const Triangle triangle = scene->triangles[i];

    const PackedMaterial material = scene->materials[triangle.material_id];
    const uint16_t tex_index      = material.luminance_map;

    // Triangles with displacement can't be light sources.
    if (dmm_active && scene->materials[triangle.material_id].normal_map)
      continue;

    const int is_textured_light = (tex_index != TEXTURE_NONE);

    // Triangle is a light if it has a light texture with non-zero value at some point on the triangle's surface or it
    // has no light texture but a non-zero constant emission.
    const int is_light = (is_textured_light) || material.emission_r || material.emission_g || material.emission_b;

    if (is_light) {
      TriangleLight light;

      light.vertex      = triangle.vertex;
      light.edge1       = triangle.edge1;
      light.edge2       = triangle.edge2;
      light.triangle_id = i;
      light.material_id = triangle.material_id;

      if (is_textured_light) {
        light.power = 0.0f;  // To be determined...

        candidate_lights[candidate_lights_count++] = light;
        if (candidate_lights_count == candidate_lights_length) {
          candidate_lights_length *= 2;
          candidate_lights = (TriangleLight*) safe_realloc(candidate_lights, sizeof(TriangleLight) * candidate_lights_length);
        }
      }
      else {
        const float luminance = 0.212655f * material.emission_r + 0.715158f * material.emission_g + 0.072187f * material.emission_b;

        light.power = luminance;

        scene->triangles[i].light_id = lights_count;

        lights[lights_count++] = light;
        if (lights_count == lights_length) {
          lights_length *= 2;
          lights = (TriangleLight*) safe_realloc(lights, sizeof(TriangleLight) * lights_length);
        }
      }
    }
  }

  log_message("Number of untextured lights: %u", lights_count);
  log_message("Number of textured candidate lights: %u", candidate_lights_count);

  candidate_lights = (TriangleLight*) safe_realloc(candidate_lights, sizeof(TriangleLight) * candidate_lights_count);

  ////////////////////////////////////////////////////////////////////
  // Iterate over all light candidates and compute their power.
  ////////////////////////////////////////////////////////////////////

  float* power_dst;
  device_malloc(&power_dst, sizeof(float) * candidate_lights_count);

  TriangleLight* device_candidate_lights;
  device_malloc(&device_candidate_lights, sizeof(TriangleLight) * candidate_lights_count);
  device_upload(device_candidate_lights, candidate_lights, sizeof(TriangleLight) * candidate_lights_count);

  lights_compute_power_host(device_candidate_lights, candidate_lights_count, power_dst);

  float* power = (float*) malloc(sizeof(float) * candidate_lights_count);

  device_download(power, power_dst, sizeof(float) * candidate_lights_count);

  device_free(power_dst, sizeof(float) * candidate_lights_count);
  device_free(device_candidate_lights, sizeof(TriangleLight) * candidate_lights_count);

  float max_power = 0.0f;

  for (uint32_t i = 0; i < candidate_lights_count; i++) {
    const float candidate_light_power = power[i];

    if (candidate_light_power > 1e-6f) {
      TriangleLight light = candidate_lights[i];

      max_power = max(max_power, candidate_light_power);

      light.power = candidate_light_power;

      scene->triangles[light.triangle_id].light_id = lights_count;

      lights[lights_count++] = light;
      if (lights_count == lights_length) {
        lights_length *= 2;
        lights = (TriangleLight*) safe_realloc(lights, sizeof(TriangleLight) * lights_length);
      }
    }
  }

  free(power);
  free(candidate_lights);

  log_message("Number of textured lights: %u", lights_count);
  log_message("Highest encountered light power: %f", max_power);

  lights = (TriangleLight*) safe_realloc(lights, sizeof(TriangleLight) * lights_count);

  scene->triangle_lights       = lights;
  scene->triangle_lights_count = lights_count;

  ////////////////////////////////////////////////////////////////////
  // Create vertex and index buffer for BVH creation.
  ////////////////////////////////////////////////////////////////////

  TriangleGeomData tri_data;

  tri_data.vertex_count   = lights_count * 3;
  tri_data.index_count    = lights_count * 3;
  tri_data.triangle_count = lights_count;

  const size_t vertex_buffer_size = sizeof(float) * 4 * 3 * lights_count;
  const size_t index_buffer_size  = sizeof(uint32_t) * 4 * lights_count;

  float* vertex_buffer   = (float*) malloc(vertex_buffer_size);
  uint32_t* index_buffer = (uint32_t*) malloc(index_buffer_size);

  for (uint32_t i = 0; i < lights_count; i++) {
    const TriangleLight l = lights[i];

    vertex_buffer[3 * 4 * i + 4 * 0 + 0] = l.vertex.x;
    vertex_buffer[3 * 4 * i + 4 * 0 + 1] = l.vertex.y;
    vertex_buffer[3 * 4 * i + 4 * 0 + 2] = l.vertex.z;
    vertex_buffer[3 * 4 * i + 4 * 0 + 3] = 1.0f;
    vertex_buffer[3 * 4 * i + 4 * 1 + 0] = l.vertex.x + l.edge1.x;
    vertex_buffer[3 * 4 * i + 4 * 1 + 1] = l.vertex.y + l.edge1.y;
    vertex_buffer[3 * 4 * i + 4 * 1 + 2] = l.vertex.z + l.edge1.z;
    vertex_buffer[3 * 4 * i + 4 * 1 + 3] = 1.0f;
    vertex_buffer[3 * 4 * i + 4 * 2 + 0] = l.vertex.x + l.edge2.x;
    vertex_buffer[3 * 4 * i + 4 * 2 + 1] = l.vertex.y + l.edge2.y;
    vertex_buffer[3 * 4 * i + 4 * 2 + 2] = l.vertex.z + l.edge2.z;
    vertex_buffer[3 * 4 * i + 4 * 2 + 3] = 1.0f;

    index_buffer[4 * i + 0] = 3 * i + 0;
    index_buffer[4 * i + 1] = 3 * i + 1;
    index_buffer[4 * i + 2] = 3 * i + 2;
    index_buffer[4 * i + 3] = 0;
  }

  device_malloc(&tri_data.vertex_buffer, vertex_buffer_size);
  device_upload(tri_data.vertex_buffer, vertex_buffer, vertex_buffer_size);

  device_malloc(&tri_data.index_buffer, index_buffer_size);
  device_upload(tri_data.index_buffer, index_buffer, index_buffer_size);

  scene->triangle_lights_data = tri_data;

  bench_toc();
}
