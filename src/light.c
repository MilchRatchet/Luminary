#include "light.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "bench.h"
#include "buffer.h"
#include "ceb.h"
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

void lights_process(Scene* scene, TextureRGBA* textures, int dmm_active) {
  bench_tic("Processing Lights");

  ////////////////////////////////////////////////////////////////////
  // Iterate over all triangles and find all light candidates.
  ////////////////////////////////////////////////////////////////////

  uint32_t lights_length = 16;
  TriangleLight* lights  = (TriangleLight*) malloc(sizeof(TriangleLight) * lights_length);
  uint32_t light_count   = 0;

  for (uint32_t i = 0; i < scene->triangle_data.triangle_count; i++) {
    const Triangle triangle = scene->triangles[i];

    const PackedMaterial material = scene->materials[triangle.material_id];
    const uint16_t tex_index      = material.luminance_map;

    // Triangles with displacement can't be light sources.
    if (dmm_active && scene->materials[triangle.material_id].normal_map)
      continue;

    int is_light = 0;

    // Triangle is a light if it has a light texture with non-zero value at some point on the triangle's surface or it
    // has no light texture but a non-zero constant emission.
    is_light |= tex_index != TEXTURE_NONE;
    is_light |= (tex_index == TEXTURE_NONE && (material.emission_r || material.emission_g || material.emission_b));

    if (is_light) {
      TriangleLight light;

      light.vertex      = triangle.vertex;
      light.edge1       = triangle.edge1;
      light.edge2       = triangle.edge2;
      light.triangle_id = i;
      light.material_id = triangle.material_id;
      light.power       = 0.0f;  // To be determined...

      // TODO: Do this only once we have our definitive list of lights.
      scene->triangles[i].light_id = light_count;

      lights[light_count++] = light;
      if (light_count == lights_length) {
        lights_length *= 2;
        lights = (TriangleLight*) safe_realloc(lights, sizeof(TriangleLight) * lights_length);
      }
    }
  }

  lights = (TriangleLight*) safe_realloc(lights, sizeof(TriangleLight) * light_count);

  ////////////////////////////////////////////////////////////////////
  // Iterate over all light candidates and compute their power.
  ////////////////////////////////////////////////////////////////////

  // TODO: Compute power using a kernel that has 32 threads per triangle allocated for integration.

  for (uint32_t i = 0; i < light_count; i++) {
    // TODO: Collect all light candidates with non-zero power.
  }

  scene->triangle_lights       = lights;
  scene->triangle_lights_count = light_count;

  ////////////////////////////////////////////////////////////////////
  // Create vertex and index buffer for BVH creation.
  ////////////////////////////////////////////////////////////////////

  TriangleGeomData tri_data;

  tri_data.vertex_count   = light_count * 3;
  tri_data.index_count    = light_count * 3;
  tri_data.triangle_count = light_count;

  const size_t vertex_buffer_size = sizeof(float) * 4 * 3 * light_count;
  const size_t index_buffer_size  = sizeof(uint32_t) * 4 * light_count;

  float* vertex_buffer   = (float*) malloc(vertex_buffer_size);
  uint32_t* index_buffer = (uint32_t*) malloc(index_buffer_size);

  for (uint32_t i = 0; i < light_count; i++) {
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

void lights_build_set_from_triangles(Scene* scene, TextureRGBA* textures, int dmm_active) {
  bench_tic("Processing Lights");

  ////////////////////////////////////////////////////////////////////
  // Iterate over all triangles and find all emissive ones.
  ////////////////////////////////////////////////////////////////////

  uint32_t lights_length = 16;
  TriangleLight* lights  = (TriangleLight*) malloc(sizeof(TriangleLight) * lights_length);
  uint32_t light_count   = 0;

  for (uint32_t i = 0; i < scene->triangle_data.triangle_count; i++) {
    const Triangle triangle = scene->triangles[i];

    const PackedMaterial material = scene->materials[triangle.material_id];
    const uint16_t tex_index      = material.luminance_map;

    // Triangles with displacement can't be light sources.
    if (dmm_active && scene->materials[triangle.material_id].normal_map)
      continue;

    int is_light = 0;

    // Triangle is a light if it has a light texture with non-zero value at some point on the triangle's surface or it
    // has no light texture but a non-zero constant emission.
    is_light |= (tex_index != TEXTURE_NONE && light_triangle_texture_has_emission(triangle, textures[tex_index]));
    is_light |= (tex_index == TEXTURE_NONE && (material.emission_r || material.emission_g || material.emission_b));

    if (is_light) {
      const TriangleLight l = {
        .vertex = triangle.vertex, .edge1 = triangle.edge1, .edge2 = triangle.edge2, .triangle_id = i, .material_id = triangle.material_id};
      scene->triangles[i].light_id = light_count;
      lights[light_count++]        = l;
      if (light_count == lights_length) {
        lights_length *= 2;
        lights = safe_realloc(lights, sizeof(TriangleLight) * lights_length);
      }
    }
  }

  lights = safe_realloc(lights, sizeof(TriangleLight) * light_count);

  scene->triangle_lights       = lights;
  scene->triangle_lights_count = light_count;

  ////////////////////////////////////////////////////////////////////
  // Create vertex and index buffer for BVH creation.
  ////////////////////////////////////////////////////////////////////

  TriangleGeomData tri_data;

  tri_data.vertex_count   = light_count * 3;
  tri_data.index_count    = light_count * 3;
  tri_data.triangle_count = light_count;

  const size_t vertex_buffer_size = sizeof(float) * 4 * 3 * light_count;
  const size_t index_buffer_size  = sizeof(uint32_t) * 4 * light_count;

  float* vertex_buffer   = (float*) malloc(vertex_buffer_size);
  uint32_t* index_buffer = (uint32_t*) malloc(index_buffer_size);

  for (uint32_t i = 0; i < light_count; i++) {
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
