#include "light.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "bench.h"
#include "error.h"
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
 * If texture is not RGB8 this will always return 1.
 */
static int contains_illumination(Triangle triangle, TextureRGBA tex) {
  if (tex.type != TexDataUINT8)
    return 1;

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
  float a0 = v0.u - m0 * v0.v;
  float a1 = v1.u - m1 * v1.v;
  float a2 = v2.u - m2 * v2.v;

  if (isinf(m0) || isnan(m0)) {
    m0 = m1;
    a0 = a1;
  }

  if (isinf(m1) || isnan(m1)) {
    m1 = m2;
    a1 = a2;
  }

  if (isinf(m2) || isnan(m2)) {
    m2 = m0;
    a2 = a0;
  }

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

  RGB8* ptr = (RGB8*) tex.data;

  for (int j = min_y; j <= max_y; j++) {
    const int coordy = j % tex.height;
    const float v    = (float) coordy;
    const float e0   = max(min(a0 + v * m0, max_e_0), min_e_0);
    const float e1   = max(min(a1 + v * m1, max_e_1), min_e_1);
    const float e2   = max(min(a2 + v * m2, max_e_2), min_e_2);
    const int min_x  = max(0, min_3_float_to_int(e0, e1, e2));
    const int max_x  = min(tex.width - 1, max_3_float_to_int(e0, e1, e2) + 1);

    for (int i = min_x; i <= max_x; i++) {
      const int coordx = i % tex.width;

      const RGB8 col = ptr[coordx + coordy * tex.width];

      if (col.r || col.g || col.b)
        return 1;
    }
  }

  return 0;
}

void process_lights(Scene* scene, TextureRGBA* textures) {
  bench_tic();

  Scene data = *scene;

  unsigned int lights_length = 16;
  TriangleLight* lights      = (TriangleLight*) malloc(sizeof(TriangleLight) * lights_length);
  unsigned int light_count   = 0;

  for (unsigned int i = 0; i < data.triangles_length; i++) {
    const Triangle triangle = data.triangles[i];

    const uint16_t tex_index = data.texture_assignments[triangle.object_maps].illuminance_map;

    if (tex_index != 0 && contains_illumination(triangle, textures[tex_index])) {
      const TriangleLight l      = {.vertex = triangle.vertex, .edge1 = triangle.edge1, .edge2 = triangle.edge2, .triangle_id = i};
      data.triangles[i].light_id = light_count;
      lights[light_count++]      = l;
      if (light_count == lights_length) {
        lights_length *= 2;
        lights = safe_realloc(lights, sizeof(TriangleLight) * lights_length);
      }
    }
  }

  lights = safe_realloc(lights, sizeof(TriangleLight) * light_count);

  data.triangle_lights        = lights;
  data.triangle_lights_length = light_count;

  *scene = data;

  bench_toc("Processing Lights");
}
