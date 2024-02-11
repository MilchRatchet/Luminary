#ifndef CU_CLOUD_NOISE_H
#define CU_CLOUD_NOISE_H

#include <cuda_runtime_api.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>

#include "bench.h"
#include "buffer.h"
#include "log.h"
#include "math.cuh"
#include "png.h"
#include "raytrace.h"
#include "texture.h"

/*
 * This file contains code to generate noise textures needed for the clouds.
 * The code of this file is based on the cloud generation in https://github.com/turanszkij/WickedEngine.
 */

/*
 * Cubic interpolation with second derivative equal to 0 on boundaries
 */
LUM_DEVICE_FUNC float interp_cubic_d2(const float x) {
  return x * x * x * (x * (x * 6.0f - 15.0f) + 10.0f);
}

LUM_DEVICE_FUNC void perlin_hash(
  vec3 grid, float scale, bool tile, float4& low0, float4& low1, float4& low2, float4& high0, float4& high1, float4& high2) {
  const float2 offset = make_float2(50.0f, 161.0f);
  const float domain  = 69.0f;
  const float3 largef = make_float3(635.298681f, 682.357502f, 668.926525f);
  const float3 z_inc  = make_float3(48.500388f, 65.294118f, 63.934599f);

  grid.x -= floorf(grid.x / domain) * domain;
  grid.y -= floorf(grid.y / domain) * domain;
  grid.z -= floorf(grid.z / domain) * domain;

  float d          = domain - 1.5f;
  float3 grid_inc1 = make_float3(step(grid.x, d) * (grid.x + 1.0f), step(grid.y, d) * (grid.y + 1.0f), step(grid.z, d) * (grid.z + 1.0f));

  grid_inc1.x = (tile) ? fmodf(grid_inc1.x, scale) : grid_inc1.x;
  grid_inc1.y = (tile) ? fmodf(grid_inc1.y, scale) : grid_inc1.y;
  grid_inc1.z = (tile) ? fmodf(grid_inc1.z, scale) : grid_inc1.z;

  float4 p = make_float4(grid.x + offset.x, grid.y + offset.y, grid_inc1.x + offset.x, grid_inc1.y + offset.y);

  p.x *= p.x;
  p.y *= p.y;
  p.z *= p.z;
  p.w *= p.w;

  p = make_float4(p.x * p.y, p.z * p.y, p.x * p.w, p.z * p.w);

  float3 low =
    make_float3(1.0f / (largef.x + grid.z * z_inc.x), 1.0f / (largef.y + grid.z * z_inc.y), 1.0f / (largef.z + grid.z * z_inc.z));
  float3 high = make_float3(
    1.0f / (largef.x + grid_inc1.z * z_inc.x), 1.0f / (largef.y + grid_inc1.z * z_inc.y), 1.0f / (largef.z + grid_inc1.z * z_inc.z));

  low0  = make_float4(fractf(p.x * low.x), fractf(p.y * low.x), fractf(p.z * low.x), fractf(p.w * low.x));
  low1  = make_float4(fractf(p.x * low.y), fractf(p.y * low.y), fractf(p.z * low.y), fractf(p.w * low.y));
  low2  = make_float4(fractf(p.x * low.z), fractf(p.y * low.z), fractf(p.z * low.z), fractf(p.w * low.z));
  high0 = make_float4(fractf(p.x * high.x), fractf(p.y * high.x), fractf(p.z * high.x), fractf(p.w * high.x));
  high1 = make_float4(fractf(p.x * high.y), fractf(p.y * high.y), fractf(p.z * high.y), fractf(p.w * high.y));
  high2 = make_float4(fractf(p.x * high.z), fractf(p.y * high.z), fractf(p.z * high.z), fractf(p.w * high.z));
}

LUM_DEVICE_FUNC float perlin(vec3 p, const float scale, const bool tile) {
  p = scale_vector(p, scale);

  vec3 p1 = get_vector(floorf(p.x), floorf(p.y), floorf(p.z));
  vec3 p2 = p1;

  vec3 pf      = sub_vector(p, p1);
  vec3 pf_min1 = add_vector_const(pf, -1.0f);

  float4 hash_x0, hash_y0, hash_z0, hash_x1, hash_y1, hash_z1;
  perlin_hash(p2, scale, tile, hash_x0, hash_y0, hash_z0, hash_x1, hash_y1, hash_z1);

  float4 grad_x0 = make_float4(hash_x0.x - 0.49999f, hash_x0.y - 0.49999f, hash_x0.z - 0.49999f, hash_x0.w - 0.49999f);
  float4 grad_y0 = make_float4(hash_y0.x - 0.49999f, hash_y0.y - 0.49999f, hash_y0.z - 0.49999f, hash_y0.w - 0.49999f);
  float4 grad_z0 = make_float4(hash_z0.x - 0.49999f, hash_z0.y - 0.49999f, hash_z0.z - 0.49999f, hash_z0.w - 0.49999f);
  float4 grad_x1 = make_float4(hash_x1.x - 0.49999f, hash_x1.y - 0.49999f, hash_x1.z - 0.49999f, hash_x1.w - 0.49999f);
  float4 grad_y1 = make_float4(hash_y1.x - 0.49999f, hash_y1.y - 0.49999f, hash_y1.z - 0.49999f, hash_y1.w - 0.49999f);
  float4 grad_z1 = make_float4(hash_z1.x - 0.49999f, hash_z1.y - 0.49999f, hash_z1.z - 0.49999f, hash_z1.w - 0.49999f);

  float4 grad0 = make_float4(
    rsqrtf(grad_x0.x * grad_x0.x + grad_y0.x * grad_y0.x + grad_z0.x * grad_z0.x)
      * (pf.x * grad_x0.x + pf.y * grad_y0.x + pf.z * grad_z0.x),
    rsqrtf(grad_x0.y * grad_x0.y + grad_y0.y * grad_y0.y + grad_z0.y * grad_z0.y)
      * (pf_min1.x * grad_x0.y + pf.y * grad_y0.y + pf.z * grad_z0.y),
    rsqrtf(grad_x0.z * grad_x0.z + grad_y0.z * grad_y0.z + grad_z0.z * grad_z0.z)
      * (pf.x * grad_x0.z + pf_min1.y * grad_y0.z + pf.z * grad_z0.z),
    rsqrtf(grad_x0.w * grad_x0.w + grad_y0.w * grad_y0.w + grad_z0.w * grad_z0.w)
      * (pf_min1.x * grad_x0.w + pf_min1.y * grad_y0.w + pf.z * grad_z0.w));

  float4 grad1 = make_float4(
    rsqrtf(grad_x1.x * grad_x1.x + grad_y1.x * grad_y1.x + grad_z1.x * grad_z1.x)
      * (pf.x * grad_x1.x + pf.y * grad_y1.x + pf_min1.z * grad_z1.x),
    rsqrtf(grad_x1.y * grad_x1.y + grad_y1.y * grad_y1.y + grad_z1.y * grad_z1.y)
      * (pf_min1.x * grad_x1.y + pf.y * grad_y1.y + pf_min1.z * grad_z1.y),
    rsqrtf(grad_x1.z * grad_x1.z + grad_y1.z * grad_y1.z + grad_z1.z * grad_z1.z)
      * (pf.x * grad_x1.z + pf_min1.y * grad_y1.z + pf_min1.z * grad_z1.z),
    rsqrtf(grad_x1.w * grad_x1.w + grad_y1.w * grad_y1.w + grad_z1.w * grad_z1.w)
      * (pf_min1.x * grad_x1.w + pf_min1.y * grad_y1.w + pf_min1.z * grad_z1.w));

  float3 blend = make_float3(interp_cubic_d2(pf.x), interp_cubic_d2(pf.y), interp_cubic_d2(pf.z));

  float4 res = make_float4(
    lerp(grad0.x, grad1.x, blend.z), lerp(grad0.y, grad1.y, blend.z), lerp(grad0.z, grad1.z, blend.z), lerp(grad0.w, grad1.w, blend.z));

  float4 blend2 = make_float4(blend.x, blend.y, 1.0f - blend.x, 1.0f - blend.y);

  float final = res.x * blend2.z * blend2.w + res.y * blend2.x * blend2.w + res.z * blend2.z * blend2.y + res.w * blend2.x * blend2.y;

  final /= sqrtf(0.75f);

  return ((final * 1.5f) + 1.0f) * 0.5f;
}

LUM_DEVICE_FUNC float perlin_octaves(const vec3 p, const float scale, const int octaves, const bool tile) {
  float frequency   = 1.0f;
  float persistence = 1.0f;

  float value = 0.0f;

#pragma unroll
  for (int i = 0; i < octaves; i++) {
    value += persistence * perlin(p, scale * frequency, tile);
    persistence *= 0.5f;
    frequency *= 2.0f;
  }

  return value;
}

LUM_DEVICE_FUNC vec3 voronoi_hash(vec3 x, float scale) {
  x.x = fmodf(x.x, scale);
  x.y = fmodf(x.y, scale);
  x.z = fmodf(x.z, scale);

  x = get_vector(
    dot_product(x, get_vector(127.1f, 311.7f, 74.7f)), dot_product(x, get_vector(269.5f, 183.3f, 246.1f)),
    dot_product(x, get_vector(113.5f, 271.9f, 124.6f)));

  const float h = 43758.5453123f;

  return get_vector(fractf(sinf(x.x) * h), fractf(sinf(x.y) * h), fractf(sinf(x.z) * h));
}

LUM_DEVICE_FUNC vec3 voronoi(vec3 x, float scale, float seed, bool inverted) {
  x = scale_vector(x, scale);
  x = add_vector_const(x, 0.5f);

  vec3 p = floor_vector(x);
  vec3 f = fract_vector(x);

  float id   = 0.0f;
  float2 res = make_float2(1.0f, 1.0f);
  for (int k = -1; k <= 1; k++) {
    for (int j = -1; j <= 1; j++) {
      for (int i = -1; i <= 1; i++) {
        vec3 b  = get_vector((float) i, (float) j, (float) k);
        vec3 r  = add_vector(sub_vector(b, f), voronoi_hash(add_vector_const(add_vector(p, b), seed * 10.0f), scale));
        float d = dot_product(r, r);

        if (d < res.x) {
          id  = dot_product(add_vector(p, b), get_vector(1.0f, 57.0f, 113.0f));
          res = make_float2(d, res.x);
        }
        else if (d < res.y) {
          res.y = d;
        }
      }
    }
  }

  id = fabsf(id);

  if (inverted) {
    return get_vector(1.0f - res.x, 1.0f - res.y, id);
  }
  else {
    return get_vector(res.x, res.y, id);
  }
}

LUM_DEVICE_FUNC float worley_octaves(const vec3 p, float scale, const int octaves, const float seed, const float persistence) {
  float value = __saturatef(voronoi(p, scale, seed, true).x);

  float frequency = 2.0f;

#pragma unroll
  for (int i = 1; i < octaves; i++) {
    value -= persistence * __saturatef(voronoi(p, scale * frequency, seed, false).x);
    frequency *= 2.0f;
  }

  return value;
}

LUM_DEVICE_FUNC float dilate_perlin_worley(const float p, const float w, float x) {
  float curve = 0.75f;

  if (x < 0.5f) {
    x *= 2.0f;
    float n = p + w * x;
    return n * lerp(1.0f, 0.5f, powf(x, curve));
  }
  else {
    x       = 2.0f * (x - 0.5f);
    float n = w + p * (1.0f - x);
    return n * lerp(0.5f, 1.0f, powf(x, 1.0f / curve));
  }
}

#endif /* CU_CLOUD_NOISE_H */
