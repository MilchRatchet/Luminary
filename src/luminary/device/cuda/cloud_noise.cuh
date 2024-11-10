#ifndef CU_CLOUD_NOISE_H
#define CU_CLOUD_NOISE_H

#include "math.cuh"
#include "utils.cuh"

/*
 * This file contains code to generate noise textures needed for the clouds.
 * The code of this file is based on the cloud generation in https://github.com/turanszkij/WickedEngine.
 */

/*
 * Cubic interpolation with second derivative equal to 0 on boundaries
 */
__device__ float interp_cubic_d2(const float x) {
  return x * x * x * (x * (x * 6.0f - 15.0f) + 10.0f);
}

__device__ void perlin_hash(
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

__device__ float perlin(vec3 p, const float scale, const bool tile) {
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

__device__ float perlin_octaves(const vec3 p, const float scale, const int octaves, const bool tile) {
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

__device__ vec3 voronoi_hash(vec3 x, float scale) {
  x.x = fmodf(x.x, scale);
  x.y = fmodf(x.y, scale);
  x.z = fmodf(x.z, scale);

  x = get_vector(
    dot_product(x, get_vector(127.1f, 311.7f, 74.7f)), dot_product(x, get_vector(269.5f, 183.3f, 246.1f)),
    dot_product(x, get_vector(113.5f, 271.9f, 124.6f)));

  const float h = 43758.5453123f;

  return get_vector(fractf(sinf(x.x) * h), fractf(sinf(x.y) * h), fractf(sinf(x.z) * h));
}

__device__ vec3 voronoi(vec3 x, float scale, float seed, bool inverted) {
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

__device__ float worley_octaves(const vec3 p, float scale, const int octaves, const float seed, const float persistence) {
  float value = __saturatef(voronoi(p, scale, seed, true).x);

  float frequency = 2.0f;

#pragma unroll
  for (int i = 1; i < octaves; i++) {
    value -= persistence * __saturatef(voronoi(p, scale * frequency, seed, false).x);
    frequency *= 2.0f;
  }

  return value;
}

__device__ float dilate_perlin_worley(const float p, const float w, float x) {
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

LUMINARY_KERNEL void cloud_compute_shape_noise(const KernelArgsCloudComputeShapeNoise args) {
  uint32_t id = THREAD_ID;

  uchar4* dst = (uchar4*) args.tex;

  const uint32_t amount = args.dim * args.dim * args.dim;
  const float scale_x   = 1.0f / args.dim;
  const float scale_y   = 1.0f / args.dim;
  const float scale_z   = 1.0f / args.dim;

  while (id < amount) {
    const uint32_t z = id / (args.dim * args.dim);
    const uint32_t y = (id - z * (args.dim * args.dim)) / args.dim;
    const uint32_t x = id - y * args.dim - z * args.dim * args.dim;

    const float sx = x * scale_x;
    const float sy = y * scale_y;
    const float sz = z * scale_z;

    const float size_scale = 1.0f;

    float perlin_dilate = perlin_octaves(get_vector(sx, sy, sz), 4.0f * size_scale, 7, true);
    float worley_dilate = worley_octaves(get_vector(sx, sy, sz), 6.0f * size_scale, 3, 0.0f, 0.3f);

    float worley_large  = worley_octaves(get_vector(sx, sy, sz), 6.0f * size_scale, 3, 0.0f, 0.3f);
    float worley_medium = worley_octaves(get_vector(sx, sy, sz), 12.0f * size_scale, 3, 0.0f, 0.3f);
    float worley_small  = worley_octaves(get_vector(sx, sy, sz), 24.0f * size_scale, 3, 0.0f, 0.3f);

    perlin_dilate = remap01(perlin_dilate, 0.3f, 1.4f);
    worley_dilate = remap01(worley_dilate, -0.3f, 1.3f);

    worley_large  = remap01(worley_large, -0.4f, 1.0f);
    worley_medium = remap01(worley_medium, -0.4f, 1.0f);
    worley_small  = remap01(worley_small, -0.4f, 1.0f);

    float perlin_worley = dilate_perlin_worley(perlin_dilate, worley_dilate, 0.3f);

    dst[x + y * args.dim + z * args.dim * args.dim] = make_uchar4(
      __saturatef(perlin_worley) * 255.0f, __saturatef(worley_large) * 255.0f, __saturatef(worley_medium) * 255.0f,
      __saturatef(worley_small) * 255.0f);

    id += blockDim.x * gridDim.x;
  }
}

LUMINARY_KERNEL void cloud_compute_detail_noise(const KernelArgsCloudComputeDetailNoise args) {
  uint32_t id = THREAD_ID;

  uchar4* dst = (uchar4*) args.tex;

  const uint32_t amount = args.dim * args.dim * args.dim;
  const float scale_x   = 1.0f / args.dim;
  const float scale_y   = 1.0f / args.dim;
  const float scale_z   = 1.0f / args.dim;

  while (id < amount) {
    const uint32_t z = id / (args.dim * args.dim);
    const uint32_t y = (id - z * (args.dim * args.dim)) / args.dim;
    const uint32_t x = id - y * args.dim - z * args.dim * args.dim;

    const float sx = x * scale_x;
    const float sy = y * scale_y;
    const float sz = z * scale_z;

    const float size_scale = 0.5f;

    float worley_large  = worley_octaves(get_vector(sx, sy, sz), 10.0f * size_scale, 3, 0.0f, 0.3f);
    float worley_medium = worley_octaves(get_vector(sx, sy, sz), 15.0f * size_scale, 3, 0.0f, 0.3f);
    float worley_small  = worley_octaves(get_vector(sx, sy, sz), 20.0f * size_scale, 3, 0.0f, 0.3f);

    worley_large  = remap01(worley_large, -1.0f, 1.0f);
    worley_medium = remap01(worley_medium, -1.0f, 1.0f);
    worley_small  = remap01(worley_small, -1.0f, 1.0f);

    dst[x + y * args.dim + z * args.dim * args.dim] =
      make_uchar4(__saturatef(worley_large) * 255.0f, __saturatef(worley_medium) * 255.0f, __saturatef(worley_small) * 255.0f, 255);

    id += blockDim.x * gridDim.x;
  }
}

LUMINARY_KERNEL void cloud_compute_weather_noise(const KernelArgsCloudComputeWeatherNoise args) {
  uint32_t id = THREAD_ID;

  uchar4* dst = (uchar4*) args.tex;

  const uint32_t amount = args.dim * args.dim;
  const float scale_x   = 1.0f / args.dim;
  const float scale_y   = 1.0f / args.dim;

  while (id < amount) {
    const uint32_t y = id / args.dim;
    const uint32_t x = id - y * args.dim;

    const float sx = x * scale_x;
    const float sy = y * scale_y;

    const float size_scale                  = 3.0f;
    const float coverage_perlin_worley_diff = 0.4f;

    const float remap_low  = 0.5f;
    const float remap_high = 1.3f;

    float perlin1 = perlin_octaves(get_vector(sx, sy, 0.0f), 2.0f * size_scale, 7, true);
    float worley1 = worley_octaves(get_vector(sx, sy, 0.0f), 3.0f * size_scale, 2, args.seed, 0.25f);
    float perlin2 = perlin_octaves(get_vector(sx, sy, 500.0f), 4.0f * size_scale, 7, true);
    float perlin3 = perlin_octaves(get_vector(sx, sy, 100.0f), 2.0f * size_scale, 7, true);
    float perlin4 = perlin_octaves(get_vector(sx, sy, 200.0f), 3.0f * size_scale, 7, true);

    perlin1 = remap01(perlin1, remap_low, remap_high);
    worley1 = remap01(worley1, remap_low, remap_high);
    perlin2 = remap01(perlin2, remap_low, remap_high);
    perlin3 = remap01(perlin3, remap_low, remap_high);
    perlin4 = remap01(perlin4, remap_low, remap_high);

    perlin1 = powf(perlin1, 1.0f);
    worley1 = powf(worley1, 0.75f);
    perlin2 = powf(perlin2, 2.0f);
    perlin3 = powf(perlin3, 3.0f);
    perlin4 = powf(perlin4, 1.0f);

    perlin1 = __saturatef(perlin1 * 1.2f) * 0.4f + 0.1f;
    worley1 = __saturatef(1.0f - worley1 * 2.0f);
    perlin2 = __saturatef(perlin2) * 0.5f;
    perlin3 = __saturatef(1.0f - perlin3 * 3.0f);
    perlin4 = __saturatef(1.0f - perlin4 * 1.5f);
    perlin4 = dilate_perlin_worley(worley1, perlin4, coverage_perlin_worley_diff);

    perlin1 -= perlin4;
    perlin2 -= perlin4 * perlin4;

    perlin1 = remap01(2.0f * perlin1, 0.05, 1.0);

    dst[x + y * args.dim] = make_uchar4(
      __saturatef(perlin1) * 255.0f, __saturatef(perlin2) * 255.0f, __saturatef(perlin3) * 255.0f, __saturatef(perlin4) * 255.0f);

    id += blockDim.x * gridDim.x;
  }
}

#endif /* CU_CLOUD_NOISE_H */
