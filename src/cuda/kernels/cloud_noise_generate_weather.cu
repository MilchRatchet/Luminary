#include "cloud_noise.cuh"
#include "math.cuh"
#include "utils.cuh"

__global__ void cloud_noise_generate_weather(const int dim, const float seed, uint8_t* tex) {
  unsigned int id = THREAD_ID;

  uchar4* dst = (uchar4*) tex;

  const int amount    = dim * dim;
  const float scale_x = 1.0f / dim;
  const float scale_y = 1.0f / dim;

  while (id < amount) {
    const int x = id % dim;
    const int y = id / dim;

    const float sx = x * scale_x;
    const float sy = y * scale_y;

    const float size_scale                  = 3.0f;
    const float coverage_perlin_worley_diff = 0.4f;

    const float remap_low  = 0.5f;
    const float remap_high = 1.3f;

    float perlin1 = perlin_octaves(get_vector(sx, sy, 0.0f), 2.0f * size_scale, 7, true);
    float worley1 = worley_octaves(get_vector(sx, sy, 0.0f), 3.0f * size_scale, 2, seed, 0.25f);
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

    dst[x + y * dim] = make_uchar4(
      __saturatef(perlin1) * 255.0f, __saturatef(perlin2) * 255.0f, __saturatef(perlin3) * 255.0f, __saturatef(perlin4) * 255.0f);

    id += blockDim.x * gridDim.x;
  }
}
