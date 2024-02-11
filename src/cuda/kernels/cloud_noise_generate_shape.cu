#include "cloud_noise.cuh"
#include "math.cuh"
#include "utils.cuh"

__global__ void cloud_noise_generate_shape(const int dim, uint8_t* tex) {
  unsigned int id = THREAD_ID;

  uchar4* dst = (uchar4*) tex;

  const int amount    = dim * dim * dim;
  const float scale_x = 1.0f / dim;
  const float scale_y = 1.0f / dim;
  const float scale_z = 1.0f / dim;

  while (id < amount) {
    const int x = id % dim;
    const int y = (id % (dim * dim)) / dim;
    const int z = id / (dim * dim);

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

    dst[x + y * dim + z * dim * dim] = make_uchar4(
      __saturatef(perlin_worley) * 255.0f, __saturatef(worley_large) * 255.0f, __saturatef(worley_medium) * 255.0f,
      __saturatef(worley_small) * 255.0f);

    id += blockDim.x * gridDim.x;
  }
}
