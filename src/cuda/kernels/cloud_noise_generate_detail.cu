#include "cloud_noise.cuh"
#include "math.cuh"
#include "utils.cuh"

__global__ void cloud_noise_generate_detail(const int dim, uint8_t* tex) {
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

    const float size_scale = 0.5f;

    float worley_large  = worley_octaves(get_vector(sx, sy, sz), 10.0f * size_scale, 3, 0.0f, 0.3f);
    float worley_medium = worley_octaves(get_vector(sx, sy, sz), 15.0f * size_scale, 3, 0.0f, 0.3f);
    float worley_small  = worley_octaves(get_vector(sx, sy, sz), 20.0f * size_scale, 3, 0.0f, 0.3f);

    worley_large  = remap01(worley_large, -1.0f, 1.0f);
    worley_medium = remap01(worley_medium, -1.0f, 1.0f);
    worley_small  = remap01(worley_small, -1.0f, 1.0f);

    dst[x + y * dim + z * dim * dim] =
      make_uchar4(__saturatef(worley_large) * 255.0f, __saturatef(worley_medium) * 255.0f, __saturatef(worley_small) * 255.0f, 255);

    id += blockDim.x * gridDim.x;
  }
}
