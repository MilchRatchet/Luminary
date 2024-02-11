#include "utils.cuh"

__global__ void mipmap_generate_level_3D_RGBAF(cudaTextureObject_t src, cudaSurfaceObject_t dst, int width, int height, int depth) {
  unsigned int id = THREAD_ID;

  const int amount = width * height * depth;

  const float scale_x = 1.0f / (width - 1);
  const float scale_y = 1.0f / (height - 1);
  const float scale_z = 1.0f / (depth - 1);

  while (id < amount) {
    const int x = id % width;
    const int y = (id % (width * height)) / width;
    const int z = id / (width * height);

    const float sx = scale_x * x + scale_x * 0.5f;
    const float sy = scale_y * y + scale_y * 0.5f;
    const float sz = scale_z * z + scale_z * 0.5f;

    float4 v = tex3D<float4>(src, sx, sy, sz);

    surf3Dwrite(v, dst, x * sizeof(float4), y, z);

    id += blockDim.x * gridDim.x;
  }
}
