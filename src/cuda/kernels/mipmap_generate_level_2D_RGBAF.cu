#include "utils.cuh"

__global__ void mipmap_generate_level_2D_RGBAF(cudaTextureObject_t src, cudaSurfaceObject_t dst, int width, int height) {
  unsigned int id = THREAD_ID;

  const int amount = width * height;

  const float scale_x = 1.0f / (width - 1);
  const float scale_y = 1.0f / (height - 1);

  while (id < amount) {
    const int x = id % width;
    const int y = id / width;

    const float sx = scale_x * x + scale_x * 0.5f;
    const float sy = scale_y * y + scale_y * 0.5f;

    float4 v = tex2D<float4>(src, sx, sy);

    surf2Dwrite(v, dst, x * sizeof(float4), y);

    id += blockDim.x * gridDim.x;
  }
}
