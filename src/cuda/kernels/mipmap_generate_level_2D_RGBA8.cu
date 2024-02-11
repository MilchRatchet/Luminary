#include "utils.cuh"

__global__ void mipmap_generate_level_2D_RGBA8(cudaTextureObject_t src, cudaSurfaceObject_t dst, int width, int height) {
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

    v.x = fminf(255.9f * v.x, 255.9f);
    v.y = fminf(255.9f * v.y, 255.9f);
    v.z = fminf(255.9f * v.z, 255.9f);
    v.w = fminf(255.9f * v.w, 255.9f);

    surf2Dwrite(make_uchar4(v.x, v.y, v.z, v.w), dst, x * sizeof(uchar4), y);

    id += blockDim.x * gridDim.x;
  }
}
