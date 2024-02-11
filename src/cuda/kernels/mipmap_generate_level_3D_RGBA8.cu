#include "utils.cuh"

__global__ void mipmap_generate_level_3D_RGBA8(cudaTextureObject_t src, cudaSurfaceObject_t dst, int width, int height, int depth) {
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

    v.x = fminf(255.9f * v.x, 255.9f);
    v.y = fminf(255.9f * v.y, 255.9f);
    v.z = fminf(255.9f * v.z, 255.9f);
    v.w = fminf(255.9f * v.w, 255.9f);

    surf3Dwrite(make_uchar4(v.x, v.y, v.z, v.w), dst, x * sizeof(uchar4), y, z);

    id += blockDim.x * gridDim.x;
  }
}
