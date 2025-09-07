#ifndef CU_MIPMAP_H
#define CU_MIPMAP_H

#include "utils.cuh"

LUMINARY_KERNEL void mipmap_generate_level_3D_RGBA8(const KernelArgsMipmapGenerateLevel3DRGBA8 args) {
  uint32_t id = THREAD_ID;

  const uint32_t amount = args.width * args.height * args.depth;

  const float scale_x = 1.0f / args.width;
  const float scale_y = 1.0f / args.height;
  const float scale_z = 1.0f / args.depth;

  while (id < amount) {
    const uint32_t z = id / (args.width * args.height);
    const uint32_t y = (id - z * (args.width * args.height)) / args.width;
    const uint32_t x = id - y * args.width - z * args.width * args.height;

    const float sx = scale_x * (x + 0.5f);
    const float sy = scale_y * (y + 0.5f);
    const float sz = scale_z * (z + 0.5f);

    float4 v = tex3D<float4>(args.src, sx, sy, sz);

    v.x = fminf(255.0f * v.x + 0.5f, 255.9f);
    v.y = fminf(255.0f * v.y + 0.5f, 255.9f);
    v.z = fminf(255.0f * v.z + 0.5f, 255.9f);
    v.w = fminf(255.0f * v.w + 0.5f, 255.9f);

    surf3Dwrite(make_uchar4(v.x, v.y, v.z, v.w), args.dst, x * sizeof(uchar4), y, z);

    id += blockDim.x * gridDim.x;
  }
}

LUMINARY_KERNEL void mipmap_generate_level_2D_RGBA8(const KernelArgsMipmapGenerateLevel2DRGBA8 args) {
  uint32_t id = THREAD_ID;

  const uint32_t amount = args.width * args.height;

  const float scale_x = 1.0f / args.width;
  const float scale_y = 1.0f / args.height;

  while (id < amount) {
    const uint32_t y = id / args.width;
    const uint32_t x = id - y * args.width;

    const float sx = scale_x * (x + 0.5f);
    const float sy = scale_y * (y + 0.5f);

    float4 v = tex2D<float4>(args.src, sx, sy);

    v.x = fminf(255.0f * v.x + 0.5f, 255.9f);
    v.y = fminf(255.0f * v.y + 0.5f, 255.9f);
    v.z = fminf(255.0f * v.z + 0.5f, 255.9f);
    v.w = fminf(255.0f * v.w + 0.5f, 255.9f);

    surf2Dwrite(make_uchar4(v.x, v.y, v.z, v.w), args.dst, x * sizeof(uchar4), y);

    id += blockDim.x * gridDim.x;
  }
}

LUMINARY_KERNEL void mipmap_generate_level_3D_RGBA16(const KernelArgsMipmapGenerateLevel3DRGBA16 args) {
  uint32_t id = THREAD_ID;

  const uint32_t amount = args.width * args.height * args.depth;

  const float scale_x = 1.0f / args.width;
  const float scale_y = 1.0f / args.height;
  const float scale_z = 1.0f / args.depth;

  while (id < amount) {
    const uint32_t z = id / (args.width * args.height);
    const uint32_t y = (id - z * (args.width * args.height)) / args.width;
    const uint32_t x = id - y * args.width - z * args.width * args.height;

    const float sx = scale_x * (x + 0.5f);
    const float sy = scale_y * (y + 0.5f);
    const float sz = scale_z * (z + 0.5f);

    float4 v = tex3D<float4>(args.src, sx, sy, sz);

    v.x = fminf(65535.0f * v.x + 0.5f, 65535.9f);
    v.y = fminf(65535.0f * v.y + 0.5f, 65535.9f);
    v.z = fminf(65535.0f * v.z + 0.5f, 65535.9f);
    v.w = fminf(65535.0f * v.w + 0.5f, 65535.9f);

    surf3Dwrite(make_ushort4(v.x, v.y, v.z, v.w), args.dst, x * sizeof(ushort4), y, z);

    id += blockDim.x * gridDim.x;
  }
}

LUMINARY_KERNEL void mipmap_generate_level_2D_RGBA16(const KernelArgsMipmapGenerateLevel2DRGBA16 args) {
  uint32_t id = THREAD_ID;

  const uint32_t amount = args.width * args.height;

  const float scale_x = 1.0f / args.width;
  const float scale_y = 1.0f / args.height;

  while (id < amount) {
    const uint32_t y = id / args.width;
    const uint32_t x = id - y * args.width;

    const float sx = scale_x * (x + 0.5f);
    const float sy = scale_y * (y + 0.5f);

    float4 v = tex2D<float4>(args.src, sx, sy);

    v.x = fminf(65535.0f * v.x + 0.5f, 65535.9f);
    v.y = fminf(65535.0f * v.y + 0.5f, 65535.9f);
    v.z = fminf(65535.0f * v.z + 0.5f, 65535.9f);
    v.w = fminf(65535.0f * v.w + 0.5f, 65535.9f);

    surf2Dwrite(make_ushort4(v.x, v.y, v.z, v.w), args.dst, x * sizeof(ushort4), y);

    id += blockDim.x * gridDim.x;
  }
}

LUMINARY_KERNEL void mipmap_generate_level_3D_RGBAF(const KernelArgsMipmapGenerateLevel3DRGBAF args) {
  uint32_t id = THREAD_ID;

  const uint32_t amount = args.width * args.height * args.depth;

  const float scale_x = 1.0f / args.width;
  const float scale_y = 1.0f / args.height;
  const float scale_z = 1.0f / args.depth;

  while (id < amount) {
    const uint32_t z = id / (args.width * args.height);
    const uint32_t y = (id - z * (args.width * args.height)) / args.width;
    const uint32_t x = id - y * args.width - z * args.width * args.height;

    const float sx = scale_x * (x + 0.5f);
    const float sy = scale_y * (y + 0.5f);
    const float sz = scale_z * (z + 0.5f);

    float4 v = tex3D<float4>(args.src, sx, sy, sz);

    surf3Dwrite(v, args.dst, x * sizeof(float4), y, z);

    id += blockDim.x * gridDim.x;
  }
}

LUMINARY_KERNEL void mipmap_generate_level_2D_RGBAF(const KernelArgsMipmapGenerateLevel2DRGBAF args) {
  uint32_t id = THREAD_ID;

  const uint32_t amount = args.width * args.height;

  const float scale_x = 1.0f / args.width;
  const float scale_y = 1.0f / args.height;

  while (id < amount) {
    const uint32_t y = id / args.width;
    const uint32_t x = id - y * args.width;

    const float sx = scale_x * (x + 0.5f);
    const float sy = scale_y * (y + 0.5f);

    float4 v = tex2D<float4>(args.src, sx, sy);

    surf2Dwrite(v, args.dst, x * sizeof(float4), y);

    id += blockDim.x * gridDim.x;
  }
}

#endif /* CU_MIPMAP_H */
