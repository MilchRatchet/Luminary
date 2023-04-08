#ifndef CU_MIPMAP_H
#define CU_MIPMAP_H

#include <cuda_runtime_api.h>

#include "device.h"
#include "structs.h"
#include "texture.h"
#include "utils.cuh"

__global__ void mipmap_generate_level_3D_RGBA8(cudaTextureObject_t src, cudaSurfaceObject_t dst, int width, int height, int depth) {
  unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;

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

__global__ void mipmap_generate_level_2D_RGBA8(cudaTextureObject_t src, cudaSurfaceObject_t dst, int width, int height) {
  unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;

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

__global__ void mipmap_generate_level_3D_RGBAF(cudaTextureObject_t src, cudaSurfaceObject_t dst, int width, int height, int depth) {
  unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;

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

__global__ void mipmap_generate_level_2D_RGBAF(cudaTextureObject_t src, cudaSurfaceObject_t dst, int width, int height) {
  unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;

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

extern "C" void device_mipmap_generate(cudaMipmappedArray_t mipmap_array, TextureRGBA* tex) {
  const int num_levels = device_mipmap_compute_max_level(tex);

  cudaTextureFilterMode filter_mode = texture_get_filter_mode(tex);
  cudaTextureReadMode read_mode     = texture_get_read_mode(tex);

  for (int level = 0; level < num_levels; level++) {
    cudaArray_t level_src;
    gpuErrchk(cudaGetMipmappedArrayLevel(&level_src, mipmap_array, level));
    cudaArray_t level_dst;
    gpuErrchk(cudaGetMipmappedArrayLevel(&level_dst, mipmap_array, level + 1));

    cudaExtent dst_size;
    gpuErrchk(cudaArrayGetInfo(NULL, &dst_size, NULL, level_dst));

    cudaTextureObject_t src_tex;
    cudaResourceDesc res_desc;
    memset(&res_desc, 0, sizeof(cudaResourceDesc));

    res_desc.resType         = cudaResourceTypeArray;
    res_desc.res.array.array = level_src;

    cudaTextureDesc tex_desc;
    memset(&tex_desc, 0, sizeof(cudaTextureDesc));

    tex_desc.normalizedCoords = 1;
    tex_desc.filterMode       = filter_mode;

    tex_desc.addressMode[0] = texture_get_address_mode(tex->wrap_mode_S);
    tex_desc.addressMode[1] = texture_get_address_mode(tex->wrap_mode_T);
    tex_desc.addressMode[2] = texture_get_address_mode(tex->wrap_mode_R);

    tex_desc.readMode = read_mode;

    gpuErrchk(cudaCreateTextureObject(&src_tex, &res_desc, &tex_desc, NULL));

    cudaSurfaceObject_t dst_surface;
    cudaResourceDesc res_desc_surface;
    memset(&res_desc_surface, 0, sizeof(cudaResourceDesc));
    res_desc_surface.resType         = cudaResourceTypeArray;
    res_desc_surface.res.array.array = level_dst;

    gpuErrchk(cudaCreateSurfaceObject(&dst_surface, &res_desc_surface));

    switch (tex->dim) {
      case Tex2D:
        switch (tex->type) {
          case TexDataFP32:
            mipmap_generate_level_2D_RGBAF<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(src_tex, dst_surface, dst_size.width, dst_size.height);
            break;
          case TexDataUINT8:
            mipmap_generate_level_2D_RGBA8<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(src_tex, dst_surface, dst_size.width, dst_size.height);
            break;
          default:
            error_message("Invalid texture data type %d", tex->type);
            break;
        }
        break;
      case Tex3D:
        switch (tex->type) {
          case TexDataFP32:
            mipmap_generate_level_3D_RGBAF<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(
              src_tex, dst_surface, dst_size.width, dst_size.height, dst_size.depth);
            break;
          case TexDataUINT8:
            mipmap_generate_level_3D_RGBA8<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(
              src_tex, dst_surface, dst_size.width, dst_size.height, dst_size.depth);
            break;
          default:
            error_message("Invalid texture data type %d", tex->type);
            break;
        }
        break;
      default:
        error_message("Invalid texture dim %d", tex->dim);
        break;
    }

    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaGetLastError());

    gpuErrchk(cudaDestroySurfaceObject(dst_surface));

    gpuErrchk(cudaDestroyTextureObject(src_tex));
  }
}

extern "C" unsigned int device_mipmap_compute_max_level(TextureRGBA* tex) {
  int max_dim;

  switch (tex->dim) {
    case Tex2D:
      max_dim = max(tex->width, tex->height);
      break;
    case Tex3D:
      max_dim = max(tex->width, max(tex->height, tex->depth));
      break;
    default:
      error_message("Invalid texture dim %d", tex->dim);
      return 0;
  }

  if (max_dim == 0)
    return 0;

  int i = 0;

  while (max_dim != 1) {
    i++;
    max_dim = max_dim >> 1;
  }

  return i;
}

#endif /* CU_MIPMAP_H */
