#include "texture.h"

#include <cuda_runtime_api.h>
#include <string.h>

#include "buffer.h"
#include "device.h"
#include "log.h"
#include "structs.h"

struct cudaExtent texture_make_cudaextent(size_t depth, size_t height, size_t width) {
  const struct cudaExtent extent = {.depth = depth, .height = height, .width = width};

  return extent;
}

struct cudaPitchedPtr texture_make_cudapitchedptr(void* ptr, size_t pitch, size_t xsize, size_t ysize) {
  const struct cudaPitchedPtr pitchedptr = {.pitch = pitch, .ptr = ptr, .xsize = xsize, .ysize = ysize};

  return pitchedptr;
}

enum cudaTextureAddressMode texture_get_address_mode(TextureWrappingMode mode) {
  switch (mode) {
    case TexModeWrap:
      return cudaAddressModeWrap;
    case TexModeClamp:
      return cudaAddressModeClamp;
    case TexModeMirror:
      return cudaAddressModeMirror;
    case TexModeBorder:
      return cudaAddressModeBorder;
    default:
      error_message("Invalid texture wrapping mode %d\n", mode);
      return cudaAddressModeWrap;
  }
}

enum cudaTextureReadMode texture_get_read_mode(TextureRGBA* tex) {
  switch (tex->type) {
    case TexDataFP32:
      return cudaReadModeElementType;
    case TexDataUINT8:
      return cudaReadModeNormalizedFloat;
    default:
      error_message("Invalid texture data type %d\n", tex->type);
      return cudaReadModeElementType;
  }
}

enum cudaTextureFilterMode texture_get_filter_mode(TextureRGBA* tex) {
  switch (tex->filter) {
    case TexFilterPoint:
      return cudaFilterModePoint;
    case TexFilterLinear:
      return cudaFilterModeLinear;
    default:
      error_message("Invalid texture filter mode %d\n", tex->filter);
      return cudaFilterModeLinear;
  }
}

enum cudaMemcpyKind texture_get_copy_to_device_type(TextureRGBA* tex) {
  switch (tex->storage) {
    case TexStorageCPU:
      return cudaMemcpyHostToDevice;
    case TexStorageGPU:
      return cudaMemcpyDeviceToDevice;
    default:
      crash_message("Invalid texture storage location %d\n", tex->storage);
      return cudaMemcpyHostToDevice;
  }
}

struct cudaChannelFormatDesc texture_get_channel_format_desc(TextureRGBA* tex) {
  switch (tex->type) {
    case TexDataFP32:
      return cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
    case TexDataUINT8:
      return cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
    default:
      crash_message("Invalid texture data type %d\n", tex->type);
      return cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
  }
}

size_t texture_get_pixel_size(TextureRGBA* tex) {
  switch (tex->type) {
    case TexDataFP32:
      return sizeof(RGBAF);
    case TexDataUINT8:
      return sizeof(RGBA8);
    default:
      crash_message("Invalid texture data type %d\n", tex->type);
      return sizeof(RGBA8);
  }
}

static void texture_allocate(cudaTextureObject_t* cudaTex, TextureRGBA* tex) {
  const size_t pixel_size = texture_get_pixel_size(tex);

  struct cudaTextureDesc tex_desc;
  memset(&tex_desc, 0, sizeof(tex_desc));
  tex_desc.addressMode[0]      = texture_get_address_mode(tex->wrap_mode_S);
  tex_desc.addressMode[1]      = texture_get_address_mode(tex->wrap_mode_T);
  tex_desc.addressMode[2]      = texture_get_address_mode(tex->wrap_mode_R);
  tex_desc.filterMode          = texture_get_filter_mode(tex);
  tex_desc.mipmapFilterMode    = cudaFilterModePoint;
  tex_desc.maxAnisotropy       = 16;
  tex_desc.readMode            = texture_get_read_mode(tex);
  tex_desc.normalizedCoords    = 1;
  tex_desc.minMipmapLevelClamp = 0;

  struct cudaResourceDesc res_desc;
  memset(&res_desc, 0, sizeof(res_desc));

  struct cudaChannelFormatDesc channelDesc = texture_get_channel_format_desc(tex);

  const unsigned int width  = tex->width;
  const unsigned int height = tex->height;
  const unsigned int depth  = tex->depth;
  const unsigned int pitch  = tex->pitch;
  void* data                = tex->data;

  switch (tex->dim) {
    case Tex2D: {
      switch (tex->mipmap) {
        default:
          error_message("Invalid texture mipmap mode %d", tex->mipmap);
        case TexMipmapNone: {
          void* data_gpu;
          size_t pitch_gpu = device_malloc_pitch((void**) &data_gpu, pitch * pixel_size, height);
          gpuErrchk(
            cudaMemcpy2D(data_gpu, pitch_gpu, data, pitch * pixel_size, width * pixel_size, height, texture_get_copy_to_device_type(tex)));

          res_desc.resType                  = cudaResourceTypePitch2D;
          res_desc.res.pitch2D.devPtr       = data_gpu;
          res_desc.res.pitch2D.width        = width;
          res_desc.res.pitch2D.height       = height;
          res_desc.res.pitch2D.desc         = channelDesc;
          res_desc.res.pitch2D.pitchInBytes = pitch_gpu;
        } break;
        case TexMipmapGenerate: {
          const unsigned int levels    = device_mipmap_compute_max_level(tex);
          tex->mipmap_max_level        = levels;
          tex_desc.maxMipmapLevelClamp = levels;

          cudaMipmappedArray_t mipmap_array;
          gpuErrchk(cudaMallocMipmappedArray(&mipmap_array, &channelDesc, texture_make_cudaextent(0, height, pitch), levels + 1, 0));

          cudaArray_t level_0;
          gpuErrchk(cudaGetMipmappedArrayLevel(&level_0, mipmap_array, 0));

          struct cudaMemcpy3DParms copy_params = {0};
          copy_params.srcPtr                   = texture_make_cudapitchedptr(data, pitch * pixel_size, width, height);
          copy_params.dstArray                 = level_0;
          copy_params.extent                   = texture_make_cudaextent(1, height, pitch);
          copy_params.kind                     = texture_get_copy_to_device_type(tex);

          gpuErrchk(cudaMemcpy3D(&copy_params));

          device_mipmap_generate(mipmap_array, tex);

          res_desc.resType           = cudaResourceTypeMipmappedArray;
          res_desc.res.mipmap.mipmap = mipmap_array;

        } break;
      }
    } break;
    case Tex3D: {
      switch (tex->mipmap) {
        default:
          error_message("Invalid texture mipmap mode %d", tex->mipmap);
        case TexMipmapNone: {
          struct cudaArray* array_gpu;
          gpuErrchk(cudaMalloc3DArray(&array_gpu, &channelDesc, texture_make_cudaextent(depth, height, pitch), 0));

          struct cudaMemcpy3DParms copy_params = {0};
          copy_params.srcPtr                   = texture_make_cudapitchedptr(data, pitch * pixel_size, width, height);
          copy_params.dstArray                 = array_gpu;
          copy_params.extent                   = texture_make_cudaextent(depth, height, pitch);
          copy_params.kind                     = texture_get_copy_to_device_type(tex);

          gpuErrchk(cudaMemcpy3D(&copy_params));

          res_desc.resType         = cudaResourceTypeArray;
          res_desc.res.array.array = array_gpu;
        } break;
        case TexMipmapGenerate: {
          const unsigned int levels    = device_mipmap_compute_max_level(tex);
          tex->mipmap_max_level        = levels;
          tex_desc.maxMipmapLevelClamp = levels;

          cudaMipmappedArray_t mipmap_array;
          gpuErrchk(cudaMallocMipmappedArray(&mipmap_array, &channelDesc, texture_make_cudaextent(depth, height, pitch), levels + 1, 0));

          cudaArray_t level_0;
          gpuErrchk(cudaGetMipmappedArrayLevel(&level_0, mipmap_array, 0));

          struct cudaMemcpy3DParms copy_params = {0};
          copy_params.srcPtr                   = texture_make_cudapitchedptr(data, pitch * pixel_size, width, height);
          copy_params.dstArray                 = level_0;
          copy_params.extent                   = texture_make_cudaextent(depth, height, pitch);
          copy_params.kind                     = texture_get_copy_to_device_type(tex);

          gpuErrchk(cudaMemcpy3D(&copy_params));

          device_mipmap_generate(mipmap_array, tex);

          res_desc.resType           = cudaResourceTypeMipmappedArray;
          res_desc.res.mipmap.mipmap = mipmap_array;
        } break;
      }

    } break;
    default:
      crash_message("Invalid texture dimension type %d\n", tex->dim);
      break;
  }

  gpuErrchk(cudaCreateTextureObject(cudaTex, &res_desc, &tex_desc, (const struct cudaResourceViewDesc*) 0));
}

void texture_create_atlas(DeviceBuffer** buffer, TextureRGBA* textures, const int textures_length) {
  if (!buffer) {
    error_message("buffer is NULL.");
    return;
  }

  DeviceTexture* textures_cpu = (DeviceTexture*) malloc(sizeof(DeviceTexture) * textures_length);

  device_buffer_init(buffer);
  device_buffer_malloc(*buffer, sizeof(DeviceTexture), textures_length);

  for (int i = 0; i < textures_length; i++) {
    texture_allocate(&textures_cpu[i].tex, textures + i);
    textures_cpu[i].inv_width  = 1.0f / (textures[i].width - 1);
    textures_cpu[i].inv_height = 1.0f / (textures[i].height - 1);
    textures_cpu[i].gamma      = textures[i].gamma;
  }

  device_buffer_upload(*buffer, textures_cpu);

  free(textures_cpu);
}

void texture_free_atlas(DeviceBuffer* texture_atlas, const int textures_length) {
  cudaTextureObject_t* textures_cpu = (cudaTextureObject_t*) malloc(device_buffer_get_size(texture_atlas));
  device_buffer_download_full(texture_atlas, textures_cpu);

  for (int i = 0; i < textures_length; i++) {
    gpuErrchk(cudaDestroyTextureObject(textures_cpu[i]));
  }

  device_buffer_free(texture_atlas);
  free(textures_cpu);
}

void texture_create(
  TextureRGBA* tex, unsigned int width, unsigned int height, unsigned int depth, unsigned int pitch, void* data, TextureDataType type,
  TextureStorageLocation storage) {
  if (tex == (TextureRGBA*) 0)
    return;

  const TextureRGBA _tex = {
    .width            = width,
    .height           = height,
    .depth            = depth,
    .pitch            = pitch,
    .data             = data,
    .dim              = (depth > 1) ? Tex3D : Tex2D,
    .storage          = storage,
    .type             = type,
    .wrap_mode_S      = TexModeWrap,
    .wrap_mode_T      = TexModeWrap,
    .wrap_mode_R      = TexModeWrap,
    .filter           = TexFilterLinear,
    .mipmap           = TexMipmapNone,
    .mipmap_max_level = 0,
    .gamma            = 1.0f};

  *tex = _tex;
}
