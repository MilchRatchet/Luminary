#include "device_texture.h"

#include "internal_error.h"

struct cudaExtent _device_texture_make_cudaextent(size_t depth, size_t height, size_t width) {
  const struct cudaExtent extent = {.depth = depth, .height = height, .width = width};

  return extent;
}

struct cudaPitchedPtr _device_texture_make_cudapitchedptr(void* ptr, size_t pitch, size_t xsize, size_t ysize) {
  const struct cudaPitchedPtr pitchedptr = {.pitch = pitch, .ptr = ptr, .xsize = xsize, .ysize = ysize};

  return pitchedptr;
}

LuminaryResult _device_texture_get_address_mode(const TextureWrappingMode mode, enum cudaTextureAddressMode* address_mode) {
  switch (mode) {
    case TexModeWrap:
      *address_mode = cudaAddressModeWrap;
      return LUMINARY_SUCCESS;
    case TexModeClamp:
      *address_mode = cudaAddressModeClamp;
      return LUMINARY_SUCCESS;
    case TexModeMirror:
      *address_mode = cudaAddressModeMirror;
      return LUMINARY_SUCCESS;
    case TexModeBorder:
      *address_mode = cudaAddressModeBorder;
      return LUMINARY_SUCCESS;
    default:
      __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Texture wrapping mode is invalid.");
  }
}

LuminaryResult _device_texture_get_read_mode(const Texture* tex, enum cudaTextureReadMode* mode) {
  switch (tex->type) {
    case TexDataFP32:
      *mode = cudaReadModeElementType;
      return LUMINARY_SUCCESS;
    case TexDataUINT8:
    case TexDataUINT16:
      switch (tex->read_mode) {
        case TexReadModeNormalized:
          *mode = cudaReadModeNormalizedFloat;
          return LUMINARY_SUCCESS;
        case TexReadModeElement:
          *mode = cudaReadModeElementType;
          return LUMINARY_SUCCESS;
        default:
          __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Texture read mode is invalid.");
      }
    default:
      __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Texture data type is invalid.");
  }
}

LuminaryResult _device_texture_get_filter_mode(const Texture* tex, enum cudaTextureFilterMode* mode) {
  switch (tex->filter) {
    case TexFilterPoint:
      *mode = cudaFilterModePoint;
      return LUMINARY_SUCCESS;
    case TexFilterLinear:
      *mode = cudaFilterModeLinear;
      return LUMINARY_SUCCESS;
    default:
      __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Texture filter mode is invalid.");
  }
}

LuminaryResult _device_texture_get_copy_to_device_type(const Texture* tex, enum cudaMemcpyKind* kind) {
  switch (tex->storage) {
    case TexStorageCPU:
      *kind = cudaMemcpyHostToDevice;
      return LUMINARY_SUCCESS;
    case TexStorageGPU:
      *kind = cudaMemcpyDeviceToDevice;
      return LUMINARY_SUCCESS;
    default:
      __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Texture storage location is invalid.");
  }
}

LuminaryResult _device_texture_get_channel_format_desc(const Texture* tex, struct cudaChannelFormatDesc* desc) {
  const int x_bits = (tex->num_components >= 1) ? 1 : 0;
  const int y_bits = (tex->num_components >= 2) ? 1 : 0;
  const int z_bits = (tex->num_components >= 3) ? 1 : 0;
  const int w_bits = (tex->num_components >= 4) ? 1 : 0;

  int bits_per_channel;
  enum cudaChannelFormatKind kind;
  switch (tex->type) {
    case TexDataFP32:
      bits_per_channel = 32;
      kind             = cudaChannelFormatKindFloat;
      break;
    case TexDataUINT8:
      bits_per_channel = 8;
      kind             = cudaChannelFormatKindUnsigned;
      break;
    case TexDataUINT16:
      bits_per_channel = 16;
      kind             = cudaChannelFormatKindUnsigned;
      break;
    default:
      __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Texture data type is invalid.");
  }

  *desc =
    cudaCreateChannelDesc(bits_per_channel * x_bits, bits_per_channel * y_bits, bits_per_channel * z_bits, bits_per_channel * w_bits, kind);

  return LUMINARY_SUCCESS;
}

LuminaryResult _device_texture_get_pixel_size(const Texture* tex, size_t* size) {
  switch (tex->type) {
    case TexDataFP32:
      *size = tex->num_components * sizeof(float);
      return LUMINARY_SUCCESS;
    case TexDataUINT8:
      *size = tex->num_components * sizeof(uint8_t);
      return LUMINARY_SUCCESS;
    case TexDataUINT16:
      *size = tex->num_components * sizeof(uint16_t);
      return LUMINARY_SUCCESS;
    default:
      __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Texture data type is invalid.");
  }
}

LuminaryResult device_texture_create(DeviceTexture** _device_texture, Texture* texture) {
  if (!_device_texture) {
    __RETURN_ERROR(LUMINARY_ERROR_ARGUMENT_NULL, "Device texture is NULL.");
  }

  if (!texture) {
    __RETURN_ERROR(LUMINARY_ERROR_ARGUMENT_NULL, "Texture is NULL.");
  }

  size_t pixel_size;
  __FAILURE_HANDLE(_device_texture_get_pixel_size(texture, &pixel_size));

  struct cudaTextureDesc tex_desc;
  memset(&tex_desc, 0, sizeof(tex_desc));

  __FAILURE_HANDLE(_device_texture_get_address_mode(texture->wrap_mode_S, tex_desc.addressMode + 0));
  __FAILURE_HANDLE(_device_texture_get_address_mode(texture->wrap_mode_T, tex_desc.addressMode + 1));
  __FAILURE_HANDLE(_device_texture_get_address_mode(texture->wrap_mode_R, tex_desc.addressMode + 2));
  __FAILURE_HANDLE(_device_texture_get_filter_mode(texture, &tex_desc.filterMode));
  tex_desc.mipmapFilterMode = cudaFilterModePoint;
  tex_desc.maxAnisotropy    = 16;
  __FAILURE_HANDLE(_device_texture_get_read_mode(texture, &tex_desc.readMode));
  tex_desc.normalizedCoords    = 1;
  tex_desc.minMipmapLevelClamp = 0;

  struct cudaResourceDesc res_desc;
  memset(&res_desc, 0, sizeof(res_desc));

  struct cudaChannelFormatDesc channel_desc;
  __FAILURE_HANDLE(texture_get_channel_format_desc(texture, &channel_desc));

  const unsigned int width  = texture->width;
  const unsigned int height = texture->height;
  const unsigned int depth  = texture->depth;
  const unsigned int pitch  = texture->pitch;
  void* data                = texture->data;

  log_message("Allocating device texture of dimension %ux%ux%u.", width, height, depth);

  switch (texture->dim) {
    case Tex2D: {
      switch (texture->mipmap) {
        default:
          __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Texture mipmap mode is invalid.");
        case TexMipmapNone: {
          void* data_gpu;
          size_t pitch_gpu;
          device_malloc_pitch((void**) &data_gpu, pitch * pixel_size, height, &pitch_gpu);

          enum cudaMemcpyKind memcpy_kind;
          __FAILURE_HANDLE(_device_texture_get_copy_to_device_type(texture, &memcpy_kind));

          gpuErrchk(cudaMemcpy2D(data_gpu, pitch_gpu, data, pitch * pixel_size, width * pixel_size, height, memcpy_kind));

          res_desc.resType                  = cudaResourceTypePitch2D;
          res_desc.res.pitch2D.devPtr       = data_gpu;
          res_desc.res.pitch2D.width        = width;
          res_desc.res.pitch2D.height       = height;
          res_desc.res.pitch2D.desc         = channel_desc;
          res_desc.res.pitch2D.pitchInBytes = pitch_gpu;
        } break;
        case TexMipmapGenerate: {
          if (texture->num_components != 4) {
            __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Texture num components is invalid.");
          }

          const unsigned int levels    = device_mipmap_compute_max_level(texture);
          tex_desc.maxMipmapLevelClamp = levels;

          cudaMipmappedArray_t mipmap_array;
          gpuErrchk(
            cudaMallocMipmappedArray(&mipmap_array, &channel_desc, _device_texture_make_cudaextent(0, height, pitch), levels + 1, 0));

          cudaArray_t level_0;
          gpuErrchk(cudaGetMipmappedArrayLevel(&level_0, mipmap_array, 0));

          struct cudaMemcpy3DParms copy_params = {0};
          copy_params.srcPtr                   = _device_texture_make_cudapitchedptr(data, pitch * pixel_size, width, height);
          copy_params.dstArray                 = level_0;
          copy_params.extent                   = _device_texture_make_cudaextent(1, height, pitch);

          __FAILURE_HANDLE(texture_get_copy_to_device_type(texture, &copy_params.kind));

          gpuErrchk(cudaMemcpy3D(&copy_params));

          device_mipmap_generate(mipmap_array, texture);

          res_desc.resType           = cudaResourceTypeMipmappedArray;
          res_desc.res.mipmap.mipmap = mipmap_array;

        } break;
      }
    } break;
    case Tex3D: {
      switch (texture->mipmap) {
        default:
          __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Texture mipmap mode is invalid.");
        case TexMipmapNone: {
          struct cudaArray* array_gpu;
          gpuErrchk(cudaMalloc3DArray(&array_gpu, &channel_desc, _device_texture_make_cudaextent(depth, height, pitch), 0));

          struct cudaMemcpy3DParms copy_params = {0};
          copy_params.srcPtr                   = _device_texture_make_cudapitchedptr(data, pitch * pixel_size, width, height);
          copy_params.dstArray                 = array_gpu;
          copy_params.extent                   = _device_texture_make_cudaextent(depth, height, pitch);

          __FAILURE_HANDLE(texture_get_copy_to_device_type(texture, &copy_params.kind));

          gpuErrchk(cudaMemcpy3D(&copy_params));

          res_desc.resType         = cudaResourceTypeArray;
          res_desc.res.array.array = array_gpu;
        } break;
        case TexMipmapGenerate: {
          if (texture->num_components != 4) {
            __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Texture num components is invalid.");
          }

          const unsigned int levels    = device_mipmap_compute_max_level(texture);
          tex_desc.maxMipmapLevelClamp = levels;

          cudaMipmappedArray_t mipmap_array;
          gpuErrchk(
            cudaMallocMipmappedArray(&mipmap_array, &channel_desc, _device_texture_make_cudaextent(depth, height, pitch), levels + 1, 0));

          cudaArray_t level_0;
          gpuErrchk(cudaGetMipmappedArrayLevel(&level_0, mipmap_array, 0));

          struct cudaMemcpy3DParms copy_params = {0};
          copy_params.srcPtr                   = _device_texture_make_cudapitchedptr(data, pitch * pixel_size, width, height);
          copy_params.dstArray                 = level_0;
          copy_params.extent                   = _device_texture_make_cudaextent(depth, height, pitch);

          __FAILURE_HANDLE(texture_get_copy_to_device_type(texture, &copy_params.kind));

          gpuErrchk(cudaMemcpy3D(&copy_params));

          device_mipmap_generate(mipmap_array, texture);

          res_desc.resType           = cudaResourceTypeMipmappedArray;
          res_desc.res.mipmap.mipmap = mipmap_array;
        } break;
      }

    } break;
    default:
      __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Texture dimension type is invalid.");
  }

  DeviceTexture* device_texture;
  __FAILURE_HANDLE(host_malloc(&device_texture, sizeof(DeviceTexture)));

  gpuErrchk(cudaCreateTextureObject(&device_texture->tex, &res_desc, &tex_desc, (const struct cudaResourceViewDesc*) 0));

  device_texture->inv_width  = 1.0f / width;
  device_texture->inv_height = 1.0f / height;
  device_texture->gamma      = texture->gamma;

  return LUMINARY_SUCCESS;
}
