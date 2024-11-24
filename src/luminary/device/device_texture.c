#include "device_texture.h"

#include "internal_error.h"

static LuminaryResult _device_texture_get_address_mode(const TextureWrappingMode mode, CUaddress_mode* address_mode) {
  switch (mode) {
    case TexModeWrap:
      *address_mode = CU_TR_ADDRESS_MODE_WRAP;
      return LUMINARY_SUCCESS;
    case TexModeClamp:
      *address_mode = CU_TR_ADDRESS_MODE_CLAMP;
      return LUMINARY_SUCCESS;
    case TexModeMirror:
      *address_mode = CU_TR_ADDRESS_MODE_MIRROR;
      return LUMINARY_SUCCESS;
    case TexModeBorder:
      *address_mode = CU_TR_ADDRESS_MODE_BORDER;
      return LUMINARY_SUCCESS;
    default:
      __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Texture wrapping mode is invalid.");
  }
}

static LuminaryResult _device_texture_get_read_mode(const Texture* tex, uint32_t* flags) {
  switch (tex->type) {
    case TexDataFP32:
      return LUMINARY_SUCCESS;
    case TexDataUINT8:
    case TexDataUINT16:
      switch (tex->read_mode) {
        case TexReadModeNormalized:
          return LUMINARY_SUCCESS;
        case TexReadModeElement:
          *flags = *flags | CU_TRSF_READ_AS_INTEGER;
          return LUMINARY_SUCCESS;
        default:
          __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Texture read mode is invalid.");
      }
    default:
      __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Texture data type is invalid.");
  }
}

static LuminaryResult _device_texture_get_format(const Texture* tex, CUarray_format* format) {
  switch (tex->type) {
    case TexDataFP32:
      *format = CU_AD_FORMAT_FLOAT;
      return LUMINARY_SUCCESS;
    case TexDataUINT8:
      *format = CU_AD_FORMAT_UNSIGNED_INT8;
      return LUMINARY_SUCCESS;
    case TexDataUINT16:
      *format = CU_AD_FORMAT_UNSIGNED_INT16;
      return LUMINARY_SUCCESS;
    default:
      __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Texture data type is invalid.");
  }
}

static LuminaryResult _device_texture_get_filter_mode(const Texture* tex, CUfilter_mode* mode) {
  switch (tex->filter) {
    case TexFilterPoint:
      *mode = CU_TR_FILTER_MODE_POINT;
      return LUMINARY_SUCCESS;
    case TexFilterLinear:
      *mode = CU_TR_FILTER_MODE_LINEAR;
      return LUMINARY_SUCCESS;
    default:
      __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Texture filter mode is invalid.");
  }
}

static LuminaryResult _device_texture_get_pixel_size(const Texture* tex, size_t* size) {
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

LuminaryResult device_texture_create(DeviceTexture** _device_texture, const Texture* texture, CUstream stream) {
  __CHECK_NULL_ARGUMENT(_device_texture);
  __CHECK_NULL_ARGUMENT(texture);

  size_t pixel_size;
  __FAILURE_HANDLE(_device_texture_get_pixel_size(texture, &pixel_size));

  CUDA_TEXTURE_DESC tex_desc;
  memset(&tex_desc, 0, sizeof(CUDA_TEXTURE_DESC));

  __FAILURE_HANDLE(_device_texture_get_address_mode(texture->wrap_mode_S, tex_desc.addressMode + 0));
  __FAILURE_HANDLE(_device_texture_get_address_mode(texture->wrap_mode_T, tex_desc.addressMode + 1));
  __FAILURE_HANDLE(_device_texture_get_address_mode(texture->wrap_mode_R, tex_desc.addressMode + 2));
  __FAILURE_HANDLE(_device_texture_get_filter_mode(texture, &tex_desc.filterMode));
  tex_desc.mipmapFilterMode    = CU_TR_FILTER_MODE_POINT;
  tex_desc.maxAnisotropy       = 16;
  tex_desc.flags               = CU_TRSF_NORMALIZED_COORDINATES;
  tex_desc.minMipmapLevelClamp = 0;
  __FAILURE_HANDLE(_device_texture_get_read_mode(texture, &tex_desc.flags));

  CUDA_RESOURCE_DESC res_desc;
  memset(&res_desc, 0, sizeof(CUDA_RESOURCE_DESC));

  const uint32_t width  = texture->width;
  const uint32_t height = texture->height;
  const uint32_t depth  = texture->depth;
  const uint32_t pitch  = texture->pitch;
  const void* data      = texture->data;

  log_message("Allocating device texture of dimension %ux%ux%u.", width, height, depth);

  DEVICE void* data_device;

  size_t pitch_gpu;

  switch (texture->dim) {
    case Tex2D: {
      switch (texture->mipmap) {
        default:
          __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Texture mipmap mode is invalid.");
        case TexMipmapNone: {
          __FAILURE_HANDLE(device_malloc2D(&data_device, width * pixel_size, height));

          __FAILURE_HANDLE(device_memory_get_pitch(data_device, &pitch_gpu));

          if (data != (void*) 0) {
            __FAILURE_HANDLE(device_upload2D(data_device, data, pitch, width * pixel_size, height, stream));
          }

          res_desc.resType                  = CU_RESOURCE_TYPE_PITCH2D;
          res_desc.res.pitch2D.devPtr       = DEVICE_CUPTR(data_device);
          res_desc.res.pitch2D.width        = width;
          res_desc.res.pitch2D.height       = height;
          res_desc.res.pitch2D.numChannels  = texture->num_components;
          res_desc.res.pitch2D.pitchInBytes = pitch_gpu;

          __FAILURE_HANDLE(_device_texture_get_format(texture, &res_desc.res.pitch2D.format));
        } break;
        case TexMipmapGenerate: {
          __RETURN_ERROR(LUMINARY_ERROR_NOT_IMPLEMENTED, "Mipmaps are currently not supported.");
        } break;
      }
    } break;
    case Tex3D: {
      switch (texture->mipmap) {
        default:
          __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Texture mipmap mode is invalid.");
        case TexMipmapNone: {
          // TODO: Add support in device_memory

          CUDA_ARRAY3D_DESCRIPTOR descriptor;
          descriptor.Width       = width;
          descriptor.Height      = height;
          descriptor.Depth       = depth;
          descriptor.Flags       = 0;
          descriptor.NumChannels = texture->num_components;

          __FAILURE_HANDLE(_device_texture_get_format(texture, &descriptor.Format));

          CUDA_FAILURE_HANDLE(cuArray3DCreate((CUarray*) &data_device, &descriptor));

          pitch_gpu = width * pixel_size;

          if (data != (void*) 0) {
            CUDA_MEMCPY3D memcpy_info;
            memset(&memcpy_info, 0, sizeof(CUDA_MEMCPY3D));

            memcpy_info.dstArray      = (CUarray) data_device;
            memcpy_info.dstMemoryType = CU_MEMORYTYPE_ARRAY;
            memcpy_info.srcHost       = data;
            memcpy_info.srcMemoryType = CU_MEMORYTYPE_HOST;
            memcpy_info.WidthInBytes  = width * pixel_size;
            memcpy_info.Height        = height;
            memcpy_info.Depth         = depth;

            CUDA_FAILURE_HANDLE(cuMemcpy3DAsync(&memcpy_info, stream));
          }

          res_desc.resType          = CU_RESOURCE_TYPE_ARRAY;
          res_desc.res.array.hArray = (CUarray) data_device;
        } break;
        case TexMipmapGenerate: {
          __RETURN_ERROR(LUMINARY_ERROR_NOT_IMPLEMENTED, "Mipmaps are currently not supported.");
        } break;
      }
    } break;
    default:
      __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Texture dimension type is invalid.");
  }

  DeviceTexture* device_texture;
  __FAILURE_HANDLE(host_malloc(&device_texture, sizeof(DeviceTexture)));

  CUDA_FAILURE_HANDLE(cuTexObjectCreate(&device_texture->tex, &res_desc, &tex_desc, (const CUDA_RESOURCE_VIEW_DESC*) 0));

  if (width > 0xFFFF || height > 0xFFFF) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Texture dimension exceeds limits: %ux%u.", width, height);
  }

  device_texture->memory = data_device;
  device_texture->width  = width;
  device_texture->height = height;
  device_texture->gamma  = texture->gamma;
  device_texture->is_3D  = (texture->dim == Tex3D);
  device_texture->pitch  = pitch_gpu;

  *_device_texture = device_texture;

  return LUMINARY_SUCCESS;
}

LuminaryResult device_texture_destroy(DeviceTexture** device_texture) {
  __CHECK_NULL_ARGUMENT(device_texture);

  CUDA_FAILURE_HANDLE(cuTexObjectDestroy((*device_texture)->tex));

  if ((*device_texture)->is_3D) {
    CUDA_FAILURE_HANDLE(cuArrayDestroy((CUarray) (*device_texture)->memory));
  }
  else {
    __FAILURE_HANDLE(device_free(&(*device_texture)->memory));
  }

  __FAILURE_HANDLE(host_free(device_texture));

  return LUMINARY_SUCCESS;
}
