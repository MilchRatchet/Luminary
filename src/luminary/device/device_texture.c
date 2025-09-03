#include "device_texture.h"

#include "internal_error.h"

static LuminaryResult _device_texture_get_address_mode(const TextureWrappingMode mode, CUaddress_mode* address_mode) {
  switch (mode) {
    case TEXTURE_WRAPPING_MODE_WRAP:
      *address_mode = CU_TR_ADDRESS_MODE_WRAP;
      return LUMINARY_SUCCESS;
    case TEXTURE_WRAPPING_MODE_CLAMP:
      *address_mode = CU_TR_ADDRESS_MODE_CLAMP;
      return LUMINARY_SUCCESS;
    case TEXTURE_WRAPPING_MODE_MIRROR:
      *address_mode = CU_TR_ADDRESS_MODE_MIRROR;
      return LUMINARY_SUCCESS;
    case TEXTURE_WRAPPING_MODE_BORDER:
      *address_mode = CU_TR_ADDRESS_MODE_BORDER;
      return LUMINARY_SUCCESS;
    default:
      __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Texture wrapping mode is invalid.");
  }
}

static LuminaryResult _device_texture_get_read_mode(const Texture* tex, uint32_t* flags) {
  switch (tex->type) {
    case TEXTURE_DATA_TYPE_FP32:
      return LUMINARY_SUCCESS;
    case TEXTURE_DATA_TYPE_U8:
    case TEXTURE_DATA_TYPE_U16:
      switch (tex->read_mode) {
        case TEXTURE_READ_MODE_NORMALIZED:
          return LUMINARY_SUCCESS;
        case TEXTURE_READ_MODE_ELEMENT:
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
    case TEXTURE_DATA_TYPE_FP32:
      *format = CU_AD_FORMAT_FLOAT;
      return LUMINARY_SUCCESS;
    case TEXTURE_DATA_TYPE_U8:
      *format = CU_AD_FORMAT_UNSIGNED_INT8;
      return LUMINARY_SUCCESS;
    case TEXTURE_DATA_TYPE_U16:
      *format = CU_AD_FORMAT_UNSIGNED_INT16;
      return LUMINARY_SUCCESS;
    default:
      __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Texture data type is invalid.");
  }
}

static LuminaryResult _device_texture_get_filter_mode(const Texture* tex, CUfilter_mode* mode) {
  switch (tex->filter) {
    case TEXTURE_FILTER_MODE_POINT:
      *mode = CU_TR_FILTER_MODE_POINT;
      return LUMINARY_SUCCESS;
    case TEXTURE_FILTER_MODE_LINEAR:
      *mode = CU_TR_FILTER_MODE_LINEAR;
      return LUMINARY_SUCCESS;
    default:
      __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Texture filter mode is invalid.");
  }
}

static LuminaryResult _device_texture_get_pixel_size(const Texture* tex, size_t* size) {
  switch (tex->type) {
    case TEXTURE_DATA_TYPE_FP32:
      *size = tex->num_components * sizeof(float);
      return LUMINARY_SUCCESS;
    case TEXTURE_DATA_TYPE_U8:
      *size = tex->num_components * sizeof(uint8_t);
      return LUMINARY_SUCCESS;
    case TEXTURE_DATA_TYPE_U16:
      *size = tex->num_components * sizeof(uint16_t);
      return LUMINARY_SUCCESS;
    default:
      __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Texture data type is invalid.");
  }
}

static LuminaryResult _device_texture_get_num_mip_levels(const Texture* tex, uint8_t* num_mip_levels) {
  if (tex->mipmap != TEXTURE_MIPMAP_MODE_GENERATE) {
    *num_mip_levels = 1;
    return LUMINARY_SUCCESS;
  }

  uint16_t max_dim = max(tex->width, max(tex->height, tex->depth));

  if (max_dim == 0) {
    *num_mip_levels = 1;
    return LUMINARY_SUCCESS;
  }

  uint32_t level = 0;

  while (max_dim != 1) {
    level++;
    max_dim = max_dim >> 1;
  }

  *num_mip_levels = level;

  return LUMINARY_SUCCESS;
}

static LuminaryResult _device_texture_generate_mipmaps(
  DeviceTexture* device_texture, const CUDA_TEXTURE_DESC* texture_desc, CUstream stream) {
  __CHECK_NULL_ARGUMENT(device_texture);
  __CHECK_NULL_ARGUMENT(texture_desc);

  for (uint32_t mip_level = 0; mip_level + 1 < device_texture->num_mip_levels; mip_level++) {
    CUarray src_level;
    CUDA_FAILURE_HANDLE(cuMipmappedArrayGetLevel(&src_level, device_texture->cuda_mipmapped_array, mip_level));

    CUarray dst_level;
    CUDA_FAILURE_HANDLE(cuMipmappedArrayGetLevel(&dst_level, device_texture->cuda_mipmapped_array, mip_level + 1));

    // Create SRC texture
    CUDA_RESOURCE_DESC texture_res_desc;
    memset(&texture_res_desc, 0, sizeof(CUDA_RESOURCE_DESC));

    texture_res_desc.resType          = CU_RESOURCE_TYPE_ARRAY;
    texture_res_desc.res.array.hArray = src_level;

    CUtexObject src_tex;
    CUDA_FAILURE_HANDLE(cuTexObjectCreate(&src_tex, &texture_res_desc, texture_desc, (const CUDA_RESOURCE_VIEW_DESC*) 0));

    // Create DST surface
    CUDA_RESOURCE_DESC surface_res_desc;
    memset(&surface_res_desc, 0, sizeof(CUDA_RESOURCE_DESC));

    surface_res_desc.resType          = CU_RESOURCE_TYPE_ARRAY;
    surface_res_desc.res.array.hArray = dst_level;

    CUsurfObject dst_surface;
    CUDA_FAILURE_HANDLE(cuSurfObjectCreate(&dst_surface, &surface_res_desc));

    // Execute kernels

    CUDA_FAILURE_HANDLE(cuTexObjectDestroy(src_tex));
    CUDA_FAILURE_HANDLE(cuSurfObjectDestroy(dst_surface));
  }

  return LUMINARY_SUCCESS;
}

LuminaryResult device_texture_create(DeviceTexture** device_texture, const Texture* texture, CUstream stream) {
  __CHECK_NULL_ARGUMENT(device_texture);
  __CHECK_NULL_ARGUMENT(texture);

  __FAILURE_HANDLE(texture_await(texture));

  size_t pixel_size;
  __FAILURE_HANDLE(_device_texture_get_pixel_size(texture, &pixel_size));

  uint8_t num_mip_levels;
  __FAILURE_HANDLE(_device_texture_get_num_mip_levels(texture, &num_mip_levels));

  CUDA_TEXTURE_DESC tex_desc;
  memset(&tex_desc, 0, sizeof(CUDA_TEXTURE_DESC));

  __FAILURE_HANDLE(_device_texture_get_address_mode(texture->wrap_mode_S, tex_desc.addressMode + 0));
  __FAILURE_HANDLE(_device_texture_get_address_mode(texture->wrap_mode_T, tex_desc.addressMode + 1));
  __FAILURE_HANDLE(_device_texture_get_address_mode(texture->wrap_mode_R, tex_desc.addressMode + 2));
  __FAILURE_HANDLE(_device_texture_get_filter_mode(texture, &tex_desc.filterMode));
  tex_desc.maxAnisotropy       = 1;
  tex_desc.flags               = CU_TRSF_NORMALIZED_COORDINATES;
  tex_desc.minMipmapLevelClamp = 0;
  tex_desc.maxMipmapLevelClamp = num_mip_levels - 1;
  tex_desc.mipmapFilterMode    = CU_TR_FILTER_MODE_POINT;
  __FAILURE_HANDLE(_device_texture_get_read_mode(texture, &tex_desc.flags));

  CUDA_RESOURCE_DESC res_desc;
  memset(&res_desc, 0, sizeof(CUDA_RESOURCE_DESC));

  const uint32_t width  = texture->width;
  const uint32_t height = texture->height;
  const uint32_t depth  = texture->depth;
  const uint32_t pitch  = texture->pitch;
  const void* data      = texture->data;

  log_message("Allocating device texture of dimension %ux%ux%u.", width, height, depth);

  __FAILURE_HANDLE(host_malloc(device_texture, sizeof(DeviceTexture)));
  memset(*device_texture, 0, sizeof(DeviceTexture));

  bool tex_is_valid;
  __FAILURE_HANDLE(texture_is_valid(texture, &tex_is_valid));

  switch (texture->dim) {
    case TEXTURE_DIMENSION_TYPE_2D: {
      switch (texture->mipmap) {
        default:
          __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Texture mipmap mode is invalid.");
        case TEXTURE_MIPMAP_MODE_NONE: {
          __FAILURE_HANDLE(device_malloc2D(&(*device_texture)->cuda_memory, width * pixel_size, height));

          __FAILURE_HANDLE(device_memory_get_pitch((*device_texture)->cuda_memory, &(*device_texture)->pitch));

          if (data != (void*) 0) {
            __FAILURE_HANDLE(device_upload2D((*device_texture)->cuda_memory, data, pitch, width * pixel_size, height, stream));
          }

          res_desc.resType                  = CU_RESOURCE_TYPE_PITCH2D;
          res_desc.res.pitch2D.devPtr       = DEVICE_CUPTR((*device_texture)->cuda_memory);
          res_desc.res.pitch2D.width        = width;
          res_desc.res.pitch2D.height       = height;
          res_desc.res.pitch2D.numChannels  = texture->num_components;
          res_desc.res.pitch2D.pitchInBytes = (*device_texture)->pitch;

          __FAILURE_HANDLE(_device_texture_get_format(texture, &res_desc.res.pitch2D.format));
        } break;
        case TEXTURE_MIPMAP_MODE_GENERATE: {
          if (texture->num_components != 4) {
            __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Mipmapping is only supported for textures with 4 components.");
          }

          if (data == (void*) 0) {
            __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "On demand mipmap generation is not supported.");
          }

          CUDA_ARRAY3D_DESCRIPTOR descriptor;
          memset(&descriptor, 0, sizeof(CUDA_ARRAY3D_DESCRIPTOR));

          descriptor.Width       = width;
          descriptor.Height      = height;
          descriptor.Depth       = 0;
          descriptor.Flags       = 0;
          descriptor.NumChannels = texture->num_components;

          __FAILURE_HANDLE(_device_texture_get_format(texture, &descriptor.Format));

          // TODO: Add support in device_memory
          CUDA_FAILURE_HANDLE(cuMipmappedArrayCreate(&(*device_texture)->cuda_mipmapped_array, &descriptor, num_mip_levels));

          CUarray base_mip;
          CUDA_FAILURE_HANDLE(cuMipmappedArrayGetLevel(&base_mip, (*device_texture)->cuda_mipmapped_array, 0));

          CUDA_MEMCPY2D memcpy_info;
          memset(&memcpy_info, 0, sizeof(CUDA_MEMCPY2D));

          memcpy_info.dstArray      = base_mip;
          memcpy_info.dstMemoryType = CU_MEMORYTYPE_ARRAY;
          memcpy_info.srcHost       = data;
          memcpy_info.srcMemoryType = CU_MEMORYTYPE_HOST;
          memcpy_info.srcPitch      = width * pixel_size;
          memcpy_info.WidthInBytes  = width * pixel_size;
          memcpy_info.Height        = height;

          CUDA_FAILURE_HANDLE(cuMemcpy2DAsync(&memcpy_info, stream));

          res_desc.resType                    = CU_RESOURCE_TYPE_MIPMAPPED_ARRAY;
          res_desc.res.mipmap.hMipmappedArray = (*device_texture)->cuda_mipmapped_array;
        } break;
      }
    } break;
    case TEXTURE_DIMENSION_TYPE_3D: {
      switch (texture->mipmap) {
        default:
          __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Texture mipmap mode is invalid.");
        case TEXTURE_MIPMAP_MODE_NONE: {
          CUDA_ARRAY3D_DESCRIPTOR descriptor;
          memset(&descriptor, 0, sizeof(CUDA_ARRAY3D_DESCRIPTOR));

          descriptor.Width       = width;
          descriptor.Height      = height;
          descriptor.Depth       = depth;
          descriptor.Flags       = 0;
          descriptor.NumChannels = texture->num_components;

          __FAILURE_HANDLE(_device_texture_get_format(texture, &descriptor.Format));

          // TODO: Add support in device_memory
          CUDA_FAILURE_HANDLE(cuArray3DCreate(&(*device_texture)->cuda_array, &descriptor));

          (*device_texture)->pitch = width * pixel_size;

          if (data != (void*) 0) {
            CUDA_MEMCPY3D memcpy_info;
            memset(&memcpy_info, 0, sizeof(CUDA_MEMCPY3D));

            memcpy_info.dstArray      = (*device_texture)->cuda_array;
            memcpy_info.dstMemoryType = CU_MEMORYTYPE_ARRAY;
            memcpy_info.srcHost       = data;
            memcpy_info.srcMemoryType = CU_MEMORYTYPE_HOST;
            memcpy_info.srcPitch      = width * pixel_size;
            memcpy_info.WidthInBytes  = width * pixel_size;
            memcpy_info.Height        = height;
            memcpy_info.Depth         = depth;

            CUDA_FAILURE_HANDLE(cuMemcpy3DAsync(&memcpy_info, stream));
          }

          res_desc.resType          = CU_RESOURCE_TYPE_ARRAY;
          res_desc.res.array.hArray = (*device_texture)->cuda_array;
        } break;
        case TEXTURE_MIPMAP_MODE_GENERATE: {
          __RETURN_ERROR(LUMINARY_ERROR_NOT_IMPLEMENTED, "Mipmaps are currently not supported for 3D textures.");
        } break;
      }
    } break;
    default:
      __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Texture dimension type is invalid.");
  }

  if (width > 0xFFFF || height > 0xFFFF) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Texture dimension exceeds limits: %ux%u.", width, height);
  }

  if (tex_is_valid) {
    CUDA_FAILURE_HANDLE(cuTexObjectCreate(&(*device_texture)->tex, &res_desc, &tex_desc, (const CUDA_RESOURCE_VIEW_DESC*) 0));
  }
  else {
    (*device_texture)->tex = TEXTURE_OBJECT_INVALID;
  }

  (*device_texture)->status         = tex_is_valid ? TEXTURE_STATUS_NONE : TEXTURE_STATUS_INVALID;
  (*device_texture)->width          = width;
  (*device_texture)->height         = height;
  (*device_texture)->depth          = depth;
  (*device_texture)->gamma          = texture->gamma;
  (*device_texture)->is_3D          = (texture->dim == TEXTURE_DIMENSION_TYPE_3D);
  (*device_texture)->pixel_size     = pixel_size;
  (*device_texture)->num_mip_levels = num_mip_levels;

  if (texture->mipmap == TEXTURE_MIPMAP_MODE_GENERATE) {
    __FAILURE_HANDLE(_device_texture_generate_mipmaps(*device_texture, &tex_desc, stream));
  }

  return LUMINARY_SUCCESS;
}

LuminaryResult device_texture_copy_from_mem(DeviceTexture* device_texture, const DEVICE void* mem, CUstream stream) {
  __CHECK_NULL_ARGUMENT(device_texture);
  __CHECK_NULL_ARGUMENT(mem);

  if (device_texture->num_mip_levels > 1) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "This is not supported for mipmapped textures.");
  }

  if (device_texture->is_3D) {
    CUDA_MEMCPY3D memcpy_info;
    memset(&memcpy_info, 0, sizeof(CUDA_MEMCPY3D));

    memcpy_info.dstArray      = device_texture->cuda_array;
    memcpy_info.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    memcpy_info.srcDevice     = DEVICE_CUPTR(mem);
    memcpy_info.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    memcpy_info.WidthInBytes  = device_texture->width * device_texture->pixel_size;
    memcpy_info.Height        = device_texture->height;
    memcpy_info.Depth         = device_texture->depth;

    CUDA_FAILURE_HANDLE(cuMemcpy3DAsync(&memcpy_info, stream));
  }
  else {
    CUDA_MEMCPY2D memcpy_info;
    memset(&memcpy_info, 0, sizeof(CUDA_MEMCPY2D));

    memcpy_info.dstDevice     = DEVICE_CUPTR(device_texture->cuda_memory);
    memcpy_info.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    memcpy_info.dstPitch      = device_texture->pitch;
    memcpy_info.dstXInBytes   = 0;
    memcpy_info.dstY          = 0;
    memcpy_info.srcDevice     = DEVICE_CUPTR(mem);
    memcpy_info.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    memcpy_info.srcPitch      = device_texture->width * device_texture->pixel_size;
    memcpy_info.srcXInBytes   = 0;
    memcpy_info.srcY          = 0;
    memcpy_info.WidthInBytes  = device_texture->width * device_texture->pixel_size;
    memcpy_info.Height        = device_texture->height;

    CUDA_FAILURE_HANDLE(cuMemcpy2DAsync(&memcpy_info, stream));
  }

  return LUMINARY_SUCCESS;
}

LuminaryResult device_texture_destroy(DeviceTexture** device_texture) {
  __CHECK_NULL_ARGUMENT(device_texture);

  if ((*device_texture)->tex != TEXTURE_OBJECT_INVALID)
    CUDA_FAILURE_HANDLE(cuTexObjectDestroy((*device_texture)->tex));

  if ((*device_texture)->is_3D) {
    CUDA_FAILURE_HANDLE(cuArrayDestroy((*device_texture)->cuda_array));
  }
  else {
    if ((*device_texture)->num_mip_levels > 1) {
      CUDA_FAILURE_HANDLE(cuMipmappedArrayDestroy((*device_texture)->cuda_mipmapped_array));
    }
    else {
      __FAILURE_HANDLE(device_free(&(*device_texture)->cuda_memory));
    }
  }

  __FAILURE_HANDLE(host_free(device_texture));

  return LUMINARY_SUCCESS;
}
