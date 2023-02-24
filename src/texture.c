#include "texture.h"

#include <cuda_runtime_api.h>
#include <string.h>

#include "buffer.h"
#include "log.h"
#include "structs.h"

#define gpuErrchk(ans)                                         \
  {                                                            \
    if (ans != cudaSuccess) {                                  \
      crash_message("GPUassert: %s", cudaGetErrorString(ans)); \
    }                                                          \
  }

static struct cudaExtent texture_make_cudaextent(size_t depth, size_t height, size_t width) {
  const struct cudaExtent extent = {.depth = depth, .height = height, .width = width};

  return extent;
}

static struct cudaPitchedPtr texture_make_cudapitchedptr(void* ptr, size_t pitch, size_t xsize, size_t ysize) {
  const struct cudaPitchedPtr pitchedptr = {.pitch = pitch, .ptr = ptr, .xsize = xsize, .ysize = ysize};

  return pitchedptr;
}

static enum cudaTextureAddressMode texture_get_address_mode(TextureRGBA* tex) {
  switch (tex->wrap_mode) {
    case TexModeWrap:
      return cudaAddressModeWrap;
    case TexModeClamp:
      return cudaAddressModeClamp;
    case TexModeMirror:
      return cudaAddressModeMirror;
    case TexModeBorder:
      return cudaAddressModeBorder;
    default:
      error_message("Invalid texture wrapping mode %d\n", tex->wrap_mode);
      return cudaAddressModeWrap;
  }
}

static enum cudaTextureReadMode texture_get_read_mode(TextureRGBA* tex) {
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

static enum cudaTextureFilterMode texture_get_filter_mode(TextureRGBA* tex) {
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

static enum cudaMemcpyKind texture_get_copy_to_device_type(TextureRGBA* tex) {
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

static struct cudaChannelFormatDesc texture_get_channel_format_desc(TextureRGBA* tex) {
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

static size_t texture_get_pixel_size(TextureRGBA* tex) {
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

  enum cudaTextureAddressMode address_mode = texture_get_address_mode(tex);

  struct cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0]   = address_mode;
  texDesc.addressMode[1]   = address_mode;
  texDesc.addressMode[2]   = address_mode;
  texDesc.filterMode       = texture_get_filter_mode(tex);
  texDesc.maxAnisotropy    = 16;
  texDesc.readMode         = texture_get_read_mode(tex);
  texDesc.normalizedCoords = 1;

  struct cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));

  const unsigned int width  = tex->width;
  const unsigned int height = tex->height;
  const unsigned int pitch  = tex->pitch;
  void* data                = tex->data;

  switch (tex->dim) {
    case Tex2D: {
      void* data_gpu;
      size_t pitch_gpu = device_malloc_pitch((void**) &data_gpu, pitch * pixel_size, height);
      gpuErrchk(
        cudaMemcpy2D(data_gpu, pitch_gpu, data, pitch * pixel_size, width * pixel_size, height, texture_get_copy_to_device_type(tex)));

      resDesc.resType                  = cudaResourceTypePitch2D;
      resDesc.res.pitch2D.devPtr       = data_gpu;
      resDesc.res.pitch2D.width        = width;
      resDesc.res.pitch2D.height       = height;
      resDesc.res.pitch2D.desc         = texture_get_channel_format_desc(tex);
      resDesc.res.pitch2D.pitchInBytes = pitch_gpu;
    } break;
    case Tex3D: {
      const unsigned int depth = tex->depth;

      struct cudaChannelFormatDesc channelDesc = texture_get_channel_format_desc(tex);

      struct cudaArray* array_gpu;
      gpuErrchk(cudaMalloc3DArray(&array_gpu, &channelDesc, texture_make_cudaextent(pitch, height, depth), 0));

      struct cudaMemcpy3DParms copy_params = {0};
      copy_params.srcPtr                   = texture_make_cudapitchedptr(data, pitch * pixel_size, height, depth);
      copy_params.dstArray                 = array_gpu;
      copy_params.extent                   = texture_make_cudaextent(pitch, height, depth);
      copy_params.kind                     = texture_get_copy_to_device_type(tex);

      gpuErrchk(cudaMemcpy3D(&copy_params));

      resDesc.resType         = cudaResourceTypeArray;
      resDesc.res.array.array = array_gpu;
    } break;
    default:
      crash_message("Invalid texture dimension type %d\n", tex->dim);
      break;
  }

  gpuErrchk(cudaCreateTextureObject(cudaTex, &resDesc, &texDesc, (const struct cudaResourceViewDesc*) 0));
}

void texture_create_atlas(DeviceBuffer** buffer, TextureRGBA* textures, const int textures_length) {
  if (!buffer) {
    error_message("buffer is NULL.");
    return;
  }

  cudaTextureObject_t* textures_cpu = (cudaTextureObject_t*) malloc(sizeof(cudaTextureObject_t) * textures_length);

  device_buffer_init(buffer);
  device_buffer_malloc(*buffer, sizeof(cudaTextureObject_t), textures_length);

  for (int i = 0; i < textures_length; i++) {
    texture_allocate(textures_cpu + i, textures + i);
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

  device_buffer_destroy(&texture_atlas);
  free(textures_cpu);
}
