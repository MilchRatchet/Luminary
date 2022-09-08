#ifndef CU_CUDATEXTURE_H
#define CU_CUDATEXTURE_H

#include "utils.cuh"

static void cudatexture_allocate(cudaTextureObject_t* cudaTex, TextureRGBA* tex) {
  const size_t pixel_size = (tex->type == TexDataFP32) ? sizeof(RGBAF) : sizeof(RGBA8);

  struct cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0]   = cudaAddressModeWrap;
  texDesc.addressMode[1]   = cudaAddressModeWrap;
  texDesc.addressMode[2]   = cudaAddressModeWrap;
  texDesc.filterMode       = cudaFilterModeLinear;
  texDesc.maxAnisotropy    = 16;
  texDesc.readMode         = (tex->type == TexDataFP32) ? cudaReadModeElementType : cudaReadModeNormalizedFloat;
  texDesc.normalizedCoords = 1;

  struct cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));

  if (tex->volume_tex) {
    const unsigned int height = tex->height;
    const unsigned int depth  = tex->depth;
    const unsigned int pitch  = tex->pitch;
    void* data                = tex->data;

    struct cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();

    cudaArray* array_gpu;
    gpuErrchk(cudaMalloc3DArray(&array_gpu, &channelDesc, make_cudaExtent(pitch, height, depth), 0));

    struct cudaMemcpy3DParms copy_params = {0};
    copy_params.srcPtr                   = make_cudaPitchedPtr(data, pitch * pixel_size, height, depth);
    copy_params.dstArray                 = array_gpu;
    copy_params.extent                   = make_cudaExtent(pitch, height, depth);
    copy_params.kind                     = (tex->gpu) ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;

    gpuErrchk(cudaMemcpy3D(&copy_params));

    resDesc.resType         = cudaResourceTypeArray;
    resDesc.res.array.array = array_gpu;
  }
  else {
    const unsigned int width  = tex->width;
    const unsigned int height = tex->height;
    const unsigned int pitch  = tex->pitch;
    const void* data          = tex->data;

    void* data_gpu;
    size_t pitch_gpu = device_malloc_pitch((void**) &data_gpu, pitch * pixel_size, height);
    if (tex->gpu) {
      gpuErrchk(cudaMemcpy2D(data_gpu, pitch_gpu, data, pitch * pixel_size, width * pixel_size, height, cudaMemcpyDeviceToDevice));
    }
    else {
      gpuErrchk(cudaMemcpy2D(data_gpu, pitch_gpu, data, pitch * pixel_size, width * pixel_size, height, cudaMemcpyHostToDevice));
    }

    resDesc.resType                  = cudaResourceTypePitch2D;
    resDesc.res.pitch2D.devPtr       = data_gpu;
    resDesc.res.pitch2D.width        = width;
    resDesc.res.pitch2D.height       = height;
    resDesc.res.pitch2D.desc         = cudaCreateChannelDesc<uchar4>();
    resDesc.res.pitch2D.pitchInBytes = pitch_gpu;
  }

  gpuErrchk(cudaCreateTextureObject(cudaTex, &resDesc, &texDesc, NULL));
}

extern "C" DeviceBuffer* cudatexture_allocate_to_buffer(TextureRGBA* textures, const int textures_length) {
  cudaTextureObject_t* textures_cpu = (cudaTextureObject_t*) malloc(sizeof(cudaTextureObject_t) * textures_length);
  DeviceBuffer* textures_gpu;

  device_buffer_init(&textures_gpu);
  device_buffer_malloc(textures_gpu, sizeof(cudaTextureObject_t), textures_length);

  for (int i = 0; i < textures_length; i++) {
    cudatexture_allocate(textures_cpu + i, textures + i);
  }

  device_buffer_upload(textures_gpu, textures_cpu);

  free(textures_cpu);

  return textures_gpu;
}

extern "C" void cudatexture_free_buffer(DeviceBuffer* texture_atlas, const int textures_length) {
  cudaTextureObject_t* textures_cpu = (cudaTextureObject_t*) malloc(device_buffer_get_size(texture_atlas));
  device_buffer_download_full(texture_atlas, textures_cpu);

  for (int i = 0; i < textures_length; i++) {
    gpuErrchk(cudaDestroyTextureObject(textures_cpu[i]));
  }

  device_buffer_destroy(texture_atlas);
  free(textures_cpu);
}

#endif /* CU_CUDATEXTURE_H */