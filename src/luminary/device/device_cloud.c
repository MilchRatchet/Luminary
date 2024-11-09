#include "device_cloud.h"

#include "device.h"
#include "internal_error.h"
#include "texture.h"

#define CLOUD_SHAPE_RES 128
#define CLOUD_DETAIL_RES 32
#define CLOUD_WEATHER_RES 1024

LuminaryResult device_cloud_noise_create(CloudNoise** cloud_noise, Device* device) {
  __CHECK_NULL_ARGUMENT(cloud_noise);
  __CHECK_NULL_ARGUMENT(device);

  __FAILURE_HANDLE(host_malloc(cloud_noise, sizeof(CloudNoise)));

  Texture* shape_tex;
  __FAILURE_HANDLE(
    texture_create(&shape_tex, CLOUD_SHAPE_RES, CLOUD_SHAPE_RES, CLOUD_SHAPE_RES, CLOUD_SHAPE_RES, (void*) 0, TexDataUINT8, 4));

  Texture* detail_tex;
  __FAILURE_HANDLE(
    texture_create(&detail_tex, CLOUD_DETAIL_RES, CLOUD_DETAIL_RES, CLOUD_DETAIL_RES, CLOUD_DETAIL_RES, (void*) 0, TexDataUINT8, 4));

  Texture* weather_tex;
  __FAILURE_HANDLE(texture_create(&weather_tex, CLOUD_WEATHER_RES, CLOUD_WEATHER_RES, 1, CLOUD_WEATHER_RES, (void*) 0, TexDataUINT8, 4));

  __FAILURE_HANDLE(device_texture_create(&(*cloud_noise)->shape_tex, shape_tex, device->stream_main));
  __FAILURE_HANDLE(device_texture_create(&(*cloud_noise)->detail_tex, detail_tex, device->stream_main));
  __FAILURE_HANDLE(device_texture_create(&(*cloud_noise)->weather_tex, weather_tex, device->stream_main));

  __FAILURE_HANDLE(texture_destroy(&shape_tex));
  __FAILURE_HANDLE(texture_destroy(&detail_tex));
  __FAILURE_HANDLE(texture_destroy(&weather_tex));

  // TODO: Execute kernels with args

  __FAILURE_HANDLE(kernel_execute(device->cuda_kernels[CUDA_KERNEL_TYPE_CLOUD_COMPUTE_SHAPE_NOISE], device->stream_main));
  __FAILURE_HANDLE(kernel_execute(device->cuda_kernels[CUDA_KERNEL_TYPE_CLOUD_COMPUTE_DETAIL_NOISE], device->stream_main));
  __FAILURE_HANDLE(kernel_execute(device->cuda_kernels[CUDA_KERNEL_TYPE_CLOUD_COMPUTE_WEATHER_NOISE], device->stream_main));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_cloud_noise_destroy(CloudNoise** cloud_noise, Device* device) {
  __CHECK_NULL_ARGUMENT(cloud_noise);
  __CHECK_NULL_ARGUMENT(*cloud_noise);

  __FAILURE_HANDLE(device_texture_destroy(&(*cloud_noise)->shape_tex));
  __FAILURE_HANDLE(device_texture_destroy(&(*cloud_noise)->detail_tex));
  __FAILURE_HANDLE(device_texture_destroy(&(*cloud_noise)->weather_tex));

  __FAILURE_HANDLE(host_free(cloud_noise));

  return LUMINARY_SUCCESS;
}
