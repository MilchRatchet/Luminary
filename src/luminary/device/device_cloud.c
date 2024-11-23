#include "device_cloud.h"

#include "device.h"
#include "internal_error.h"
#include "kernel_args.h"
#include "texture.h"

#define CLOUD_SHAPE_RES 128
#define CLOUD_DETAIL_RES 32
#define CLOUD_WEATHER_RES 1024

LuminaryResult device_cloud_noise_create(CloudNoise** cloud_noise, Cloud* cloud, Device* device) {
  __CHECK_NULL_ARGUMENT(cloud_noise);
  __CHECK_NULL_ARGUMENT(device);

  __FAILURE_HANDLE(host_malloc(cloud_noise, sizeof(CloudNoise)));

  Texture* shape_tex;
  __FAILURE_HANDLE(texture_create(&shape_tex, CLOUD_SHAPE_RES, CLOUD_SHAPE_RES, CLOUD_SHAPE_RES, (void*) 0, TexDataUINT8, 4));

  Texture* detail_tex;
  __FAILURE_HANDLE(texture_create(&detail_tex, CLOUD_DETAIL_RES, CLOUD_DETAIL_RES, CLOUD_DETAIL_RES, (void*) 0, TexDataUINT8, 4));

  Texture* weather_tex;
  __FAILURE_HANDLE(texture_create(&weather_tex, CLOUD_WEATHER_RES, CLOUD_WEATHER_RES, 1, (void*) 0, TexDataUINT8, 4));

  __FAILURE_HANDLE(device_texture_create(&(*cloud_noise)->shape_tex, shape_tex, device->stream_main));
  __FAILURE_HANDLE(device_texture_create(&(*cloud_noise)->detail_tex, detail_tex, device->stream_main));
  __FAILURE_HANDLE(device_texture_create(&(*cloud_noise)->weather_tex, weather_tex, device->stream_main));

  __FAILURE_HANDLE(texture_destroy(&shape_tex));
  __FAILURE_HANDLE(texture_destroy(&detail_tex));
  __FAILURE_HANDLE(texture_destroy(&weather_tex));

  KernelArgsCloudComputeShapeNoise shape_args;
  shape_args.dim = CLOUD_SHAPE_RES;
  shape_args.tex = (*cloud_noise)->shape_tex->memory;

  __FAILURE_HANDLE(
    kernel_execute_with_args(device->cuda_kernels[CUDA_KERNEL_TYPE_CLOUD_COMPUTE_SHAPE_NOISE], &shape_args, device->stream_main));

  KernelArgsCloudComputeDetailNoise detail_args;
  detail_args.dim = CLOUD_DETAIL_RES;
  detail_args.tex = (*cloud_noise)->detail_tex->memory;

  __FAILURE_HANDLE(
    kernel_execute_with_args(device->cuda_kernels[CUDA_KERNEL_TYPE_CLOUD_COMPUTE_DETAIL_NOISE], &detail_args, device->stream_main));

  KernelArgsCloudComputeWeatherNoise weather_args;
  weather_args.dim  = CLOUD_WEATHER_RES;
  weather_args.seed = cloud->seed;
  weather_args.tex  = (*cloud_noise)->weather_tex->memory;

  __FAILURE_HANDLE(
    kernel_execute_with_args(device->cuda_kernels[CUDA_KERNEL_TYPE_CLOUD_COMPUTE_WEATHER_NOISE], &weather_args, device->stream_main));

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
