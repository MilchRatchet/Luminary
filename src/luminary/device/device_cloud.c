#include "device_cloud.h"

#include "cloud.h"
#include "device.h"
#include "internal_error.h"
#include "kernel_args.h"
#include "texture.h"

#define CLOUD_SHAPE_RES 128
#define CLOUD_DETAIL_RES 32
#define CLOUD_WEATHER_RES 1024

static LuminaryResult _device_cloud_noise_generate_weather_map(DeviceCloudNoise* cloud_noise, Device* device) {
  __CHECK_NULL_ARGUMENT(cloud_noise);
  __CHECK_NULL_ARGUMENT(device);

  KernelArgsCloudComputeWeatherNoise weather_args;
  weather_args.dim  = CLOUD_WEATHER_RES;
  weather_args.seed = cloud_noise->seed;
  weather_args.tex  = DEVICE_PTR(cloud_noise->weather_tex->cuda_memory);
  weather_args.ld   = cloud_noise->weather_tex->pitch / (sizeof(uint8_t) * 4);

  __FAILURE_HANDLE(
    kernel_execute_with_args(device->cuda_kernels[CUDA_KERNEL_TYPE_CLOUD_COMPUTE_WEATHER_NOISE], &weather_args, device->stream_main));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_cloud_noise_create(DeviceCloudNoise** cloud_noise, Device* device) {
  __CHECK_NULL_ARGUMENT(cloud_noise);
  __CHECK_NULL_ARGUMENT(device);

  __FAILURE_HANDLE(host_malloc(cloud_noise, sizeof(DeviceCloudNoise)));
  memset(*cloud_noise, 0, sizeof(DeviceCloudNoise));

  Texture* shape_tex;
  __FAILURE_HANDLE(texture_create(&shape_tex));
  __FAILURE_HANDLE(texture_fill(shape_tex, CLOUD_SHAPE_RES, CLOUD_SHAPE_RES, CLOUD_SHAPE_RES, (void*) 0, TEXTURE_DATA_TYPE_U8, 4));

  Texture* detail_tex;
  __FAILURE_HANDLE(texture_create(&detail_tex));
  __FAILURE_HANDLE(texture_fill(detail_tex, CLOUD_DETAIL_RES, CLOUD_DETAIL_RES, CLOUD_DETAIL_RES, (void*) 0, TEXTURE_DATA_TYPE_U8, 4));

  Texture* weather_tex;
  __FAILURE_HANDLE(texture_create(&weather_tex));
  __FAILURE_HANDLE(texture_fill(weather_tex, CLOUD_WEATHER_RES, CLOUD_WEATHER_RES, 1, (void*) 0, TEXTURE_DATA_TYPE_U8, 4));

  __FAILURE_HANDLE(device_texture_create(&(*cloud_noise)->shape_tex, shape_tex, device->stream_main));
  __FAILURE_HANDLE(device_texture_create(&(*cloud_noise)->detail_tex, detail_tex, device->stream_main));
  __FAILURE_HANDLE(device_texture_create(&(*cloud_noise)->weather_tex, weather_tex, device->stream_main));

  __FAILURE_HANDLE(texture_destroy(&shape_tex));
  __FAILURE_HANDLE(texture_destroy(&detail_tex));
  __FAILURE_HANDLE(texture_destroy(&weather_tex));

  (*cloud_noise)->initialized = false;
  (*cloud_noise)->seed        = CLOUD_DEFAULT_SEED;

  return LUMINARY_SUCCESS;
}

LuminaryResult device_cloud_noise_generate(DeviceCloudNoise* cloud_noise, const Cloud* cloud, Device* device) {
  __CHECK_NULL_ARGUMENT(cloud_noise);
  __CHECK_NULL_ARGUMENT(cloud);
  __CHECK_NULL_ARGUMENT(device);

  if (!cloud_noise->initialized) {
    DEVICE uint8_t* shape_noise_data;
    __FAILURE_HANDLE(device_malloc(&shape_noise_data, CLOUD_SHAPE_RES * CLOUD_SHAPE_RES * CLOUD_SHAPE_RES * sizeof(uint8_t) * 4));

    DEVICE uint16_t* detail_noise_data;
    __FAILURE_HANDLE(device_malloc(&detail_noise_data, CLOUD_DETAIL_RES * CLOUD_DETAIL_RES * CLOUD_DETAIL_RES * sizeof(uint8_t) * 4));

    KernelArgsCloudComputeShapeNoise shape_args;
    shape_args.dim = CLOUD_SHAPE_RES;
    shape_args.tex = DEVICE_PTR(shape_noise_data);

    __FAILURE_HANDLE(
      kernel_execute_with_args(device->cuda_kernels[CUDA_KERNEL_TYPE_CLOUD_COMPUTE_SHAPE_NOISE], &shape_args, device->stream_main));

    KernelArgsCloudComputeDetailNoise detail_args;
    detail_args.dim = CLOUD_DETAIL_RES;
    detail_args.tex = DEVICE_PTR(detail_noise_data);

    __FAILURE_HANDLE(
      kernel_execute_with_args(device->cuda_kernels[CUDA_KERNEL_TYPE_CLOUD_COMPUTE_DETAIL_NOISE], &detail_args, device->stream_main));

    __FAILURE_HANDLE(device_texture_copy_from_mem(cloud_noise->shape_tex, shape_noise_data, device->stream_main));
    __FAILURE_HANDLE(device_texture_copy_from_mem(cloud_noise->detail_tex, detail_noise_data, device->stream_main));

    __FAILURE_HANDLE(device_free(&shape_noise_data));
    __FAILURE_HANDLE(device_free(&detail_noise_data));
  }

  if (!cloud_noise->initialized || (cloud_noise->seed != cloud->seed)) {
    cloud_noise->seed = cloud->seed;

    __FAILURE_HANDLE(_device_cloud_noise_generate_weather_map(cloud_noise, device));
  }

  cloud_noise->initialized = true;

  return LUMINARY_SUCCESS;
}

LuminaryResult device_cloud_noise_destroy(DeviceCloudNoise** cloud_noise) {
  __CHECK_NULL_ARGUMENT(cloud_noise);
  __CHECK_NULL_ARGUMENT(*cloud_noise);

  __FAILURE_HANDLE(device_texture_destroy(&(*cloud_noise)->shape_tex));
  __FAILURE_HANDLE(device_texture_destroy(&(*cloud_noise)->detail_tex));
  __FAILURE_HANDLE(device_texture_destroy(&(*cloud_noise)->weather_tex));

  __FAILURE_HANDLE(host_free(cloud_noise));

  return LUMINARY_SUCCESS;
}
