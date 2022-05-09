#ifndef CU_DENOISE_H
#define CU_DENOISE_H

#include <math.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include "log.h"

struct OptixDenoiseInstance {
  OptixDeviceContext ctx;
  OptixDenoiser denoiser;
  OptixDenoiserOptions opt;
  OptixDenoiserSizes denoiserReturnSizes;
  DeviceBuffer* denoiserState;
  DeviceBuffer* denoiserScratch;
  OptixDenoiserLayer layer;
  OptixDenoiserGuideLayer guide_layer;
  DeviceBuffer* hdr_intensity;
  DeviceBuffer* avg_color;
  DeviceBuffer* output;
} typedef OptixDenoiseInstance;

extern "C" void optix_denoise_create(RaytraceInstance* instance) {
  OPTIX_CHECK(optixInit());

  if (!instance) {
    log_message("Raytrace Instance is NULL.");
    instance->denoise_setup = (void*) 0;
    return;
  }

  OptixDenoiseInstance* denoise_setup = (OptixDenoiseInstance*) malloc(sizeof(OptixDenoiseInstance));

  OPTIX_CHECK(optixDeviceContextCreate((CUcontext) 0, (OptixDeviceContextOptions*) 0, &denoise_setup->ctx));

  denoise_setup->opt.guideAlbedo = 1;
  denoise_setup->opt.guideNormal = 0;

  OptixDenoiserModelKind kind = OPTIX_DENOISER_MODEL_KIND_HDR;

  OPTIX_CHECK(optixDenoiserCreate(denoise_setup->ctx, kind, &denoise_setup->opt, &denoise_setup->denoiser));

  OPTIX_CHECK(
    optixDenoiserComputeMemoryResources(denoise_setup->denoiser, instance->width, instance->height, &denoise_setup->denoiserReturnSizes));

  device_buffer_init(&denoise_setup->denoiserState);
  device_buffer_malloc(denoise_setup->denoiserState, denoise_setup->denoiserReturnSizes.stateSizeInBytes, 1);

  const size_t scratchSize =
    (denoise_setup->denoiserReturnSizes.withoutOverlapScratchSizeInBytes > denoise_setup->denoiserReturnSizes.withOverlapScratchSizeInBytes)
      ? denoise_setup->denoiserReturnSizes.withoutOverlapScratchSizeInBytes
      : denoise_setup->denoiserReturnSizes.withOverlapScratchSizeInBytes;

  device_buffer_init(&denoise_setup->denoiserScratch);
  device_buffer_malloc(denoise_setup->denoiserScratch, scratchSize, 1);

  OPTIX_CHECK(optixDenoiserSetup(
    denoise_setup->denoiser, 0, instance->width, instance->height, (CUdeviceptr) device_buffer_get_pointer(denoise_setup->denoiserState),
    device_buffer_get_size(denoise_setup->denoiserState), (CUdeviceptr) device_buffer_get_pointer(denoise_setup->denoiserScratch),
    device_buffer_get_size(denoise_setup->denoiserScratch)));

  denoise_setup->layer.input.data               = (CUdeviceptr) 0;
  denoise_setup->layer.input.width              = instance->width;
  denoise_setup->layer.input.height             = instance->height;
  denoise_setup->layer.input.rowStrideInBytes   = instance->width * sizeof(RGBAhalf);
  denoise_setup->layer.input.pixelStrideInBytes = sizeof(RGBAhalf);
  denoise_setup->layer.input.format             = OPTIX_PIXEL_FORMAT_HALF3;

  denoise_setup->layer.output.data               = (CUdeviceptr) 0;
  denoise_setup->layer.output.width              = instance->width;
  denoise_setup->layer.output.height             = instance->height;
  denoise_setup->layer.output.rowStrideInBytes   = instance->width * sizeof(RGBAhalf);
  denoise_setup->layer.output.pixelStrideInBytes = sizeof(RGBAhalf);
  denoise_setup->layer.output.format             = OPTIX_PIXEL_FORMAT_HALF3;

  denoise_setup->guide_layer.albedo.data               = (CUdeviceptr) device_buffer_get_pointer(instance->albedo_buffer);
  denoise_setup->guide_layer.albedo.width              = instance->width;
  denoise_setup->guide_layer.albedo.height             = instance->height;
  denoise_setup->guide_layer.albedo.rowStrideInBytes   = instance->width * sizeof(RGBAhalf);
  denoise_setup->guide_layer.albedo.pixelStrideInBytes = sizeof(RGBAhalf);
  denoise_setup->guide_layer.albedo.format             = OPTIX_PIXEL_FORMAT_HALF3;

  device_buffer_init(&denoise_setup->hdr_intensity);
  device_buffer_malloc(denoise_setup->hdr_intensity, sizeof(float), 1);

  device_buffer_init(&denoise_setup->avg_color);
  device_buffer_malloc(denoise_setup->avg_color, sizeof(float), 3);

  device_buffer_init(&denoise_setup->output);
  device_buffer_malloc(denoise_setup->output, sizeof(RGBAhalf), denoise_setup->layer.input.width * denoise_setup->layer.input.height);

  instance->denoise_setup = denoise_setup;
}

extern "C" DeviceBuffer* optix_denoise_apply(RaytraceInstance* instance, RGBF* src) {
  if (!instance) {
    log_message("Raytrace Instance is NULL.");
    return (DeviceBuffer*) 0;
  }

  if (!instance->denoise_setup) {
    log_message("OptiX Denoise Instance is NULL.");
    return (DeviceBuffer*) 0;
  }

  if (!src) {
    crash_message("Source pointer is NULL.");
    return (DeviceBuffer*) 0;
  }

  OptixDenoiseInstance* denoise_setup = (OptixDenoiseInstance*) instance->denoise_setup;

  denoise_setup->layer.input.data  = (CUdeviceptr) src;
  denoise_setup->layer.output.data = (CUdeviceptr) device_buffer_get_pointer(denoise_setup->output);

  OPTIX_CHECK(optixDenoiserComputeIntensity(
    denoise_setup->denoiser, 0, &(denoise_setup->layer.input), (CUdeviceptr) device_buffer_get_pointer(denoise_setup->hdr_intensity),
    (CUdeviceptr) device_buffer_get_pointer(denoise_setup->denoiserScratch), device_buffer_get_size(denoise_setup->denoiserScratch)));

  OPTIX_CHECK(optixDenoiserComputeAverageColor(
    denoise_setup->denoiser, 0, &(denoise_setup->layer.input), (CUdeviceptr) device_buffer_get_pointer(denoise_setup->avg_color),
    (CUdeviceptr) device_buffer_get_pointer(denoise_setup->denoiserScratch), device_buffer_get_size(denoise_setup->denoiserScratch)));

  OptixDenoiserParams denoiserParams;
  denoiserParams.denoiseAlpha    = 0;
  denoiserParams.hdrIntensity    = (CUdeviceptr) device_buffer_get_pointer(denoise_setup->hdr_intensity);
  denoiserParams.blendFactor     = 0.0f;
  denoiserParams.hdrAverageColor = (CUdeviceptr) device_buffer_get_pointer(denoise_setup->avg_color);

  OPTIX_CHECK(optixDenoiserInvoke(
    denoise_setup->denoiser, 0, &denoiserParams, (CUdeviceptr) device_buffer_get_pointer(denoise_setup->denoiserState),
    device_buffer_get_size(denoise_setup->denoiserState), &(denoise_setup->guide_layer), &(denoise_setup->layer), 1, 0, 0,
    (CUdeviceptr) device_buffer_get_pointer(denoise_setup->denoiserScratch), device_buffer_get_size(denoise_setup->denoiserScratch)));

  return denoise_setup->output;
}

extern "C" float optix_denoise_auto_exposure(RaytraceInstance* instance) {
  if (!instance) {
    log_message("Raytrace Instance is NULL.");
    return 0.0f;
  }

  const float exposure = instance->scene_gpu.camera.exposure;

  if (!instance->denoise_setup) {
    log_message("OptiX Denoise Instance is NULL.");
    return exposure;
  }

  if (instance->shading_mode != SHADING_DEFAULT)
    return 1.0f;

  OptixDenoiseInstance denoise_setup = *(OptixDenoiseInstance*) instance->denoise_setup;

  float target_exposure = 1.0f;

  switch (instance->scene_gpu.camera.tonemap) {
    case TONEMAP_NONE:
      target_exposure = 3.0f;
      break;
    case TONEMAP_ACES:
      target_exposure = 6.0f;
      break;
    case TONEMAP_REINHARD:
      target_exposure = 3.5f;
      break;
    case TONEMAP_UNCHARTED2:
      target_exposure = 3.5f;
      break;
  }

  float brightness;
  device_buffer_download_full(denoise_setup.hdr_intensity, &brightness);

  if (isnan(brightness) || isinf(brightness) || brightness < 0.0f)
    return exposure;

  const float lerp_factor = 0.2f * (1.0f - 1.0f / (1 + instance->temporal_frames));

  return lerp(exposure, target_exposure * log2f(1.0f + brightness), lerp_factor);
}

extern "C" void optix_denoise_free(RaytraceInstance* instance) {
  if (!instance) {
    log_message("Raytrace Instance is NULL.");
    return;
  }

  if (!instance->denoise_setup) {
    log_message("OptiX Denoise Instance is NULL.");
    return;
  }

  OptixDenoiseInstance denoise_setup = *(OptixDenoiseInstance*) instance->denoise_setup;

  device_buffer_destroy(denoise_setup.hdr_intensity);
  device_buffer_destroy(denoise_setup.avg_color);
  device_buffer_destroy(denoise_setup.denoiserState);
  device_buffer_destroy(denoise_setup.denoiserScratch);

  OPTIX_CHECK(optixDeviceContextDestroy(denoise_setup.ctx));
  OPTIX_CHECK(optixDenoiserDestroy(denoise_setup.denoiser));

  free(instance->denoise_setup);

  instance->denoise_setup = (void*) 0;
}

#endif /* CU_DENOISE_H */
