#ifndef CU_DENOISE_H
#define CU_DENOISE_H

#include <math.h>
#include <optix.h>
#include <optix_stubs.h>

#include "log.h"

struct realtime_denoise {
  OptixDeviceContext ctx;
  OptixDenoiser denoiser;
  OptixDenoiserOptions opt;
  OptixDenoiserSizes denoiserReturnSizes;
  DeviceBuffer* denoiserState;
  DeviceBuffer* denoiserScratch;
  OptixImage2D inputLayer[2];
  OptixImage2D outputLayer;
  DeviceBuffer* hdr_intensity;
  DeviceBuffer* avg_color;
  DeviceBuffer* output;
} typedef realtime_denoise;

extern "C" void* initialize_optix_denoise_for_realtime(RaytraceInstance* instance) {
  if (!instance->realtime) {
    log_message("Tried to initialize realtime denoiser in offline mode.");
    return (void*) 0;
  }
  OPTIX_CHECK(optixInit());

  realtime_denoise* denoise_setup = (realtime_denoise*) malloc(sizeof(realtime_denoise));

  OPTIX_CHECK(optixDeviceContextCreate((CUcontext) 0, (OptixDeviceContextOptions*) 0, &denoise_setup->ctx));

  denoise_setup->opt.inputKind = OPTIX_DENOISER_INPUT_RGB_ALBEDO;

  OPTIX_CHECK(optixDenoiserCreate(denoise_setup->ctx, &denoise_setup->opt, &denoise_setup->denoiser));
  OPTIX_CHECK(optixDenoiserSetModel(denoise_setup->denoiser, OPTIX_DENOISER_MODEL_KIND_HDR, nullptr, 0));

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

  denoise_setup->inputLayer[0].data               = (CUdeviceptr) device_buffer_get_pointer(instance->frame_output);
  denoise_setup->inputLayer[0].width              = instance->width;
  denoise_setup->inputLayer[0].height             = instance->height;
  denoise_setup->inputLayer[0].rowStrideInBytes   = instance->width * sizeof(RGBF);
  denoise_setup->inputLayer[0].pixelStrideInBytes = sizeof(RGBF);
  denoise_setup->inputLayer[0].format             = OPTIX_PIXEL_FORMAT_FLOAT3;

  denoise_setup->inputLayer[1].data               = (CUdeviceptr) device_buffer_get_pointer(instance->albedo_buffer);
  denoise_setup->inputLayer[1].width              = instance->width;
  denoise_setup->inputLayer[1].height             = instance->height;
  denoise_setup->inputLayer[1].rowStrideInBytes   = instance->width * sizeof(RGBF);
  denoise_setup->inputLayer[1].pixelStrideInBytes = sizeof(RGBF);
  denoise_setup->inputLayer[1].format             = OPTIX_PIXEL_FORMAT_FLOAT3;

  device_buffer_init(&denoise_setup->output);
  device_buffer_malloc(denoise_setup->output, sizeof(RGBF), instance->width * instance->height);

  denoise_setup->outputLayer.data               = (CUdeviceptr) device_buffer_get_pointer(denoise_setup->output);
  denoise_setup->outputLayer.width              = instance->width;
  denoise_setup->outputLayer.height             = instance->height;
  denoise_setup->outputLayer.rowStrideInBytes   = instance->width * sizeof(RGBF);
  denoise_setup->outputLayer.pixelStrideInBytes = sizeof(RGBF);
  denoise_setup->outputLayer.format             = OPTIX_PIXEL_FORMAT_FLOAT3;

  device_buffer_init(&denoise_setup->hdr_intensity);
  device_buffer_malloc(denoise_setup->hdr_intensity, sizeof(float), 1);

  device_buffer_init(&denoise_setup->avg_color);
  device_buffer_malloc(denoise_setup->avg_color, sizeof(float), 3);

  return denoise_setup;
}

extern "C" RGBF* denoise_with_optix_realtime(void* input) {
  if (!input) {
    crash_message("Denoise Setup is NULL");
  }

  realtime_denoise* denoise_setup = (realtime_denoise*) input;

  OPTIX_CHECK(optixDenoiserComputeIntensity(
    denoise_setup->denoiser, 0, &denoise_setup->inputLayer[0], (CUdeviceptr) device_buffer_get_pointer(denoise_setup->hdr_intensity),
    (CUdeviceptr) device_buffer_get_pointer(denoise_setup->denoiserScratch), device_buffer_get_size(denoise_setup->denoiserScratch)));

  OPTIX_CHECK(optixDenoiserComputeAverageColor(
    denoise_setup->denoiser, 0, &denoise_setup->inputLayer[0], (CUdeviceptr) device_buffer_get_pointer(denoise_setup->avg_color),
    (CUdeviceptr) device_buffer_get_pointer(denoise_setup->denoiserScratch), device_buffer_get_size(denoise_setup->denoiserScratch)));

  OptixDenoiserParams denoiserParams;
  denoiserParams.denoiseAlpha    = 0;
  denoiserParams.hdrIntensity    = (CUdeviceptr) device_buffer_get_pointer(denoise_setup->hdr_intensity);
  denoiserParams.blendFactor     = 0.0f;
  denoiserParams.hdrAverageColor = (CUdeviceptr) device_buffer_get_pointer(denoise_setup->avg_color);

  OPTIX_CHECK(optixDenoiserInvoke(
    denoise_setup->denoiser, 0, &denoiserParams, (CUdeviceptr) device_buffer_get_pointer(denoise_setup->denoiserState),
    device_buffer_get_size(denoise_setup->denoiserState), &denoise_setup->inputLayer[0], 2, 0, 0, &denoise_setup->outputLayer,
    (CUdeviceptr) device_buffer_get_pointer(denoise_setup->denoiserScratch), device_buffer_get_size(denoise_setup->denoiserScratch)));

  return (RGBF*) device_buffer_get_pointer(denoise_setup->output);
}

extern "C" float get_auto_exposure_from_optix(void* input, RaytraceInstance* instance) {
  if (!input) {
    return instance->scene_gpu.camera.exposure;
  }

  realtime_denoise denoise_setup = *(realtime_denoise*) input;
  const float exposure           = instance->scene_gpu.camera.exposure;

  float target_exposure = 1.0f;

  switch (instance->scene_gpu.camera.tonemap) {
    case TONEMAP_NONE:
      target_exposure = 0.75f;
      break;
    case TONEMAP_ACES:
      target_exposure = 1.75f;
      break;
    case TONEMAP_REINHARD:
      target_exposure = 0.75f;
      break;
    case TONEMAP_UNCHARTED2:
      target_exposure = 0.75f;
      break;
  }

  float brightness;
  device_buffer_download_full(denoise_setup.hdr_intensity, &brightness);

  return 0.8f * exposure + 0.2f * target_exposure * log2f(1.0f + brightness);
}

extern "C" void free_realtime_denoise(RaytraceInstance* instance, void* input) {
  if (!input) {
    log_message("Realtime denoise setup is NULL.");
    return;
  }

  realtime_denoise denoise_setup = *(realtime_denoise*) input;

  device_buffer_destroy(denoise_setup.output);
  device_buffer_destroy(denoise_setup.hdr_intensity);
  device_buffer_destroy(denoise_setup.avg_color);
  device_buffer_destroy(denoise_setup.denoiserState);
  device_buffer_destroy(denoise_setup.denoiserScratch);

  OPTIX_CHECK(optixDeviceContextDestroy(denoise_setup.ctx));
  OPTIX_CHECK(optixDenoiserDestroy(denoise_setup.denoiser));

  instance->denoise_setup = (void*) 0;
}

#endif /* CU_DENOISE_H */
