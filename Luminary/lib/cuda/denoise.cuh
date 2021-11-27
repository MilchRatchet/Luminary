#ifndef CU_DENOISE_H
#define CU_DENOISE_H

#include <math.h>
#include <optix.h>
#include <optix_stubs.h>

struct realtime_denoise {
  OptixDeviceContext ctx;
  OptixDenoiser denoiser;
  OptixDenoiserOptions opt;
  OptixDenoiserSizes denoiserReturnSizes;
  CUdeviceptr denoiserState;
  CUdeviceptr denoiserScratch;
  OptixImage2D inputLayer[2];
  OptixImage2D outputLayer;
  CUdeviceptr hdr_intensity;
  CUdeviceptr avg_color;
} typedef realtime_denoise;

extern "C" void* initialize_optix_denoise_for_realtime(RaytraceInstance* instance) {
  OPTIX_CHECK(optixInit());

  realtime_denoise* denoise_setup = (realtime_denoise*) malloc(sizeof(realtime_denoise));

  OPTIX_CHECK(optixDeviceContextCreate((CUcontext) 0, (OptixDeviceContextOptions*) 0, &denoise_setup->ctx));

  denoise_setup->opt.inputKind = OPTIX_DENOISER_INPUT_RGB_ALBEDO;

  OPTIX_CHECK(optixDenoiserCreate(denoise_setup->ctx, &denoise_setup->opt, &denoise_setup->denoiser));
  OPTIX_CHECK(optixDenoiserSetModel(denoise_setup->denoiser, OPTIX_DENOISER_MODEL_KIND_HDR, nullptr, 0));

  OPTIX_CHECK(
    optixDenoiserComputeMemoryResources(denoise_setup->denoiser, instance->width, instance->height, &denoise_setup->denoiserReturnSizes));

  device_malloc((void**) &denoise_setup->denoiserState, denoise_setup->denoiserReturnSizes.stateSizeInBytes);

  const size_t scratchSize =
    (denoise_setup->denoiserReturnSizes.withoutOverlapScratchSizeInBytes > denoise_setup->denoiserReturnSizes.withOverlapScratchSizeInBytes)
      ? denoise_setup->denoiserReturnSizes.withoutOverlapScratchSizeInBytes
      : denoise_setup->denoiserReturnSizes.withOverlapScratchSizeInBytes;

  device_malloc((void**) &denoise_setup->denoiserScratch, scratchSize);

  OPTIX_CHECK(optixDenoiserSetup(
    denoise_setup->denoiser, 0, instance->width, instance->height, denoise_setup->denoiserState,
    denoise_setup->denoiserReturnSizes.stateSizeInBytes, denoise_setup->denoiserScratch, scratchSize));

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

  RGBF* output;
  device_malloc((void**) &output, sizeof(RGBF) * instance->width * instance->height);

  denoise_setup->outputLayer.data               = (CUdeviceptr) output;
  denoise_setup->outputLayer.width              = instance->width;
  denoise_setup->outputLayer.height             = instance->height;
  denoise_setup->outputLayer.rowStrideInBytes   = instance->width * sizeof(RGBF);
  denoise_setup->outputLayer.pixelStrideInBytes = sizeof(RGBF);
  denoise_setup->outputLayer.format             = OPTIX_PIXEL_FORMAT_FLOAT3;

  gpuErrchk(cudaMalloc((void**) &denoise_setup->hdr_intensity, sizeof(float)));

  gpuErrchk(cudaMalloc((void**) &denoise_setup->avg_color, sizeof(float) * 3));

  return denoise_setup;
}

extern "C" RGBF* denoise_with_optix_realtime(void* input) {
  realtime_denoise* denoise_setup = (realtime_denoise*) input;

  const size_t scratchSize =
    (denoise_setup->denoiserReturnSizes.withoutOverlapScratchSizeInBytes > denoise_setup->denoiserReturnSizes.withOverlapScratchSizeInBytes)
      ? denoise_setup->denoiserReturnSizes.withoutOverlapScratchSizeInBytes
      : denoise_setup->denoiserReturnSizes.withOverlapScratchSizeInBytes;

  OPTIX_CHECK(optixDenoiserComputeIntensity(
    denoise_setup->denoiser, 0, &denoise_setup->inputLayer[0], denoise_setup->hdr_intensity, denoise_setup->denoiserScratch, scratchSize));

  OPTIX_CHECK(optixDenoiserComputeAverageColor(
    denoise_setup->denoiser, 0, &denoise_setup->inputLayer[0], denoise_setup->avg_color, denoise_setup->denoiserScratch, scratchSize));

  OptixDenoiserParams denoiserParams;
  denoiserParams.denoiseAlpha    = 0;
  denoiserParams.hdrIntensity    = denoise_setup->hdr_intensity;
  denoiserParams.blendFactor     = 0.0f;
  denoiserParams.hdrAverageColor = denoise_setup->avg_color;

  OPTIX_CHECK(optixDenoiserInvoke(
    denoise_setup->denoiser, 0, &denoiserParams, denoise_setup->denoiserState, denoise_setup->denoiserReturnSizes.stateSizeInBytes,
    &denoise_setup->inputLayer[0], 2, 0, 0, &denoise_setup->outputLayer, denoise_setup->denoiserScratch, scratchSize));

  return (RGBF*) denoise_setup->outputLayer.data;
}

extern "C" float get_auto_exposure_from_optix(void* input, RaytraceInstance* instance) {
  realtime_denoise denoise_setup = *(realtime_denoise*) input;
  const float exposure           = instance->scene_gpu.camera.exposure;

  float target_exposure = 1.0f;

  switch (instance->scene_gpu.camera.tonemap) {
    case TONEMAP_NONE:
      target_exposure = 0.75f;
      break;
    case TONEMAP_ACES:
      target_exposure = 2.0f;
      break;
    case TONEMAP_REINHARD:
      target_exposure = 0.75f;
      break;
    case TONEMAP_UNCHARTED2:
      target_exposure = 0.75f;
      break;
  }

  float brightness;
  gpuErrchk(cudaMemcpy(&brightness, (void*) denoise_setup.hdr_intensity, sizeof(float), cudaMemcpyDeviceToHost));

  return 0.8f * exposure + 0.2f * target_exposure * log2f(1.0f + brightness);
}

extern "C" void free_realtime_denoise(void* input) {
  realtime_denoise denoise_setup = *(realtime_denoise*) input;

  OPTIX_CHECK(optixDeviceContextDestroy(denoise_setup.ctx));
  OPTIX_CHECK(optixDenoiserDestroy(denoise_setup.denoiser));

  gpuErrchk(cudaFree((void*) denoise_setup.outputLayer.data));
  gpuErrchk(cudaFree((void*) denoise_setup.hdr_intensity));
  gpuErrchk(cudaFree((void*) denoise_setup.avg_color));
  gpuErrchk(cudaFree((void*) denoise_setup.denoiserState));
  gpuErrchk(cudaFree((void*) denoise_setup.denoiserScratch));
}

#endif /* CU_DENOISE_H */
