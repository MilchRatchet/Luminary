#include <cuda_runtime.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include "bench.h"
#include "buffer.h"
#include "denoiser.h"
#include "error.h"
#include "log.h"

#define OPTIX_CHECK(call)                                                \
  {                                                                      \
    OptixResult res = call;                                              \
                                                                         \
    if (res != OPTIX_SUCCESS) {                                          \
      crash_message("Optix returned error %d in call (%s)", res, #call); \
    }                                                                    \
  }

#define gpuErrchk(ans)                                         \
  {                                                            \
    if (ans != cudaSuccess) {                                  \
      crash_message("GPUassert: %s", cudaGetErrorString(ans)); \
    }                                                          \
  }

extern "C" void denoise_with_optix(RaytraceInstance* instance) {
  bench_tic();

  OPTIX_CHECK(optixInit());

  OptixDeviceContext ctx;
  OPTIX_CHECK(optixDeviceContextCreate((CUcontext) 0, (OptixDeviceContextOptions*) 0, &ctx));

  OptixDenoiser denoiser;

  OptixDenoiserOptions opt;
  opt.inputKind = OPTIX_DENOISER_INPUT_RGB_ALBEDO;

  OPTIX_CHECK(optixDenoiserCreate(ctx, &opt, &denoiser));
  OPTIX_CHECK(optixDenoiserSetModel(denoiser, OPTIX_DENOISER_MODEL_KIND_HDR, nullptr, 0));

  OptixDenoiserSizes denoiserReturnSizes;
  OPTIX_CHECK(optixDenoiserComputeMemoryResources(denoiser, instance->width, instance->height, &denoiserReturnSizes));

  DeviceBuffer* denoiserState;
  device_buffer_init(&denoiserState);
  device_buffer_malloc(denoiserState, denoiserReturnSizes.stateSizeInBytes, 1);

  const size_t scratchSize = (denoiserReturnSizes.withoutOverlapScratchSizeInBytes > denoiserReturnSizes.withOverlapScratchSizeInBytes)
                               ? denoiserReturnSizes.withoutOverlapScratchSizeInBytes
                               : denoiserReturnSizes.withOverlapScratchSizeInBytes;

  DeviceBuffer* denoiserScratch;
  device_buffer_init(&denoiserScratch);
  device_buffer_malloc(denoiserScratch, scratchSize, 1);

  OPTIX_CHECK(optixDenoiserSetup(
    denoiser, 0, instance->width, instance->height, (CUdeviceptr) device_buffer_get_pointer(denoiserState),
    device_buffer_get_size(denoiserState), (CUdeviceptr) device_buffer_get_pointer(denoiserScratch),
    device_buffer_get_size(denoiserScratch)));

  OptixImage2D inputLayer[2];

  inputLayer[0].data               = (CUdeviceptr) device_buffer_get_pointer(instance->frame_output);
  inputLayer[0].width              = instance->width;
  inputLayer[0].height             = instance->height;
  inputLayer[0].rowStrideInBytes   = instance->width * sizeof(RGBF);
  inputLayer[0].pixelStrideInBytes = sizeof(RGBF);
  inputLayer[0].format             = OPTIX_PIXEL_FORMAT_FLOAT3;

  inputLayer[1].data               = (CUdeviceptr) device_buffer_get_pointer(instance->albedo_buffer);
  inputLayer[1].width              = instance->width;
  inputLayer[1].height             = instance->height;
  inputLayer[1].rowStrideInBytes   = instance->width * sizeof(RGBF);
  inputLayer[1].pixelStrideInBytes = sizeof(RGBF);
  inputLayer[1].format             = OPTIX_PIXEL_FORMAT_FLOAT3;

  DeviceBuffer* output;
  device_buffer_init(&output);
  device_buffer_malloc(output, sizeof(RGBF), instance->width * instance->height);

  OptixImage2D outputLayer;

  outputLayer.data               = (CUdeviceptr) device_buffer_get_pointer(output);
  outputLayer.width              = instance->width;
  outputLayer.height             = instance->height;
  outputLayer.rowStrideInBytes   = instance->width * sizeof(RGBF);
  outputLayer.pixelStrideInBytes = sizeof(RGBF);
  outputLayer.format             = OPTIX_PIXEL_FORMAT_FLOAT3;

  DeviceBuffer* hdr_intensity;
  device_buffer_init(&hdr_intensity);
  device_buffer_malloc(hdr_intensity, sizeof(float), 1);

  OPTIX_CHECK(optixDenoiserComputeIntensity(
    denoiser, 0, &inputLayer[0], (CUdeviceptr) device_buffer_get_pointer(hdr_intensity),
    (CUdeviceptr) device_buffer_get_pointer(denoiserScratch), device_buffer_get_size(denoiserScratch)));

  DeviceBuffer* avg_color;
  device_buffer_init(&avg_color);
  device_buffer_malloc(avg_color, sizeof(float), 3);

  OPTIX_CHECK(optixDenoiserComputeAverageColor(
    denoiser, 0, &inputLayer[0], (CUdeviceptr) device_buffer_get_pointer(avg_color),
    (CUdeviceptr) device_buffer_get_pointer(denoiserScratch), device_buffer_get_size(denoiserScratch)));

  OptixDenoiserParams denoiserParams;
  denoiserParams.denoiseAlpha    = 0;
  denoiserParams.hdrIntensity    = (CUdeviceptr) device_buffer_get_pointer(hdr_intensity);
  denoiserParams.blendFactor     = 0.0f;
  denoiserParams.hdrAverageColor = (CUdeviceptr) device_buffer_get_pointer(avg_color);

  OPTIX_CHECK(optixDenoiserInvoke(
    denoiser, 0, &denoiserParams, (CUdeviceptr) device_buffer_get_pointer(denoiserState), device_buffer_get_size(denoiserState),
    &inputLayer[0], 2, 0, 0, &outputLayer, (CUdeviceptr) device_buffer_get_pointer(denoiserScratch),
    device_buffer_get_size(denoiserScratch)));

  device_buffer_copy(output, instance->frame_output);

  gpuErrchk(cudaDeviceSynchronize());

  device_buffer_destroy(output);
  device_buffer_destroy(hdr_intensity);
  device_buffer_destroy(avg_color);
  device_buffer_destroy(denoiserState);
  device_buffer_destroy(denoiserScratch);

  OPTIX_CHECK(optixDeviceContextDestroy(ctx));
  OPTIX_CHECK(optixDenoiserDestroy(denoiser));

  bench_toc("Optix Denoiser");
}
