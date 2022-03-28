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
  opt.guideAlbedo = 1;
  opt.guideNormal = 0;

  OptixDenoiserModelKind kind = OPTIX_DENOISER_MODEL_KIND_HDR;

  OPTIX_CHECK(optixDenoiserCreate(ctx, kind, &opt, &denoiser));

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

  DeviceBuffer* output;
  device_buffer_init(&output);
  device_buffer_malloc(output, sizeof(RGBF), instance->width * instance->height);

  OptixDenoiserLayer layer;
  layer.input.data                = (CUdeviceptr) device_buffer_get_pointer(instance->frame_output);
  layer.input.width               = instance->width;
  layer.input.height              = instance->height;
  layer.input.rowStrideInBytes    = instance->width * sizeof(RGBF);
  layer.input.pixelStrideInBytes  = sizeof(RGBF);
  layer.input.format              = OPTIX_PIXEL_FORMAT_FLOAT3;
  layer.output.data               = (CUdeviceptr) device_buffer_get_pointer(output);
  layer.output.width              = instance->width;
  layer.output.height             = instance->height;
  layer.output.rowStrideInBytes   = instance->width * sizeof(RGBF);
  layer.output.pixelStrideInBytes = sizeof(RGBF);
  layer.output.format             = OPTIX_PIXEL_FORMAT_FLOAT3;

  OptixDenoiserGuideLayer guide_layer;
  guide_layer.albedo.data               = (CUdeviceptr) device_buffer_get_pointer(instance->albedo_buffer);
  guide_layer.albedo.width              = instance->width;
  guide_layer.albedo.height             = instance->height;
  guide_layer.albedo.rowStrideInBytes   = instance->width * sizeof(RGBF);
  guide_layer.albedo.pixelStrideInBytes = sizeof(RGBF);
  guide_layer.albedo.format             = OPTIX_PIXEL_FORMAT_FLOAT3;

  DeviceBuffer* hdr_intensity;
  device_buffer_init(&hdr_intensity);
  device_buffer_malloc(hdr_intensity, sizeof(float), 1);

  OPTIX_CHECK(optixDenoiserComputeIntensity(
    denoiser, 0, &(layer.input), (CUdeviceptr) device_buffer_get_pointer(hdr_intensity),
    (CUdeviceptr) device_buffer_get_pointer(denoiserScratch), device_buffer_get_size(denoiserScratch)));

  DeviceBuffer* avg_color;
  device_buffer_init(&avg_color);
  device_buffer_malloc(avg_color, sizeof(float), 3);

  OPTIX_CHECK(optixDenoiserComputeAverageColor(
    denoiser, 0, &(layer.input), (CUdeviceptr) device_buffer_get_pointer(avg_color),
    (CUdeviceptr) device_buffer_get_pointer(denoiserScratch), device_buffer_get_size(denoiserScratch)));

  OptixDenoiserParams denoiserParams;
  denoiserParams.denoiseAlpha    = 0;
  denoiserParams.hdrIntensity    = (CUdeviceptr) device_buffer_get_pointer(hdr_intensity);
  denoiserParams.blendFactor     = 0.0f;
  denoiserParams.hdrAverageColor = (CUdeviceptr) device_buffer_get_pointer(avg_color);

  OPTIX_CHECK(optixDenoiserInvoke(
    denoiser, 0, &denoiserParams, (CUdeviceptr) device_buffer_get_pointer(denoiserState), device_buffer_get_size(denoiserState),
    &guide_layer, &layer, 1, 0, 0, (CUdeviceptr) device_buffer_get_pointer(denoiserScratch), device_buffer_get_size(denoiserScratch)));

  device_buffer_copy(output, instance->frame_output);

  gpuErrchk(cudaDeviceSynchronize());

  device_buffer_destroy(output);
  device_buffer_destroy(hdr_intensity);
  device_buffer_destroy(avg_color);
  device_buffer_destroy(denoiserState);
  device_buffer_destroy(denoiserScratch);

  OPTIX_CHECK(optixDeviceContextDestroy(ctx));
  OPTIX_CHECK(optixDenoiserDestroy(denoiser));

  bench_toc((char*) "Optix Denoiser");
}
