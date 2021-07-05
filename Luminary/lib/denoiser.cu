#include "denoiser.h"
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>


#define OPTIX_CHECK( call )                                             \
  {                                                                     \
    OptixResult res = call;                                             \
    if( res != OPTIX_SUCCESS )                                          \
      {                                                                 \
        fprintf( stderr, "Optix returned error %d in call (%s) (line %d)\n", res, #call, __LINE__ ); \
        system("pause");                                                \
        exit(-1);                                                       \
      }                                                                 \
  }

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
  inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
  {
     if (code != cudaSuccess)
     {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) {
          system("pause");
          exit(code);
        }
     }
  }

extern "C" void denoise_with_optix(raytrace_instance* instance) {

  OPTIX_CHECK(optixInit());

  OptixDeviceContext ctx;
  OPTIX_CHECK(optixDeviceContextCreate((CUcontext)0,(OptixDeviceContextOptions*)0, &ctx));

  OptixDenoiser denoiser;

  OptixDenoiserOptions opt;
  opt.inputKind = OPTIX_DENOISER_INPUT_RGB_ALBEDO;

  OPTIX_CHECK(optixDenoiserCreate(ctx, &opt, &denoiser));
  OPTIX_CHECK(optixDenoiserSetModel(denoiser, OPTIX_DENOISER_MODEL_KIND_HDR, nullptr, 0));

  OptixDenoiserSizes denoiserReturnSizes;
  OPTIX_CHECK(optixDenoiserComputeMemoryResources(denoiser, instance->width, instance->height, &denoiserReturnSizes));

  CUdeviceptr denoiserState;
  gpuErrchk(cudaMalloc((void**) &denoiserState, denoiserReturnSizes.stateSizeInBytes));

  const size_t scratchSize = (denoiserReturnSizes.withoutOverlapScratchSizeInBytes > denoiserReturnSizes.withOverlapScratchSizeInBytes) ?
                          denoiserReturnSizes.withoutOverlapScratchSizeInBytes :
                          denoiserReturnSizes.withOverlapScratchSizeInBytes;

  CUdeviceptr denoiserScratch;
  gpuErrchk(cudaMalloc((void**) &denoiserScratch, scratchSize));


  OPTIX_CHECK(optixDenoiserSetup(denoiser, 0,
    instance->width, instance->height,
    denoiserState,
    denoiserReturnSizes.stateSizeInBytes,
    denoiserScratch,
    scratchSize));

  gpuErrchk(cudaMemcpy(instance->frame_output_gpu, instance->frame_output, sizeof(RGBF) * instance->width * instance->height, cudaMemcpyHostToDevice));

  OptixImage2D inputLayer[2];

  inputLayer[0].data = (CUdeviceptr)instance->frame_output_gpu;
  inputLayer[0].width = instance->width;
  inputLayer[0].height = instance->height;
  inputLayer[0].rowStrideInBytes = instance->width * sizeof(RGBF);
  inputLayer[0].pixelStrideInBytes = sizeof(RGBF);
  inputLayer[0].format = OPTIX_PIXEL_FORMAT_FLOAT3;

  inputLayer[1].data = (CUdeviceptr)instance->albedo_buffer_gpu;
  inputLayer[1].width = instance->width;
  inputLayer[1].height = instance->height;
  inputLayer[1].rowStrideInBytes = instance->width * sizeof(RGBF);
  inputLayer[1].pixelStrideInBytes = sizeof(RGBF);
  inputLayer[1].format = OPTIX_PIXEL_FORMAT_FLOAT3;

  RGBF* output;
  gpuErrchk(cudaMalloc((void**) &output, sizeof(RGBF) * instance->width * instance->height));

  OptixImage2D outputLayer;

  outputLayer.data = (CUdeviceptr)output;
  outputLayer.width = instance->width;
  outputLayer.height = instance->height;
  outputLayer.rowStrideInBytes = instance->width * sizeof(RGBF);
  outputLayer.pixelStrideInBytes = sizeof(RGBF);
  outputLayer.format = OPTIX_PIXEL_FORMAT_FLOAT3;

  CUdeviceptr hdr_intensity;
  gpuErrchk(cudaMalloc((void**) &hdr_intensity, sizeof(float)));

  OPTIX_CHECK(optixDenoiserComputeIntensity(denoiser, 0, &inputLayer[0], hdr_intensity, denoiserScratch, scratchSize));

  CUdeviceptr avg_color;
  gpuErrchk(cudaMalloc((void**) &avg_color, sizeof(float) * 3));

  OPTIX_CHECK(optixDenoiserComputeAverageColor(denoiser, 0, &inputLayer[0], avg_color, denoiserScratch, scratchSize));

  OptixDenoiserParams denoiserParams;
  denoiserParams.denoiseAlpha = 0;
  denoiserParams.hdrIntensity = hdr_intensity;
  denoiserParams.blendFactor = 0.0f;
  denoiserParams.hdrAverageColor = avg_color;

  OPTIX_CHECK(optixDenoiserInvoke(denoiser,
    0,
    &denoiserParams,
    denoiserState,
    denoiserReturnSizes.stateSizeInBytes,
    &inputLayer[0],
    2,
    0,
    0,
    &outputLayer,
    denoiserScratch,
    scratchSize));


  gpuErrchk(cudaMemcpy(instance->frame_output, output, sizeof(RGBF) * instance->width * instance->height, cudaMemcpyDeviceToHost));

  OPTIX_CHECK(optixDeviceContextDestroy(ctx));
  OPTIX_CHECK(optixDenoiserDestroy(denoiser));

  gpuErrchk(cudaFree(output));
  gpuErrchk(cudaFree((void*)hdr_intensity));
  gpuErrchk(cudaFree((void*)avg_color));
  gpuErrchk(cudaFree((void*)denoiserState));
  gpuErrchk(cudaFree((void*)denoiserScratch));
}
