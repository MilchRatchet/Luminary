#include <math.h>
#include <optix.h>
#include <optix_stubs.h>

#include "buffer.h"
#include "log.h"
#include "raytrace.h"
#include "utils.h"

struct OptixDenoiseInstance {
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

void denoise_create(RaytraceInstance* instance) {
  if (!instance) {
    log_message("Raytrace Instance is NULL.");
    instance->denoise_setup = (void*) 0;
    return;
  }

  OptixDenoiserModelKind kind;

  switch (instance->denoiser) {
    case DENOISING_ON:
      // We only need OPTIX_DENOISER_MODEL_KIND_HDR, however, a forum post by Nvidia
      // (https://forums.developer.nvidia.com/t/optix-8-0-denoiser-camera-space-vs-world-space/262875/2) suggests that these are legacy and
      // no longer being improved. This is not documented anywhere else but we use the AOV model instead then.
      kind = OPTIX_DENOISER_MODEL_KIND_AOV;
      break;
    case DENOISING_UPSCALING: {
      if (instance->width * instance->height > 18144000) {
        crash_message("Internal resolution is too high for denoising! The maximum is ~18144000 pixels.");
      }
      kind = OPTIX_DENOISER_MODEL_KIND_UPSCALE2X;
    } break;
    default:
      instance->denoiser = DENOISING_OFF;
      return;
  }

  OptixDenoiseInstance* denoise_setup = (OptixDenoiseInstance*) calloc(1, sizeof(OptixDenoiseInstance));

  denoise_setup->opt.guideAlbedo  = 1;
  denoise_setup->opt.guideNormal  = 1;
  denoise_setup->opt.denoiseAlpha = OPTIX_DENOISER_ALPHA_MODE_COPY;

  OPTIX_CHECK(optixDenoiserCreate(instance->optix_ctx, kind, &denoise_setup->opt, &denoise_setup->denoiser));

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
  denoise_setup->layer.input.rowStrideInBytes   = instance->width * sizeof(RGBF);
  denoise_setup->layer.input.pixelStrideInBytes = sizeof(RGBF);
  denoise_setup->layer.input.format             = OPTIX_PIXEL_FORMAT_FLOAT3;

  denoise_setup->layer.output.data               = (CUdeviceptr) 0;
  denoise_setup->layer.output.width              = instance->output_width;
  denoise_setup->layer.output.height             = instance->output_height;
  denoise_setup->layer.output.rowStrideInBytes   = instance->output_width * sizeof(RGBF);
  denoise_setup->layer.output.pixelStrideInBytes = sizeof(RGBF);
  denoise_setup->layer.output.format             = OPTIX_PIXEL_FORMAT_FLOAT3;

  denoise_setup->guide_layer.albedo.data               = (CUdeviceptr) device_buffer_get_pointer(instance->albedo_buffer);
  denoise_setup->guide_layer.albedo.width              = instance->width;
  denoise_setup->guide_layer.albedo.height             = instance->height;
  denoise_setup->guide_layer.albedo.rowStrideInBytes   = instance->width * sizeof(RGBF);
  denoise_setup->guide_layer.albedo.pixelStrideInBytes = sizeof(RGBF);
  denoise_setup->guide_layer.albedo.format             = OPTIX_PIXEL_FORMAT_FLOAT3;

  denoise_setup->guide_layer.normal.data               = (CUdeviceptr) device_buffer_get_pointer(instance->normal_buffer);
  denoise_setup->guide_layer.normal.width              = instance->width;
  denoise_setup->guide_layer.normal.height             = instance->height;
  denoise_setup->guide_layer.normal.rowStrideInBytes   = instance->width * sizeof(RGBF);
  denoise_setup->guide_layer.normal.pixelStrideInBytes = sizeof(RGBF);
  denoise_setup->guide_layer.normal.format             = OPTIX_PIXEL_FORMAT_FLOAT3;

  device_buffer_init(&denoise_setup->hdr_intensity);
  device_buffer_malloc(denoise_setup->hdr_intensity, sizeof(float), 1);

  device_buffer_init(&denoise_setup->avg_color);
  device_buffer_malloc(denoise_setup->avg_color, sizeof(float), 3);

  device_buffer_init(&denoise_setup->output);
  device_buffer_malloc(denoise_setup->output, sizeof(RGBF), denoise_setup->layer.output.width * denoise_setup->layer.output.height);

  instance->denoise_setup = denoise_setup;
}

DeviceBuffer* denoise_apply(RaytraceInstance* instance, RGBF* src) {
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
  denoiserParams.hdrIntensity    = (CUdeviceptr) device_buffer_get_pointer(denoise_setup->hdr_intensity);
  denoiserParams.blendFactor     = 0.0f;
  denoiserParams.hdrAverageColor = (CUdeviceptr) device_buffer_get_pointer(denoise_setup->avg_color);

  OPTIX_CHECK(optixDenoiserInvoke(
    denoise_setup->denoiser, 0, &denoiserParams, (CUdeviceptr) device_buffer_get_pointer(denoise_setup->denoiserState),
    device_buffer_get_size(denoise_setup->denoiserState), &(denoise_setup->guide_layer), &(denoise_setup->layer), 1, 0, 0,
    (CUdeviceptr) device_buffer_get_pointer(denoise_setup->denoiserScratch), device_buffer_get_size(denoise_setup->denoiserScratch)));

  return denoise_setup->output;
}

static float lerp(const float a, const float b, const float t) {
  return a + t * (b - a);
}

float denoise_auto_exposure(RaytraceInstance* instance) {
  if (!instance) {
    log_message("Raytrace Instance is NULL.");
    return 0.0f;
  }

  if (instance->shading_mode != SHADING_DEFAULT)
    return 1.0f;

  const float exposure = instance->scene.camera.exposure;

  if (!instance->denoise_setup) {
    return exposure;
  }

  OptixDenoiseInstance denoise_setup = *(OptixDenoiseInstance*) instance->denoise_setup;

  float target_exposure = 1.0f;

  switch (instance->scene.camera.tonemap) {
    case TONEMAP_NONE:
      target_exposure = 2.5f;
      break;
    case TONEMAP_ACES:
      target_exposure = 5.0f;
      break;
    case TONEMAP_REINHARD:
      target_exposure = 3.0f;
      break;
    case TONEMAP_UNCHARTED2:
      target_exposure = 3.0f;
      break;
    default:
      target_exposure = 1.0f;
      break;
  }

  float brightness;
  device_buffer_download_full(denoise_setup.hdr_intensity, &brightness);

  if (isnan(brightness) || isinf(brightness) || brightness < 0.0f)
    return exposure;

  const float lerp_factor = 0.2f * (1.0f - 1.0f / (1 + instance->temporal_frames));

  const float min_exposure = instance->scene.camera.min_exposure;
  const float max_exposure = instance->scene.camera.max_exposure;

  const float computed_exposure = lerp(exposure, target_exposure * powf(1.0f + brightness, 0.6f), lerp_factor);

  return max(min_exposure, min(max_exposure, computed_exposure));
}

void denoise_free(RaytraceInstance* instance) {
  if (!instance) {
    log_message("Raytrace Instance is NULL.");
    return;
  }

  if (!instance->denoise_setup) {
    log_message("OptiX Denoise Instance is NULL.");
    return;
  }

  OptixDenoiseInstance denoise_setup = *(OptixDenoiseInstance*) instance->denoise_setup;

  device_buffer_destroy(&denoise_setup.hdr_intensity);
  device_buffer_destroy(&denoise_setup.avg_color);
  device_buffer_destroy(&denoise_setup.denoiserState);
  device_buffer_destroy(&denoise_setup.denoiserScratch);
  device_buffer_destroy(&denoise_setup.output);

  OPTIX_CHECK(optixDenoiserDestroy(denoise_setup.denoiser));

  free(instance->denoise_setup);

  instance->denoise_setup = (void*) 0;
}
