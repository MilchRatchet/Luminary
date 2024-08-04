#ifndef CU_SKY_HDRI_H
#define CU_SKY_HDRI_H

#include "bench.h"
#include "raytrace.h"
#include "structs.h"
#include "texture.h"
#include "utils.cuh"
#include "utils.h"

/*
 * For documentation, see the host version in raytrace.c which is used for the camera rays.
 */
__device__ float sky_hdri_tent_filter_importance_sample(const float x) {
  if (x > 0.5f) {
    return 1.0f - sqrtf(2.0f) * sqrtf(1.0f - x);
  }
  else {
    return -1.0f + sqrtf(2.0f) * sqrtf(x);
  }
}

// This file contains the code for the precomputation of the sky and storing it in a HDRI like LUT.
// Note that since the LUT will contain cloud data, it will not parametrized like in Hillaire2020.
// The main goal is to eliminate the cost of the atmosphere if that is desired.
// Sampling Sun => (pos - hdri_pos) and then precompute the sun pos based on hdri values instead.

LUMINARY_KERNEL void sky_hdri_compute_hdri_lut(float4* dst, float* dst_alpha) {
  unsigned int id = THREAD_ID;

  const int amount = device.scene.sky.hdri_dim * device.scene.sky.hdri_dim;

  const float step_size = 1.0f / (device.scene.sky.hdri_dim - 1);

  while (id < amount) {
    const int y = id / device.scene.sky.hdri_dim;
    const int x = id - y * device.scene.sky.hdri_dim;

    const ushort2 pixel_coords = make_ushort2(x, y);
    const uint32_t pixel       = x + y * device.scene.sky.hdri_dim;

    float2 random_jitter = quasirandom_sequence_2D(QUASI_RANDOM_TARGET_CAMERA_JITTER, pixel_coords);

    const float jitter_u = 0.5f + sky_hdri_tent_filter_importance_sample(random_jitter.x);
    const float jitter_v = 0.5f + sky_hdri_tent_filter_importance_sample(random_jitter.y);

    const float u = (((float) x) + jitter_u) * step_size;
    const float v = 1.0f - (((float) y) + jitter_v) * step_size;

    const float altitude = PI * v - 0.5f * PI;
    const float azimuth  = 2.0f * PI * u - PI;

    const vec3 ray = angles_to_direction(altitude, azimuth);

    RGBF color                = get_color(0.0f, 0.0f, 0.0f);
    RGBF transmittance        = get_color(1.0f, 1.0f, 1.0f);
    float cloud_transmittance = 1.0f;
    vec3 sky_origin           = world_to_sky_transform(device.scene.sky.hdri_origin);

    if (device.scene.sky.cloud.active) {
      const float offset = clouds_render(sky_origin, ray, FLT_MAX, pixel_coords, color, transmittance, cloud_transmittance);

      sky_origin = add_vector(sky_origin, scale_vector(ray, offset));
    }

    const RGBF sky = sky_get_color(sky_origin, ray, FLT_MAX, false, device.scene.sky.steps, pixel_coords);

    color = add_color(color, mul_color(sky, transmittance));

    RGBF result;
    float variance;
    float alpha;
    if (device.temporal_frames) {
      float4 data = __ldcs(dst + pixel);
      alpha       = __ldcs(dst_alpha + pixel);

      result   = get_color(data.x, data.y, data.z);
      variance = data.w;

      const float deviation = sqrtf(fmaxf(variance, eps));

      variance  = variance * (device.temporal_frames - 1.0f);
      RGBF diff = sub_color(color, result);
      diff      = mul_color(diff, diff);

      variance = variance + color_importance(diff);
      variance = variance * (1.0f / device.temporal_frames);

      // Same as in temporal accumulation
      // Here this trick has no real downside
      // Just got to make sure we don't do this in the case of 2 samples
      if (device.temporal_frames == 1 && device.scene.sky.hdri_samples != 2) {
        RGBF min = min_color(color, result);

        result = min;
        color  = min;
      }

      RGBF firefly_rejection = add_color(get_color(0.1f, 0.1f, 0.1f), add_color(result, get_color(deviation, deviation, deviation)));
      firefly_rejection      = max_color(get_color(0.0f, 0.0f, 0.0f), sub_color(color, firefly_rejection));

      result = scale_color(result, device.temporal_frames);
      alpha *= device.temporal_frames;

      color = sub_color(color, firefly_rejection);
    }
    else {
      result   = get_color(0.0f, 0.0f, 0.0f);
      variance = 1.0f;
      alpha    = 0.0f;
    }

    result = add_color(result, color);
    alpha += cloud_transmittance;

    result = scale_color(result, 1.0f / (device.temporal_frames + 1));
    alpha *= 1.0f / (device.temporal_frames + 1);

    __stcs(dst + pixel, make_float4(result.r, result.g, result.b, variance));
    __stcs(dst_alpha + pixel, alpha);

    id += blockDim.x * gridDim.x;
  }
}

extern "C" void sky_hdri_generate_LUT(RaytraceInstance* instance) {
  bench_tic((const char*) "Sky HDRI Computation");

  if (instance->scene.sky.hdri_initialized) {
    texture_free_atlas(instance->sky_hdri_luts, 2);
  }

  instance->scene.sky.hdri_dim = instance->scene.sky.settings_hdri_dim;

  const int dim = instance->scene.sky.hdri_dim;

  if (dim == 0) {
    error_message("Failed to allocated HDRI because resolution was 0. Turned off HDRI.");
    instance->scene.sky.hdri_active = 0;

    // Update GPU constants again because we may have already pushed hdri_active.
    raytrace_update_device_scene(instance);
    return;
  }

  instance->scene.sky.hdri_initialized = 1;

  raytrace_update_device_scene(instance);

  TextureRGBA luts_hdri_tex[2];
  texture_create(luts_hdri_tex + 0, dim, dim, 1, dim, (void*) 0, TexDataFP32, 4, TexStorageGPU);
  luts_hdri_tex[0].wrap_mode_S = TexModeWrap;
  luts_hdri_tex[0].wrap_mode_T = TexModeClamp;
  luts_hdri_tex[0].mipmap      = TexMipmapGenerate;

  texture_create(luts_hdri_tex + 1, dim, dim, 1, dim, (void*) 0, TexDataFP32, 1, TexStorageGPU);
  luts_hdri_tex[1].wrap_mode_S = TexModeWrap;
  luts_hdri_tex[1].wrap_mode_T = TexModeClamp;
  luts_hdri_tex[1].mipmap      = TexMipmapNone;

  device_malloc((void**) &luts_hdri_tex[0].data, luts_hdri_tex[0].height * luts_hdri_tex[0].pitch * 4 * sizeof(float));
  device_malloc((void**) &luts_hdri_tex[1].data, luts_hdri_tex[1].height * luts_hdri_tex[1].pitch * 1 * sizeof(float));

  int depth = 0;
  device_update_symbol(depth, depth);

  for (int i = 0; i < instance->scene.sky.hdri_samples; i++) {
    device_update_symbol(temporal_frames, i);
    print_info_inline("HDRI Progress: %i/%i", i, instance->scene.sky.hdri_samples);
    sky_hdri_compute_hdri_lut<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>((float4*) luts_hdri_tex[0].data, (float*) luts_hdri_tex[1].data);
    gpuErrchk(cudaDeviceSynchronize());
  }

  texture_create_atlas(&instance->sky_hdri_luts, luts_hdri_tex, 2);

  device_free(luts_hdri_tex[0].data, luts_hdri_tex[0].height * luts_hdri_tex[0].pitch * 4 * sizeof(float));
  device_free(luts_hdri_tex[1].data, luts_hdri_tex[1].height * luts_hdri_tex[1].pitch * 1 * sizeof(float));

  raytrace_update_device_pointers(instance);

  bench_toc();
}

extern "C" void sky_hdri_set_pos_to_cam(RaytraceInstance* instance) {
  instance->scene.sky.hdri_origin = instance->scene.camera.pos;
}

#endif /* CU_SKY_HDRI_H */
