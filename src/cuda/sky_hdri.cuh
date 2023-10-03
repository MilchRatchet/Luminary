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

__global__ __launch_bounds__(THREADS_PER_BLOCK, 5) void sky_hdri_compute_hdri_lut(float4* dst) {
  unsigned int id = THREAD_ID;

  uint32_t seed = device.ptrs.randoms[THREAD_ID];

  const int amount = device.scene.sky.hdri_dim * device.scene.sky.hdri_dim;

  const float step_size = 1.0f / (device.scene.sky.hdri_dim - 1);

  while (id < amount) {
    const int x = id % device.scene.sky.hdri_dim;
    const int y = id / device.scene.sky.hdri_dim;

    RGBF result   = get_color(0.0f, 0.0f, 0.0f);
    RGBF variance = get_color(1.0f, 1.0f, 1.0f);

    const vec3 sky_origin = world_to_sky_transform(device.scene.sky.hdri_origin);

    for (int i = 0; i < device.scene.sky.hdri_samples; i++) {
      const float jitter_u = 0.5f + sky_hdri_tent_filter_importance_sample(white_noise());
      const float jitter_v = 0.5f + sky_hdri_tent_filter_importance_sample(white_noise());

      const float u = (((float) x) + jitter_u) * step_size;
      const float v = 1.0f - (((float) y) + jitter_v) * step_size;

      const float altitude = PI * v - 0.5f * PI;
      const float azimuth  = 2.0f * PI * u - PI;

      const vec3 ray = angles_to_direction(altitude, azimuth);

      vec3 iter_origin = sky_origin;

      RGBF color               = get_color(0.0f, 0.0f, 0.0f);
      RGBF cloud_transmittance = get_color(1.0f, 1.0f, 1.0f);

      if (device.scene.sky.cloud.active) {
        const float offset = clouds_render(sky_origin, ray, FLT_MAX, color, cloud_transmittance, seed);

        iter_origin = add_vector(iter_origin, scale_vector(ray, offset));
      }

      const RGBF sky = sky_get_color(iter_origin, ray, FLT_MAX, true, device.scene.sky.steps, seed);

      color = add_color(color, mul_color(sky, cloud_transmittance));

      if (i) {
        RGBF deviation = max_color(variance, get_color(eps, eps, eps));

        deviation.r = sqrtf(deviation.r);
        deviation.g = sqrtf(deviation.g);
        deviation.b = sqrtf(deviation.b);

        result = scale_color(result, 1.0f / i);

        variance  = scale_color(variance, i - 1.0f);
        RGBF diff = sub_color(color, result);
        diff      = mul_color(diff, diff);

        variance = add_color(variance, diff);
        variance = scale_color(variance, 1.0f / i);

        // Same as in temporal accumulation
        // Here this trick has no real downside
        // Just got to make sure we don't do this in the case of 2 samples
        if (i == 1 && device.scene.sky.hdri_samples != 2) {
          RGBF min = min_color(color, result);

          result = min;
          color  = min;
        }

        RGBF firefly_rejection = add_color(get_color(0.1f, 0.1f, 0.1f), add_color(result, scale_color(deviation, 4.0f)));
        firefly_rejection      = max_color(get_color(0.0f, 0.0f, 0.0f), sub_color(color, firefly_rejection));

        result = scale_color(result, i);

        color = sub_color(color, firefly_rejection);
      }

      result = add_color(result, color);
    }

    result = scale_color(result, 1.0f / device.scene.sky.hdri_samples);

    dst[x + y * device.scene.sky.hdri_dim] = make_float4(result.r, result.g, result.b, 0.0f);

    id += blockDim.x * gridDim.x;
  }

  device.ptrs.randoms[THREAD_ID] = seed;
}

extern "C" void sky_hdri_generate_LUT(RaytraceInstance* instance) {
  bench_tic();

  if (instance->scene.sky.hdri_initialized) {
    texture_free_atlas(instance->sky_hdri_luts, 1);
  }

  instance->scene.sky.hdri_dim = instance->scene.sky.settings_hdri_dim;

  const int dim = instance->scene.sky.hdri_dim;

  instance->scene.sky.hdri_initialized = 1;

  raytrace_update_device_scene(instance);

  TextureRGBA luts_hdri_tex[1];
  texture_create(luts_hdri_tex + 0, dim, dim, 1, dim, (void*) 0, TexDataFP32, TexStorageGPU);
  luts_hdri_tex[0].wrap_mode_S = TexModeWrap;
  luts_hdri_tex[0].wrap_mode_T = TexModeClamp;
  luts_hdri_tex[0].mipmap      = TexMipmapGenerate;

  device_malloc((void**) &luts_hdri_tex[0].data, luts_hdri_tex[0].height * luts_hdri_tex[0].pitch * 4 * sizeof(float));

  RayIterationType iter_type = TYPE_CAMERA;
  device_update_symbol(iteration_type, iter_type);

  sky_hdri_compute_hdri_lut<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>((float4*) luts_hdri_tex[0].data);

  gpuErrchk(cudaDeviceSynchronize());

  texture_create_atlas(&instance->sky_hdri_luts, luts_hdri_tex, 1);

  device_free(luts_hdri_tex[0].data, luts_hdri_tex[0].height * luts_hdri_tex[0].pitch * 4 * sizeof(float));

  raytrace_update_device_pointers(instance);

  bench_toc((char*) "Sky HDRI Computation");
}

extern "C" void sky_hdri_set_pos_to_cam(RaytraceInstance* instance) {
  instance->scene.sky.hdri_origin = instance->scene.camera.pos;
}

#endif /* CU_SKY_HDRI_H */
