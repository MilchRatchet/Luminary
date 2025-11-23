#ifndef CU_SKY_HDRI_H
#define CU_SKY_HDRI_H

#include "math.cuh"
#include "sky_utils.cuh"
#include "utils.cuh"

// This file contains the code for the precomputation of the sky and storing it in a HDRI like LUT.
// Note that since the LUT will contain cloud data, it will not parametrized like in Hillaire2020.
// The main goal is to eliminate the cost of the atmosphere if that is desired.
// Sampling Sun => (pos - hdri_pos) and then precompute the sun pos based on hdri values instead.

// TODO: Use MoN based firefly rejection

LUMINARY_KERNEL void sky_compute_hdri(const KernelArgsSkyComputeHDRI args) {
  const uint32_t pixel_id = WARP_ID;

  const uint32_t y = pixel_id / args.dim;
  const uint32_t x = pixel_id - y * args.dim;

  const float step_size = 1.0f / (args.dim - 1);

  RGBF color_thread           = splat_color(0.0f);
  float alpha_thread          = 0.0f;
  uint32_t num_samples_thread = 0;

  for (uint32_t sample_id = THREAD_ID_IN_WARP; sample_id < args.sample_count; sample_id += WARP_SIZE) {
    const PathID path_id = path_id_get(x, y, sample_id);

    const float2 jitter = random_2D(RANDOM_TARGET_CAMERA_JITTER, path_id);

    const float u = (((float) x) + jitter.x) * step_size;
    const float v = 1.0f - (((float) y) + jitter.y) * step_size;

    const float altitude = PI * v - 0.5f * PI;
    const float azimuth  = 2.0f * PI * u - PI;

    const vec3 ray = angles_to_direction(altitude, azimuth);

    RGBF sky_color            = splat_color(0.0f);
    RGBF transmittance        = splat_color(1.0f);
    float cloud_transmittance = 1.0f;
    vec3 sky_origin           = world_to_sky_transform(args.origin);

    if (device.cloud.active) {
      const float offset = clouds_render(sky_origin, ray, FLT_MAX, path_id, sky_color, transmittance, cloud_transmittance);

      sky_origin = add_vector(sky_origin, scale_vector(ray, offset));
    }

    const RGBF sky = sky_get_color(sky_origin, ray, FLT_MAX, false, device.sky.steps, path_id);

    sky_color = add_color(sky_color, mul_color(sky, transmittance));

    color_thread = add_color(color_thread, sky_color);
    alpha_thread += cloud_transmittance;

    num_samples_thread++;
  }

  const float red_warp            = warp_reduce_sum(color_thread.r);
  const float green_warp          = warp_reduce_sum(color_thread.g);
  const float blue_warp           = warp_reduce_sum(color_thread.b);
  const float alpha_warp          = warp_reduce_sum(alpha_thread);
  const uint32_t num_samples_warp = warp_reduce_sum(num_samples_thread);

  if (THREAD_ID_IN_WARP == 0) {
    const float normalization = 1.0f / num_samples_warp;

    const RGBF color  = scale_color(get_color(red_warp, green_warp, blue_warp), normalization);
    const float alpha = alpha_warp * normalization;

    const uint32_t index_color  = x + y * args.ld_color;
    const uint32_t index_shadow = x + y * args.ld_shadow;

    __stcs(args.dst_color + index_color, make_float4(color.r, color.g, color.b, 0.0f));
    __stcs(args.dst_shadow + index_shadow, alpha);
  }
}

#endif /* CU_SKY_HDRI_H */
