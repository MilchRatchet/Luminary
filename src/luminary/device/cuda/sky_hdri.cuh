#ifndef CU_SKY_HDRI_H
#define CU_SKY_HDRI_H

#include "math.cuh"
#include "sky_utils.cuh"
#include "utils.cuh"

// This file contains the code for the precomputation of the sky and storing it in a HDRI like LUT.
// Note that since the LUT will contain cloud data, it will not parametrized like in Hillaire2020.
// The main goal is to eliminate the cost of the atmosphere if that is desired.
// Sampling Sun => (pos - hdri_pos) and then precompute the sun pos based on hdri values instead.

LUMINARY_FUNCTION float sky_hdri_warp_apply_median_of_means(float buckets[], const uint32_t num_buckets) {
  const uint32_t bucket_offset = (THREAD_ID_IN_BLOCK >> WARP_SIZE_LOG) * WARP_SIZE;

  // Sort
  for (uint32_t i = bucket_offset + 1; i < bucket_offset + num_buckets; i++) {
    const float x = buckets[i];
    uint32_t j    = i;
    while (j > bucket_offset && buckets[j - 1] > x) {
      buckets[j] = buckets[j - 1];
      j--;
    }

    buckets[j] = x;
  }

  // Gini - Median of Means

  float num   = 0.0f;
  float denom = 0.0f;

  for (uint32_t bucket_id = bucket_offset; bucket_id < bucket_offset + num_buckets; bucket_id++) {
    const float value = buckets[bucket_id];
    num += (bucket_id - bucket_offset) * value;
    denom += value;
  }

  num *= 2.0f;
  denom *= num_buckets;

  const float G = __saturatef((num / denom) - (num_buckets + 1.0f) / num_buckets);

  const uint32_t k = num_buckets >> 1;
  const uint32_t c = k - (1.0f - G) * k;

  float output = 0.0f;

  for (uint32_t bucket_id = bucket_offset + c; bucket_id < bucket_offset + num_buckets - c; bucket_id++) {
    output += buckets[bucket_id];
  }

  output /= num_buckets - 2 * c;

  return output;
}

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

  const uint32_t bucket_count = min(WARP_SIZE, args.sample_count);

  __syncwarp();

  __shared__ float shared_thread_values[THREADS_PER_BLOCK];

  shared_thread_values[THREAD_ID_IN_BLOCK] = (num_samples_thread > 0) ? color_thread.r / num_samples_thread : 0.0f;

  __syncwarp();

  float red_warp = 0.0f;
  if (THREAD_ID_IN_WARP == 0) {
    red_warp = sky_hdri_warp_apply_median_of_means(shared_thread_values, bucket_count);
  }

  __syncwarp();

  shared_thread_values[THREAD_ID_IN_BLOCK] = (num_samples_thread > 0) ? color_thread.g / num_samples_thread : 0.0f;

  __syncwarp();

  float green_warp = 0.0f;
  if (THREAD_ID_IN_WARP == 0) {
    green_warp = sky_hdri_warp_apply_median_of_means(shared_thread_values, bucket_count);
  }

  __syncwarp();

  shared_thread_values[THREAD_ID_IN_BLOCK] = (num_samples_thread > 0) ? color_thread.b / num_samples_thread : 0.0f;

  __syncwarp();

  if (THREAD_ID_IN_WARP == 0) {
    const float blue_warp = sky_hdri_warp_apply_median_of_means(shared_thread_values, bucket_count);

    const uint32_t index_color = x + y * args.ld_color;

    __stcs(args.dst_color + index_color, make_float4(red_warp, green_warp, blue_warp, 0.0f));
  }

  __syncwarp();

  shared_thread_values[THREAD_ID_IN_BLOCK] = (num_samples_thread > 0) ? alpha_thread / num_samples_thread : 0.0f;

  __syncwarp();

  if (THREAD_ID_IN_WARP == 0) {
    const float alpha_warp = sky_hdri_warp_apply_median_of_means(shared_thread_values, bucket_count);

    const uint32_t index_shadow = x + y * args.ld_shadow;

    __stcs(args.dst_shadow + index_shadow, alpha_warp);
  }
}

#endif /* CU_SKY_HDRI_H */
