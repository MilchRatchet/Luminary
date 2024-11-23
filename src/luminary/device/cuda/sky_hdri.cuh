#ifndef CU_SKY_HDRI_H
#define CU_SKY_HDRI_H

#include "math.cuh"
#include "sky_utils.cuh"
#include "utils.cuh"

// This file contains the code for the precomputation of the sky and storing it in a HDRI like LUT.
// Note that since the LUT will contain cloud data, it will not parametrized like in Hillaire2020.
// The main goal is to eliminate the cost of the atmosphere if that is desired.
// Sampling Sun => (pos - hdri_pos) and then precompute the sun pos based on hdri values instead.

LUMINARY_KERNEL void sky_compute_hdri(float4* dst, float* dst_alpha, const uint32_t dim, const vec3 origin, const uint32_t sample_count) {
  unsigned int id = THREAD_ID;

  const int amount = dim * dim;

  const float step_size = 1.0f / (dim - 1);

  while (id < amount) {
    const int y = id / dim;
    const int x = id - y * dim;

    const ushort2 pixel_coords = make_ushort2(x, y);
    const uint32_t pixel       = x + y * dim;

    const float2 jitter = quasirandom_sequence_2D(QUASI_RANDOM_TARGET_CAMERA_JITTER, pixel_coords);

    const float u = (((float) x) + jitter.x) * step_size;
    const float v = 1.0f - (((float) y) + jitter.y) * step_size;

    const float altitude = PI * v - 0.5f * PI;
    const float azimuth  = 2.0f * PI * u - PI;

    const vec3 ray = angles_to_direction(altitude, azimuth);

    RGBF color                = get_color(0.0f, 0.0f, 0.0f);
    RGBF transmittance        = get_color(1.0f, 1.0f, 1.0f);
    float cloud_transmittance = 1.0f;
    vec3 sky_origin           = world_to_sky_transform(origin);

    if (device.cloud.active) {
      const float offset = clouds_render(sky_origin, ray, FLT_MAX, pixel_coords, color, transmittance, cloud_transmittance);

      sky_origin = add_vector(sky_origin, scale_vector(ray, offset));
    }

    const RGBF sky = sky_get_color(sky_origin, ray, FLT_MAX, false, device.sky.steps, pixel_coords);

    color = add_color(color, mul_color(sky, transmittance));

    RGBF result;
    float variance;
    float alpha;
    if (device.sample_id != 0.0f) {
      float4 data = __ldcs(dst + pixel);
      alpha       = __ldcs(dst_alpha + pixel);

      result   = get_color(data.x, data.y, data.z);
      variance = data.w;

      const float deviation = sqrtf(fmaxf(variance, eps));

      variance  = variance * (device.sample_id - 1.0f);
      RGBF diff = sub_color(color, result);
      diff      = mul_color(diff, diff);

      variance = variance + color_importance(diff);
      variance = variance * (1.0f / device.sample_id);

      // Same as in temporal accumulation
      // Here this trick has no real downside
      // Just got to make sure we don't do this in the case of 2 samples
      if (device.sample_id == 1 && sample_count != 2) {
        RGBF min = min_color(color, result);

        result = min;
        color  = min;
      }

      RGBF firefly_rejection = add_color(get_color(0.1f, 0.1f, 0.1f), add_color(result, get_color(deviation, deviation, deviation)));
      firefly_rejection      = max_color(get_color(0.0f, 0.0f, 0.0f), sub_color(color, firefly_rejection));

      result = scale_color(result, device.sample_id);
      alpha *= device.sample_id;

      color = sub_color(color, firefly_rejection);
    }
    else {
      result   = get_color(0.0f, 0.0f, 0.0f);
      variance = 1.0f;
      alpha    = 0.0f;
    }

    result = add_color(result, color);
    alpha += cloud_transmittance;

    result = scale_color(result, 1.0f / (device.sample_id + 1));
    alpha *= 1.0f / (device.sample_id + 1);

    __stcs(dst + pixel, make_float4(result.r, result.g, result.b, variance));
    __stcs(dst_alpha + pixel, alpha);

    id += blockDim.x * gridDim.x;
  }
}

#endif /* CU_SKY_HDRI_H */
