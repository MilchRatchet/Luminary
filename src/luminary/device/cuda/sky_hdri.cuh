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
// TODO: Use splatting

LUMINARY_KERNEL void sky_compute_hdri(const KernelArgsSkyComputeHDRI args) {
  uint32_t id = THREAD_ID;

  const uint32_t amount = args.dim * args.dim;

  const float step_size = 1.0f / (args.dim - 1);

  while (id < amount) {
    const int y = id / args.dim;
    const int x = id - y * args.dim;

    const PathID path_id = path_id_get(x, y, args.sample_id);

    const uint32_t index_color  = x + y * args.ld_color;
    const uint32_t index_shadow = x + y * args.ld_shadow;

    const float2 jitter = random_2D(RANDOM_TARGET_CAMERA_JITTER, path_id);

    const float u = (((float) x) + jitter.x) * step_size;
    const float v = 1.0f - (((float) y) + jitter.y) * step_size;

    const float altitude = PI * v - 0.5f * PI;
    const float azimuth  = 2.0f * PI * u - PI;

    const vec3 ray = angles_to_direction(altitude, azimuth);

    RGBF color                = get_color(0.0f, 0.0f, 0.0f);
    RGBF transmittance        = get_color(1.0f, 1.0f, 1.0f);
    float cloud_transmittance = 1.0f;
    vec3 sky_origin           = world_to_sky_transform(args.origin);

    if (device.cloud.active) {
      const float offset = clouds_render(sky_origin, ray, FLT_MAX, path_id, color, transmittance, cloud_transmittance);

      sky_origin = add_vector(sky_origin, scale_vector(ray, offset));
    }

    const RGBF sky = sky_get_color(sky_origin, ray, FLT_MAX, false, device.sky.steps, path_id);

    color = add_color(color, mul_color(sky, transmittance));

    RGBF result;
    float variance;
    float alpha;
    if (args.sample_id != 0) {
      float4 data = __ldcs(args.dst_color + index_color);
      alpha       = __ldcs(args.dst_shadow + index_shadow);

      result   = get_color(data.x, data.y, data.z);
      variance = data.w;

      const float deviation = sqrtf(fmaxf(variance, eps));

      variance  = variance * (args.sample_id - 1.0f);
      RGBF diff = sub_color(color, result);
      diff      = mul_color(diff, diff);

      variance = variance + color_importance(diff);
      variance = variance * (1.0f / args.sample_id);

      // Same as in temporal accumulation
      // Here this trick has no real downside
      // Just got to make sure we don't do this in the case of 2 samples
      if (args.sample_id == 1 && args.sample_count != 2) {
        RGBF min = min_color(color, result);

        result = min;
        color  = min;
      }

      RGBF firefly_rejection = add_color(get_color(0.1f, 0.1f, 0.1f), add_color(result, get_color(deviation, deviation, deviation)));
      firefly_rejection      = max_color(get_color(0.0f, 0.0f, 0.0f), sub_color(color, firefly_rejection));

      result = scale_color(result, args.sample_id);
      alpha *= args.sample_id;

      color = sub_color(color, firefly_rejection);
    }
    else {
      result   = get_color(0.0f, 0.0f, 0.0f);
      variance = 1.0f;
      alpha    = 0.0f;
    }

    result = add_color(result, color);
    alpha += cloud_transmittance;

    result = scale_color(result, 1.0f / (args.sample_id + 1));
    alpha *= 1.0f / (args.sample_id + 1);

    __stcs(args.dst_color + index_color, make_float4(result.r, result.g, result.b, variance));
    __stcs(args.dst_shadow + index_shadow, alpha);

    id += blockDim.x * gridDim.x;
  }
}

#endif /* CU_SKY_HDRI_H */
