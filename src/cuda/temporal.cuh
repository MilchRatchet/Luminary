#ifndef CU_TEMPORAL_H
#define CU_TEMPORAL_H

#include "structs.h"
#include "utils.cuh"

/*
 * TAA implementation based on blog by Alex Tardiff (http://alextardif.com/TAA.html)
 * CatmullRom filter implementation based on https://gist.github.com/TheRealMJP/c83b8c0f46b63f3a88a5986f4fa982b1
 */

__device__ RGBF sample_pixel_catmull_rom(const RGBF* image, float x, float y, const int width, const int height) {
  float px = floorf(x - 0.5f) + 0.5f;
  float py = floorf(y - 0.5f) + 0.5f;

  float fx = x - px;
  float fy = y - py;

  float wx0 = fx * (-0.5f + fx * (1.0f - 0.5f * fx));
  float wx1 = 1.0f + fx * fx * (-2.5f + 1.5f * fx);
  float wx2 = fx * (0.5f + fx * (2.0f - 1.5f * fx));
  float wx3 = fx * fx * (-0.5f + 0.5f * fx);
  float wy0 = fy * (-0.5f + fy * (1.0f - 0.5f * fy));
  float wy1 = 1.0f + fy * fy * (-2.5f + 1.5f * fy);
  float wy2 = fy * (0.5f + fy * (2.0f - 1.5f * fy));
  float wy3 = fy * fy * (-0.5f + 0.5f * fy);

  float wx12 = wx1 + wx2;
  float wy12 = wy1 + wy2;

  float ox12 = wx2 / wx12;
  float oy12 = wy2 / wy12;

  float x0  = px - 1.0f;
  float y0  = py - 1.0f;
  float x3  = px + 2.0f;
  float y3  = py + 2.0f;
  float x12 = px + ox12;
  float y12 = py + oy12;

  x0 /= (width - 1);
  y0 /= (height - 1);
  x3 /= (width - 1);
  y3 /= (height - 1);
  x12 /= (width - 1);
  y12 /= (height - 1);

  RGBF result = get_color(0.0f, 0.0f, 0.0f);

  result = add_color(result, scale_color(sample_pixel_clamp(image, x0, y0, width, height), wx0 * wy0));
  result = add_color(result, scale_color(sample_pixel_clamp(image, x12, y0, width, height), wx12 * wy0));
  result = add_color(result, scale_color(sample_pixel_clamp(image, x3, y0, width, height), wx3 * wy0));

  result = add_color(result, scale_color(sample_pixel_clamp(image, x0, y12, width, height), wx0 * wy12));
  result = add_color(result, scale_color(sample_pixel_clamp(image, x12, y12, width, height), wx12 * wy12));
  result = add_color(result, scale_color(sample_pixel_clamp(image, x3, y12, width, height), wx3 * wy12));

  result = add_color(result, scale_color(sample_pixel_clamp(image, x0, y3, width, height), wx0 * wy3));
  result = add_color(result, scale_color(sample_pixel_clamp(image, x12, y3, width, height), wx12 * wy3));
  result = add_color(result, scale_color(sample_pixel_clamp(image, x3, y3, width, height), wx3 * wy3));

  return result;
}

__global__ void temporal_accumulation() {
  const int amount = device.width * device.height;

  for (int offset = threadIdx.x + blockIdx.x * blockDim.x; offset < amount; offset += blockDim.x * gridDim.x) {
    RGBF buffer = load_RGBF(device.ptrs.frame_buffer + offset);
    RGBF output;
    RGBF variance;

    if (device.temporal_frames == 0) {
      output   = buffer;
      variance = get_color(1.0f, 1.0f, 1.0f);
    }
    else {
      output   = load_RGBF(device.ptrs.frame_accumulate + offset);
      variance = load_RGBF(device.ptrs.frame_variance + offset);
    }

    RGBF deviation = max_color(variance, get_color(eps, eps, eps));

    deviation.r = sqrtf(deviation.r);
    deviation.g = sqrtf(deviation.g);
    deviation.b = sqrtf(deviation.b);

    if (device.temporal_frames) {
      variance  = scale_color(variance, device.temporal_frames - 1.0f);
      RGBF diff = sub_color(buffer, output);

      // Hard firefly rejection.
      // Fireflies that appear during the first frame are accepted by our method since there is
      // no reference yet to reject them from. This trick here hard rejects them by taking the
      // dimmer of the two frames. This is not unbiased but since it only happens exactly once
      // the bias will decrease with the number of frames.
      // Taking neighbouring pixels as reference is not the target since I want to consider each
      // pixel as its own independent entity to preserve fine details.
      // TODO: Improve this method to remove the visible dimming during the second frame.
      if (device.temporal_frames == 1) {
        RGBF min = min_color(buffer, output);

        output = min;
        buffer = min;
      }

      diff = mul_color(diff, diff);

      variance = add_color(variance, diff);
      variance = scale_color(variance, 1.0f / device.temporal_frames);
    }

    store_RGBF(device.ptrs.frame_variance + offset, variance);

    RGBF firefly_rejection = add_color(get_color(0.1f, 0.1f, 0.1f), add_color(output, scale_color(deviation, 4.0f)));
    firefly_rejection      = max_color(get_color(0.0f, 0.0f, 0.0f), sub_color(buffer, firefly_rejection));

    buffer = sub_color(buffer, firefly_rejection);

    output = scale_color(output, device.temporal_frames);
    output = add_color(buffer, output);
    output = scale_color(output, 1.0f / (device.temporal_frames + 1));

    store_RGBF(device.ptrs.frame_accumulate + offset, output);
  }
}

__global__ void temporal_reprojection() {
  const int amount = device.width * device.height;

  for (int offset = threadIdx.x + blockIdx.x * blockDim.x; offset < amount; offset += blockDim.x * gridDim.x) {
    RGBF output               = load_RGBF(device.ptrs.frame_buffer + offset);
    const float closest_depth = device.ptrs.trace_result_buffer[offset].depth;

    vec3 hit = add_vector(device.scene.camera.pos, scale_vector(device.ptrs.raydir_buffer[offset], closest_depth));

    vec4 pos;
    pos.x = hit.x;
    pos.y = hit.y;
    pos.z = hit.z;
    pos.w = 1.0f;

    vec4 prev_pixel = transform_vec4(device.emitter.projection, transform_vec4(device.emitter.view_space, pos));

    prev_pixel.x /= -prev_pixel.w;
    prev_pixel.y /= -prev_pixel.w;

    prev_pixel.x = device.width * (1.0f - prev_pixel.x) * 0.5f;
    prev_pixel.y = device.height * (prev_pixel.y + 1.0f) * 0.5f;

    prev_pixel.x -= device.emitter.jitter.x - 0.5f;
    prev_pixel.y -= device.emitter.jitter.y - 0.5f;

    const int prev_x = prev_pixel.x;
    const int prev_y = prev_pixel.y;

    if (prev_x >= 0 && prev_x < device.width && prev_y >= 0 && prev_y < device.height) {
      RGBF temporal = sample_pixel_catmull_rom(device.ptrs.frame_temporal, prev_pixel.x, prev_pixel.y, device.width, device.height);

      float alpha = device.scene.camera.temporal_blend_factor;
      output      = add_color(scale_color(output, alpha), scale_color(temporal, 1.0f - alpha));
    }

    if (isinf(output.r) || isnan(output.r) || isinf(output.g) || isnan(output.g) || isinf(output.b) || isnan(output.b)) {
      output = get_color(0.0f, 0.0f, 0.0f);
    }

    device.ptrs.frame_accumulate[offset] = output;

    // Interesting motion vector visualization
    // device.frame_output[offset] = get_color(fabsf(curr_x - prev_pixel.x), 0.0f, fabsf(curr_y - prev_pixel.y));
  }
}

#endif /* CU_TEMPORAL_H */
