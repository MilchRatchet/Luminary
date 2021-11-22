#include "image.h"
#include "utils.cuh"

/*
 * TAA implementation based on blog by Alex Tardiff (http://alextardif.com/TAA.html)
 * CatmullRom filter implementation based on https://gist.github.com/TheRealMJP/c83b8c0f46b63f3a88a5986f4fa982b1
 * Mitchell-Netravali filter based on https://github.com/TheRealMJP/MSAAFilter
 */

__device__ float temporal_cubic_filter(float x, float a, float b) {
  float y  = 0.0f;
  float x2 = x * x;
  float x3 = x2 * x;
  if (x < 1.0f) {
    y = (12.0f - 9.0f * a - 6.0f * b) * x3 + (-18.0f + 12.0f * a + 6.0f * b) * x2 + (6.0f - 2.0f * a);
  }
  else if (x <= 2.0f) {
    y = (-a - 6.0f * b) * x3 + (6.0f * a + 30.0f * b) * x2 + (-12.0f * a - 48.0f * b) * x + (8.0f * a + 24.0f * b);
  }

  return y / 6.0f;
}

__device__ float mitchell_netravali(int x, int y) {
  float fx = (float) x;
  float fy = (float) y;

  float len = sqrtf(fx * fx + fy * fy);

  return temporal_cubic_filter(len, 1.0f / 3.0f, 1.0f / 3.0f);
}

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

  result = add_color(result, scale_color(sample_pixel(image, x0, y0, width, height), wx0 * wy0));
  result = add_color(result, scale_color(sample_pixel(image, x12, y0, width, height), wx12 * wy0));
  result = add_color(result, scale_color(sample_pixel(image, x3, y0, width, height), wx3 * wy0));

  result = add_color(result, scale_color(sample_pixel(image, x0, y12, width, height), wx0 * wy12));
  result = add_color(result, scale_color(sample_pixel(image, x12, y12, width, height), wx12 * wy12));
  result = add_color(result, scale_color(sample_pixel(image, x3, y12, width, height), wx3 * wy12));

  result = add_color(result, scale_color(sample_pixel(image, x0, y3, width, height), wx0 * wy3));
  result = add_color(result, scale_color(sample_pixel(image, x12, y3, width, height), wx12 * wy3));
  result = add_color(result, scale_color(sample_pixel(image, x3, y3, width, height), wx3 * wy3));

  return result;
}

__global__ void temporal_accumulation() {
  int offset = threadIdx.x + blockIdx.x * blockDim.x;

  for (; offset < device_amount; offset += blockDim.x * gridDim.x) {
    RGBF buffer = device_frame_buffer[offset];
    RGBF output;
    RGBF variance;
    RGBF bias_cache;

    if (device_temporal_frames == 0) {
      output     = buffer;
      variance   = get_color(1.0f, 1.0f, 1.0f);
      bias_cache = get_color(0.0f, 0.0f, 0.0f);
    }
    else {
      output     = device_frame_output[offset];
      variance   = device_frame_variance[offset];
      bias_cache = device_frame_bias_cache[offset];
    }

    RGBF deviation;
    deviation.r = sqrtf(fmaxf(eps, variance.r));
    deviation.g = sqrtf(fmaxf(eps, variance.g));
    deviation.b = sqrtf(fmaxf(eps, variance.b));

    if (device_temporal_frames) {
      variance  = scale_color(variance, device_temporal_frames - 1.0f);
      RGBF diff = sub_color(buffer, output);
      diff      = mul_color(diff, diff);
      variance  = add_color(variance, diff);
      variance  = scale_color(variance, 1.0f / device_temporal_frames);
    }

    device_frame_variance[offset] = variance;

    RGBF firefly_rejection;
    firefly_rejection.r = 0.1f + output.r + deviation.r * 4.0f;
    firefly_rejection.g = 0.1f + output.g + deviation.g * 4.0f;
    firefly_rejection.b = 0.1f + output.b + deviation.b * 4.0f;

    firefly_rejection.r = fmaxf(0.0f, buffer.r - firefly_rejection.r);
    firefly_rejection.g = fmaxf(0.0f, buffer.g - firefly_rejection.g);
    firefly_rejection.b = fmaxf(0.0f, buffer.b - firefly_rejection.b);

    bias_cache = add_color(bias_cache, firefly_rejection);
    buffer     = sub_color(buffer, firefly_rejection);

    RGBF debias;
    debias.r = fmaxf(0.0f, fminf(bias_cache.r, output.r - deviation.r * 2.0f - buffer.r));
    debias.g = fmaxf(0.0f, fminf(bias_cache.g, output.g - deviation.g * 2.0f - buffer.g));
    debias.b = fmaxf(0.0f, fminf(bias_cache.b, output.b - deviation.b * 2.0f - buffer.b));

    buffer     = add_color(buffer, debias);
    bias_cache = sub_color(bias_cache, debias);

    device_frame_bias_cache[offset] = bias_cache;

    output = scale_color(output, device_temporal_frames);
    output = add_color(buffer, output);
    output = scale_color(output, 1.0f / (device_temporal_frames + 1));

    device_frame_output[offset] = output;
  }
}

__global__ void temporal_reprojection() {
  for (int offset = threadIdx.x + blockIdx.x * blockDim.x; offset < device_amount; offset += blockDim.x * gridDim.x) {
    const int curr_x = offset % device_width;
    const int curr_y = offset / device_width;

    RGBF sum_color      = get_color(0.0f, 0.0f, 0.0f);
    float sum_weights   = 0.0f;
    float closest_depth = FLT_MAX;

    for (int i = -1; i <= -1; i++) {
      for (int j = -1; j <= -1; j++) {
        const int x = max(0, min(device_width, curr_x + j));
        const int y = max(0, min(device_height, curr_y + i));

        RGBF color = device_frame_buffer[x + y * device_width];

        float weight = mitchell_netravali(i, j);

        sum_color = add_color(sum_color, scale_color(color, weight));
        sum_weights += weight;

        TraceResult trace = device_trace_result_buffer[x + y * device_width];

        if (trace.depth < closest_depth) {
          closest_depth = trace.depth;
        }
      }
    }

    RGBF output = scale_color(sum_color, 1.0f / sum_weights);

    vec3 hit = add_vector(device_scene.camera.pos, scale_vector(device_raydir_buffer[offset], closest_depth));

    vec4 pos;
    pos.x = hit.x;
    pos.y = hit.y;
    pos.z = hit.z;
    pos.w = 1.0f;

    vec4 prev_pixel = transform_vec4(device_projection, transform_vec4(device_view_space, pos));

    prev_pixel.x /= -prev_pixel.w;
    prev_pixel.y /= -prev_pixel.w;

    prev_pixel.x = device_width * (1.0f - prev_pixel.x) * 0.5f;
    prev_pixel.y = device_height * (prev_pixel.y + 1.0f) * 0.5f;

    prev_pixel.x -= device_jitter.x - 0.5f;
    prev_pixel.y -= device_jitter.y - 0.5f;

    const int prev_x = prev_pixel.x;
    const int prev_y = prev_pixel.y;

    if (prev_x >= 0 && prev_x < device_width && prev_y >= 0 && prev_y < device_height) {
      RGBF temporal = sample_pixel_catmull_rom(device_frame_temporal, prev_pixel.x, prev_pixel.y, device_width, device_height);

      float alpha = device_scene.camera.temporal_blend_factor;
      output      = add_color(scale_color(output, alpha), scale_color(temporal, 1.0f - alpha));
    }

    device_frame_output[offset] = output;

    // Interesting motion vector visualization
    // device_frame_output[offset] = get_color(fabsf(curr_x - prev_pixel.x), 0.0f, fabsf(curr_y - prev_pixel.y));
  }
}
