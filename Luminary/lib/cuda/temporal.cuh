#include "image.h"
#include "utils.cuh"

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
    RGBF buffer = device_frame_buffer[offset];
    vec3 hit    = device_world_space_hit[offset];

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

    prev_pixel.x -= device_jitter.x;
    prev_pixel.y -= device_jitter.y;

    const int prev_x = prev_pixel.x;
    const int prev_y = prev_pixel.y;

    float2 w = make_float2(prev_pixel.x - floorf(prev_pixel.x), prev_pixel.y - floorf(prev_pixel.y));

    RGBF temporal     = get_color(0.0f, 0.0f, 0.0f);
    float sum_weights = 0.0f;

    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 2; j++) {
        const int x = prev_x + i;
        const int y = prev_y + j;

        if (x < 0 || x >= device_width || y < 0 || y >= device_height)
          continue;

        const float weight = ((i == 0) ? (1.0f - w.x) : w.x) * ((j == 0) ? (1.0f - w.y) : w.y);

        temporal = add_color(temporal, scale_color(device_frame_temporal[y * device_width + x], weight));
        sum_weights += weight;
      }
    }

    if (sum_weights > 0.01f) {
      temporal    = scale_color(temporal, 1.0f / sum_weights);
      float alpha = 0.01f;
      buffer      = add_color(scale_color(buffer, alpha), scale_color(temporal, 1.0f - alpha));
    }

    device_frame_output[offset] = buffer;
  }
}
