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
      output   = buffer;
      variance = get_color(1.0f, 1.0f, 1.0f);

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
