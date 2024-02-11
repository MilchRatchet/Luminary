#include "math.cuh"
#include "memory.cuh"
#include "utils.cuh"

__global__ void temporal_accumulation() {
  const int amount = device.width * device.height;

  for (int offset = THREAD_ID; offset < amount; offset += blockDim.x * gridDim.x) {
    RGBF buffer = load_RGBF(device.ptrs.frame_buffer + offset);
    RGBF output;
    float variance;

    if (device.temporal_frames == 0) {
      output   = buffer;
      variance = 1.0f;
    }
    else {
      output   = load_RGBF(device.ptrs.frame_accumulate + offset);
      variance = __ldcs(device.ptrs.frame_variance + offset);
    }

    float luminance_buffer = luminance(buffer);
    float luminance_output = luminance(output);

    const float deviation = fminf(0.1f, sqrtf(fmaxf(variance, eps)));

    if (device.temporal_frames) {
      variance *= device.temporal_frames - 1.0f;

      float diff = luminance_buffer - luminance_output;
      diff       = diff * diff;

      // Hard firefly rejection.
      // Fireflies that appear during the first frame are accepted by our method since there is
      // no reference yet to reject them from. This trick here hard rejects them by taking the
      // dimmer of the two frames. This is not unbiased but since it only happens exactly once
      // the bias will decrease with the number of frames.
      // Taking neighbouring pixels as reference is not the target since I want to consider each
      // pixel as its own independent entity to preserve fine details.
      // TODO: Improve this method to remove the visible dimming during the second frame.
      if (device.temporal_frames == 1) {
        float min_luminance = fminf(luminance_buffer, luminance_output);

        output = (luminance_output > eps) ? scale_color(output, min_luminance / luminance_output) : get_color(0.0f, 0.0f, 0.0f);
        buffer = (luminance_buffer > eps) ? scale_color(buffer, min_luminance / luminance_buffer) : get_color(0.0f, 0.0f, 0.0f);
      }

      variance += diff;
      variance *= 1.0f / device.temporal_frames;
    }

    __stcs(device.ptrs.frame_variance + offset, variance);

    const float firefly_rejection    = 0.1f + luminance_output + deviation * 4.0f;
    const float new_luminance_buffer = fminf(luminance_buffer, firefly_rejection);

    buffer = (luminance_buffer > eps) ? scale_color(buffer, new_luminance_buffer / luminance_buffer) : get_color(0.0f, 0.0f, 0.0f);

    output = scale_color(output, device.temporal_frames);
    output = add_color(buffer, output);
    output = scale_color(output, 1.0f / (device.temporal_frames + 1));

    store_RGBF(device.ptrs.frame_accumulate + offset, output);
  }
}
