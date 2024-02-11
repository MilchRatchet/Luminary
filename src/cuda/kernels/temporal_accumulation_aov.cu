#include "math.cuh"
#include "memory.cuh"
#include "utils.cuh"

__global__ void temporal_accumulation_aov(const RGBF* buffer, RGBF* accumulate) {
  const int amount = device.width * device.height;

  for (int offset = THREAD_ID; offset < amount; offset += blockDim.x * gridDim.x) {
    RGBF input  = load_RGBF(buffer + offset);
    RGBF output = (device.temporal_frames == 0) ? input : load_RGBF(accumulate + offset);

    output = scale_color(output, device.temporal_frames);
    output = add_color(input, output);
    output = scale_color(output, 1.0f / (device.temporal_frames + 1));

    store_RGBF(accumulate + offset, output);
  }
}
