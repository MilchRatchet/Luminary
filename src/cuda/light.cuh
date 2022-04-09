#ifndef CU_LIGHT_H
#define CU_LIGHT_H

#include "brdf.cuh"
#include "utils.cuh"

__global__ void generate_light_samples() {
  const int task_count = device.task_counts[(threadIdx.x + blockIdx.x * blockDim.x) * 6 + 5];

  for (int i = 0; i < task_count; i++) {
    const int offset     = get_task_address(i);
    const ushort2 index  = __ldg((ushort2*) (device_trace_tasks + offset));
    const uint32_t pixel = get_pixel_id(index.x, index.y);

    const LightEvalData data = load_light_eval_data(pixel);
    LightSample sample;
    if (data.flags) {
      sample = sample_light(data.position);
    }
    else {
      sample.id     = LIGHT_ID_NONE;
      sample.weight = 0.0f;
    }
    store_light_sample(device.light_samples, sample, pixel);

    if (device_iteration_type != TYPE_CAMERA) {
      device.light_eval_data[pixel].flags = 0;
    }
  }
}

__global__ void spatial_resampling(LightSample* input, LightSample* output) {
  for (int offset = threadIdx.x + blockIdx.x * blockDim.x; offset < device_amount; offset += blockDim.x * gridDim.x) {
    const LightEvalData data  = load_light_eval_data(offset);
    const LightSample current = load_light_sample(input, offset);

    LightSample selected = current;

    if (data.flags) {
      const int x = offset % device_width;
      const int y = offset / device_width;

      for (int i = 0; i < device_spatial_samples; i++) {
        int sample_x = x + (int) (2.0f * (white_noise() - 0.5f) * 30.0f);
        int sample_y = y + (int) (2.0f * (white_noise() - 0.5f) * 30.0f);

        sample_x = max(sample_x, 0);
        sample_y = max(sample_y, 0);
        sample_x = min(sample_x, device_width - 1);
        sample_y = min(sample_y, device_height - 1);

        const LightSample spatial = load_light_sample(input, sample_x + sample_y * device_width);

        if (spatial.id == LIGHT_ID_NONE || spatial.id == selected.id)
          continue;

        selected = resample_light(selected, spatial, data);
      }
    }

    store_light_sample(output, selected, offset);
  }
}

#endif /* CU_LIGHT_H */
