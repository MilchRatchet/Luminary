#ifndef CU_LIGHT_H
#define CU_LIGHT_H

#include "brdf.cuh"
#include "math.cuh"
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
  const int task_count = device.task_counts[(threadIdx.x + blockIdx.x * blockDim.x) * 6 + 5];

  for (int i = 0; i < task_count; i++) {
    const int offset     = get_task_address(i);
    const ushort2 index  = __ldcs((ushort2*) (device_trace_tasks + offset));
    const uint32_t pixel = get_pixel_id(index.x, index.y);

    const LightEvalData data  = load_light_eval_data(pixel);
    const LightSample current = load_light_sample(input, pixel);

    LightSample selected = current;

    if (data.flags) {
      const float ran1 = white_noise();
      const float ran2 = white_noise();

      uint32_t ran_x = 1 + ran1 * 64;
      uint32_t ran_y = 1 + ran2 * 64;

      for (int i = 0; i < device_spatial_samples; i++) {
        ran_x = xorshift_uint32(ran_x);
        ran_y = xorshift_uint32(ran_y);

        int sample_x = index.x + ((ran_x & 0x3f) - 32);
        int sample_y = index.y + ((ran_y & 0x3f) - 32);

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

    store_light_sample(output, selected, pixel);
  }
}

#endif /* CU_LIGHT_H */
