#ifndef CU_LIGHT_H
#define CU_LIGHT_H

#include "brdf.cuh"
#include "math.cuh"
#include "utils.cuh"

__global__ void generate_light_samples() {
  const int task_count = device.ptrs.task_counts[(threadIdx.x + blockIdx.x * blockDim.x) * 6 + 5];

  uint32_t ran_offset = device.ptrs.randoms[threadIdx.x + blockIdx.x * blockDim.x];

  for (int i = 0; i < task_count; i++) {
    const int offset     = get_task_address(i);
    const ushort2 index  = __ldg((ushort2*) (device.trace_tasks + offset));
    const uint32_t pixel = get_pixel_id(index.x, index.y);

    const LightEvalData data = load_light_eval_data(pixel);
    LightSample sample;
    if (data.flags) {
      sample = sample_light(data.position, ran_offset);
    }
    else {
      sample.id     = LIGHT_ID_NONE;
      sample.weight = 0.0f;
    }
    store_light_sample(device.ptrs.light_samples, sample, pixel);

    if (device.iteration_type != TYPE_CAMERA) {
      device.ptrs.light_eval_data[pixel].flags = 0;
    }
  }

  device.ptrs.randoms[threadIdx.x + blockIdx.x * blockDim.x] = ran_offset;
}

__global__ void spatial_resampling(LightSample* input, LightSample* output) {
  const int task_count = device.ptrs.task_counts[(threadIdx.x + blockIdx.x * blockDim.x) * 6 + 5];

  uint32_t ran_offset = device.ptrs.randoms[threadIdx.x + blockIdx.x * blockDim.x];

  for (int i = 0; i < task_count; i++) {
    const int offset     = get_task_address(i);
    const ushort2 index  = __ldcs((ushort2*) (device.trace_tasks + offset));
    const uint32_t pixel = get_pixel_id(index.x, index.y);

    const LightEvalData data  = load_light_eval_data(pixel);
    const LightSample current = load_light_sample(input, pixel);

    LightSample selected = current;

    if (data.flags) {
      for (int i = 0; i < device.spatial_samples; i++) {
        const uint32_t ran_x = random_uint32_t(ran_offset++);
        const uint32_t ran_y = random_uint32_t(ran_offset++);

        int sample_x = index.x + ((ran_x & 0x1f) - 16);
        int sample_y = index.y + ((ran_y & 0x1f) - 16);

        sample_x = max(sample_x, 0);
        sample_y = max(sample_y, 0);
        sample_x = min(sample_x, device.width - 1);
        sample_y = min(sample_y, device.height - 1);

        LightSample spatial = load_light_sample(input, sample_x + sample_y * device.width);

        if (spatial.id == LIGHT_ID_NONE)
          continue;

        const float target_weight = brdf_light_sample_target_weight(spatial);

        spatial.solid_angle = brdf_light_sample_solid_angle(spatial, data.position);

        const float target_weight_resampled = brdf_light_sample_target_weight(spatial);

        const float r = white_noise_offset(ran_offset++);

        selected = brdf_light_sample_update(selected, spatial, spatial.weight * (target_weight_resampled / target_weight), r);
      }
    }

    store_light_sample(output, selected, pixel);
  }

  device.ptrs.randoms[threadIdx.x + blockIdx.x * blockDim.x] = ran_offset;
}

#endif /* CU_LIGHT_H */
