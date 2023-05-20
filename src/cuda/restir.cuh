#ifndef CU_RESTIR_H
#define CU_RESTIR_H

#include "memory.cuh"
#include "utils.cuh"

////////////////////////////////////////////////////////////////////
// Literature
////////////////////////////////////////////////////////////////////

// [Bit20]
// B. Bitterli, C. Wyman, M. Pharr, P. Shirley, A. Lefohn, W. Jarosz,
// "Spatiotemporal reservoir resampling for real-time ray tracing with dynamic direct lighting",
// ACM Transactions on Graphics (Proceedings of SIGGRAPH), 39(4), 2020.

// [Wym21]
// C. Wyman, A. Panteleev, "Rearchitecting Spatiotemporal Resampling for Production",
// High-Performance Graphics - Symposium Papers, 2021

__device__ LightSample restir_sample_empty() {
  LightSample s;

  s.id         = LIGHT_ID_NONE;
  s.M          = 0;
  s.target_pdf = 0.0f;
  s.weight     = 0.0f;

  return s;
}

__device__ LightSample restir_sample_update(LightSample x, uint32_t id, float weight, uint32_t& seed) {
  x.weight += weight;
  x.M++;
  if (white_noise_offset(seed++) * x.weight < weight) {
    x.id = id;
  }

  return x;
}

/**
 * Compute the target PDF of a light sample. Currently, this is a basic light intensity multiplied with the solid angle of the object.
 * @param x Light sample
 * @param pos Position to compute the solid angle from.
 * @result Target PDF of light sample.
 */
__device__ float restir_sample_target_pdf(LightSample x, vec3 pos) {
  switch (x.id) {
    case LIGHT_ID_SUN:
      return (2e+04f * device.scene.sky.sun_strength)
             * sample_sphere_solid_angle(device.sun_pos, SKY_SUN_RADIUS, world_to_sky_transform(pos));
    case LIGHT_ID_TOY:
      return device.scene.toy.material.b * toy_get_solid_angle(pos);
    case LIGHT_ID_NONE:
      return 0.0f;
    default:
      const TriangleLight triangle = load_triangle_light(x.id);
      return device.scene.material.default_material.b * sample_triangle_solid_angle(triangle, pos);
  }
}

/**
 * Compute the weight of a light sample. This is supposed to be called after resampling where the weight field contains
 * the sum of weights.
 * @param s Light sample
 * @param pos Position to compute the solid angle from.
 * @result Copy of the LightSample s with computed weight.
 */
__device__ LightSample restir_compute_weight(LightSample s, vec3 pos) {
  if (s.id == LIGHT_ID_NONE) {
    s.weight     = 0.0f;
    s.target_pdf = 0.0f;
  }
  else {
    s.target_pdf = restir_sample_target_pdf(s, pos);
    s.weight     = (1.0f / s.target_pdf) * (s.weight / s.M);
  }

  return s;
}

__device__ LightSample restir_sample_reservoir(vec3 pos, uint32_t& seed) {
  LightSample selected = restir_sample_empty();

  const vec3 sky_pos = world_to_sky_transform(pos);

  const int sun_visible = !sph_ray_hit_p0(normalize_vector(sub_vector(device.sun_pos, sky_pos)), sky_pos, SKY_EARTH_RADIUS);
  const int toy_visible = (device.scene.toy.active && device.scene.toy.emissive);
  uint32_t light_count  = 0;
  light_count += sun_visible;
  light_count += toy_visible;
  light_count += (device.scene.material.lights_active) ? device.scene.triangle_lights_count : 0;

  if (!light_count)
    return selected;

  // Sun as a hero light is being sampled more often
  if (sun_visible && white_noise_offset(seed++) < 0.5f) {
    LightSample sampled;
    sampled.id = LIGHT_ID_SUN;
    sampled.M  = 1;

    const float sampled_target_pdf = restir_sample_target_pdf(sampled, pos);
    const float sampled_pdf        = 0.5f;

    selected = restir_sample_update(selected, sampled.id, sampled_target_pdf / sampled_pdf, seed);

    selected = restir_compute_weight(selected, pos);

    return selected;
  }

  const float light_count_float = ((float) light_count) - 1.0f + 0.9999999f;

  for (int i = 0; i < device.reservoir_size; i++) {
    uint32_t light_index = random_uint32_t(seed++) % light_count;

    light_index += !sun_visible;
    light_index += (!toy_visible && light_index) ? 1 : 0;

    LightSample sampled;
    sampled.id = LIGHT_ID_NONE;
    sampled.M  = 1;

    switch (light_index) {
      case 0:
        sampled.id = LIGHT_ID_SUN;
        break;
      case 1:
        sampled.id = LIGHT_ID_TOY;
        break;
      default:
        sampled.id = light_index - 2;
        break;
    }

    const float sampled_target_pdf = restir_sample_target_pdf(sampled, pos);
    const float sampled_pdf        = 1.0f / light_count_float;

    selected = restir_sample_update(selected, sampled.id, sampled_target_pdf / sampled_pdf, seed);
  }

  selected = restir_compute_weight(selected, pos);

  return selected;
}

__device__ LightSample restir_combine_reservoirs(LightSample x, LightSample y, vec3 pos, uint32_t& seed) {
  LightSample s = restir_sample_empty();

  s = restir_sample_update(s, x.id, x.weight * x.M * x.target_pdf, seed);
  s = restir_sample_update(s, y.id, y.weight * y.M * y.target_pdf, seed);

  s.M = x.M + y.M;

  s = restir_compute_weight(s, pos);

  return s;
}

__global__ void restir_initial_sampling(LightSample* samples) {
  const int task_count = device.ptrs.task_counts[(threadIdx.x + blockIdx.x * blockDim.x) * 6 + 5];

  uint32_t seed = device.ptrs.randoms[threadIdx.x + blockIdx.x * blockDim.x];

  for (int i = 0; i < task_count; i++) {
    const int offset     = get_task_address(i);
    const ushort2 index  = __ldg((ushort2*) (device.trace_tasks + offset));
    const uint32_t pixel = get_pixel_id(index.x, index.y);

    const LightEvalData data = load_light_eval_data(pixel);

    LightSample sample = (data.flags) ? restir_sample_reservoir(data.position, seed) : restir_sample_empty();

    store_light_sample(device.ptrs.light_samples, sample, pixel);

    if (device.iteration_type != TYPE_CAMERA) {
      device.ptrs.light_eval_data[pixel].flags = 0;
    }
  }

  device.ptrs.randoms[threadIdx.x + blockIdx.x * blockDim.x] = seed;
}

__global__ void restir_temporal_resampling(LightSample* samples, LightSample* samples_prev) {
  const int task_count = device.ptrs.task_counts[(threadIdx.x + blockIdx.x * blockDim.x) * 6 + 5];

  uint32_t seed = device.ptrs.randoms[threadIdx.x + blockIdx.x * blockDim.x];

  for (int i = 0; i < task_count; i++) {
    const int offset     = get_task_address(i);
    const ushort2 index  = __ldcs((ushort2*) (device.trace_tasks + offset));
    const uint32_t pixel = get_pixel_id(index.x, index.y);

    const LightEvalData data  = load_light_eval_data(pixel);
    const LightSample current = load_light_sample(samples, pixel);

    LightSample selected = current;

    if (data.flags) {
      LightSample temporal = load_light_sample(samples_prev, pixel);

      temporal.M = min(selected.M * 20, temporal.M);

      selected = restir_combine_reservoirs(selected, temporal, data.position, seed);
    }

    store_light_sample(samples, selected, pixel);
  }

  device.ptrs.randoms[threadIdx.x + blockIdx.x * blockDim.x] = seed;
}

__global__ void restir_temporal_visibility(LightSample* samples) {
  const int amount = device.width * device.height;

  for (int pixel = threadIdx.x + blockIdx.x * blockDim.x; pixel < amount; pixel += blockDim.x * gridDim.x) {
    LightSample current   = load_light_sample(samples, pixel);
    const RGBAhalf result = load_RGBAhalf(device.ptrs.frame_buffer + pixel);

    if (!any_RGBAhalf(result)) {
      current.weight = 0.0f;
    }

    store_light_sample(samples, current, pixel);
  }
}

__global__ void restir_spatial_resampling(LightSample* input, LightSample* output) {
  const int task_count = device.ptrs.task_counts[(threadIdx.x + blockIdx.x * blockDim.x) * 6 + 5];

  uint32_t seed = device.ptrs.randoms[threadIdx.x + blockIdx.x * blockDim.x];

  for (int i = 0; i < task_count; i++) {
    const int offset     = get_task_address(i);
    const ushort2 index  = __ldcs((ushort2*) (device.trace_tasks + offset));
    const uint32_t pixel = get_pixel_id(index.x, index.y);

    const LightEvalData data  = load_light_eval_data(pixel);
    const LightSample current = load_light_sample(input, pixel);

    LightSample selected = current;

    if (data.flags) {
      for (int i = 0; i < device.spatial_samples; i++) {
        const uint32_t ran_x = random_uint32_t(seed++);
        const uint32_t ran_y = random_uint32_t(seed++);

        int sample_x = index.x + ((ran_x & 0x1f) - 16);
        int sample_y = index.y + ((ran_y & 0x1f) - 16);

        sample_x = max(sample_x, 0);
        sample_y = max(sample_y, 0);
        sample_x = min(sample_x, device.width - 1);
        sample_y = min(sample_y, device.height - 1);

        LightSample spatial = load_light_sample(input, sample_x + sample_y * device.width);

        selected = restir_combine_reservoirs(selected, spatial, data.position, seed);
      }
    }

    store_light_sample(output, selected, pixel);
  }

  device.ptrs.randoms[threadIdx.x + blockIdx.x * blockDim.x] = seed;
}

#endif /* CU_RESTIR_H */
