#ifndef CU_RESTIR_H
#define CU_RESTIR_H

#include "memory.cuh"
#include "utils.cuh"

//
// This file implements light sampling based on ReSTIR. However, I ultimately decided to only use
// weighted reservoir sampling using the BRDF as the target pdf. This is because I want to avoid
// introducing any bias here. Initially, the plan was to reuse visibility that was computed in
// the previous frame. However, to achieve an unbiased implementation we need to evaluate the target
// pdf of the chosen sample for all neighbours. If I include visibility then I must reevaluate visibility
// for each neighbour which is not feasible. [Wym21] suggested not computing visibility but using
// heuristics and MIS weights to keep the bias to a minimum but I am not happy with that.
//
// Since I don't use visibility, there is no need for temporal resampling, the main point behind temporal resampling is
// that the chosen candidate from the initial candidate sampling may be occluded and hence through temporal
// resampling we increase the likelihood that a pixel has a non occluded sample ready for spatial resampling.
// Spatial resampling then redistributes those samples so that any single pixel has a high chance of visiting
// a new and likely unoccluded light. Obviously, we also additionally increase the number of samples any pixel
// has visited exponentially with a non exponential cost.
//
// The increased sample quality through spatial resampling turns out to not be worth much in my case where
// no visibility is used in the target pdf. Hence both temporal and spatial resampling are not part of this
// algorithm. This also has some additional benefits in that it reduces memory consumption since we don't need
// two light sample buffers and because light sampling is equally good for all bounce depths.
//

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

  s.seed   = 0;
  s.id     = LIGHT_ID_NONE;
  s.M      = 0;
  s.weight = 0.0f;

  return s;
}

__device__ LightSample restir_sample_update(LightSample x, uint32_t id, float weight, uint32_t random_key, uint32_t& seed) {
  x.M++;

  x.weight += weight;
  if (white_noise_offset(seed++) * x.weight < weight) {
    x.id   = id;
    x.seed = random_key;
  }

  return x;
}

/**
 * Compute the target PDF of a light sample. Currently, this is a basic light intensity multiplied with the solid angle of the object.
 * @param x Light sample
 * @param pos Position to compute the solid angle from.
 * @result Target PDF of light sample.
 */
__device__ float restir_sample_target_pdf(LightSample x, LightEvalData data) {
  if (x.id == LIGHT_ID_NONE) {
    return 0.0f;
  }

  BRDFInstance brdf = brdf_get_instance(get_RGBAhalf(1.0f, 1.0f, 1.0f, 1.0f), data.V, data.normal, data.roughness, data.metallic);

  // We overwrite the local scope copy of the light sample
  x.weight = 1.0f;

  BRDFInstance result = (data.flags & LIGHT_EVAL_DATA_VOLUME_HIT)
                          ? brdf_apply_sample_scattering(brdf, x, data.position, device.scene.ocean.anisotropy)
                          : brdf_apply_sample(brdf, x, data.position);

  float value = luminance(result.term);

  if (isinf(value) || isnan(value)) {
    value = 0.0f;
  }

  switch (x.id) {
    case LIGHT_ID_SUN:
      value *= (2e+04f * device.scene.sky.sun_strength);
      break;
    case LIGHT_ID_TOY:
      value *= device.scene.toy.material.b;
      break;
    default:
      value *= device.scene.material.default_material.b;
      break;
  }

  return value;
}

/**
 * Compute the weight of a light sample. This is supposed to be called after resampling where the weight field contains
 * the sum of weights.
 * @param s Light sample
 * @param pos Position to compute the solid angle from.
 * @result Copy of the LightSample s with computed weight.
 */
__device__ LightSample restir_compute_weight(LightSample s, LightEvalData data) {
  if (s.id == LIGHT_ID_NONE) {
    s.weight = 0.0f;
  }
  else {
    const float target_pdf = restir_sample_target_pdf(s, data);
    s.weight               = (1.0f / target_pdf) * (s.weight / s.M);
  }

  return s;
}

__device__ LightSample restir_sample_reservoir(LightEvalData data, uint32_t& seed) {
  LightSample selected = restir_sample_empty();

  const vec3 sky_pos = world_to_sky_transform(data.position);

  const int sun_visible = !sph_ray_hit_p0(normalize_vector(sub_vector(device.sun_pos, sky_pos)), sky_pos, SKY_EARTH_RADIUS);
  const int toy_visible = (device.scene.toy.active && device.scene.toy.emissive);
  uint32_t light_count  = 0;
  light_count += sun_visible;
  light_count += toy_visible;
  light_count += (device.scene.material.lights_active) ? device.scene.triangle_lights_count : 0;

  if (!light_count)
    return selected;

  float base_probability = 1.0f;

  // Sun as a hero light is being sampled more often
  if (sun_visible) {
    base_probability *= 0.5f;
    if (white_noise_offset(seed++) < 0.5f) {
      LightSample sampled;
      sampled.seed = random_uint32_t(seed++);
      sampled.id   = LIGHT_ID_SUN;
      sampled.M    = 1;

      const float sampled_target_pdf = restir_sample_target_pdf(sampled, data);
      const float sampled_pdf        = base_probability;

      selected = restir_sample_update(selected, sampled.id, sampled_target_pdf / sampled_pdf, sampled.seed, seed);

      selected = restir_compute_weight(selected, data);

      return selected;
    }
  }

  const float light_count_float = ((float) light_count) - 1.0f + 0.9999999f;

  for (int i = 0; i < device.restir.initial_reservoir_size; i++) {
    uint32_t light_index = random_uint32_t(seed++) % light_count;

    light_index += !sun_visible;
    light_index += (!toy_visible && light_index) ? 1 : 0;

    LightSample sampled;
    sampled.seed = random_uint32_t(seed++);
    sampled.id   = LIGHT_ID_NONE;
    sampled.M    = 1;

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

    const float sampled_target_pdf = restir_sample_target_pdf(sampled, data);
    const float sampled_pdf        = base_probability * 1.0f / light_count_float;

    selected = restir_sample_update(selected, sampled.id, sampled_target_pdf / sampled_pdf, sampled.seed, seed);
  }

  selected = restir_compute_weight(selected, data);

  return selected;
}

__global__ void restir_weighted_reservoir_sampling() {
  const int task_count = device.ptrs.task_counts[(threadIdx.x + blockIdx.x * blockDim.x) * 6 + 5];

  uint32_t seed = device.ptrs.randoms[threadIdx.x + blockIdx.x * blockDim.x];

  for (int i = 0; i < task_count; i++) {
    const int offset     = get_task_address(i);
    const ushort2 index  = __ldg((ushort2*) (device.trace_tasks + offset));
    const uint32_t pixel = get_pixel_id(index.x, index.y);

    const LightEvalData data = load_light_eval_data(pixel);

    LightSample sample = (data.flags & LIGHT_EVAL_DATA_REQUIRES_SAMPLING) ? restir_sample_reservoir(data, seed) : restir_sample_empty();

    store_light_sample(device.ptrs.light_samples, sample, pixel);

    if (device.iteration_type != TYPE_CAMERA) {
      device.ptrs.light_eval_data[pixel].flags = 0;
    }
  }

  device.ptrs.randoms[threadIdx.x + blockIdx.x * blockDim.x] = seed;
}

#endif /* CU_RESTIR_H */
