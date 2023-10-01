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
// High-Performance Graphics - Symposium Papers, pp. 23-41, 2021

/**
 * Returns an empty light sample.
 * @result Empty light sample.
 */
__device__ LightSample restir_sample_empty() {
  LightSample s;

  s.seed          = 0;
  s.presampled_id = LIGHT_ID_NONE;
  s.id            = LIGHT_ID_NONE;
  s.weight        = 0.0f;

  return s;
}

/**
 * Compute the target PDF of a light sample. Currently, this is basically BRDF weight multiplied by general light intensity.
 * @param x Light sample
 * @param data Data to compute the target pdf from.
 * @result Target PDF of light sample.
 */
__device__ float restir_sample_target_pdf(LightSample x, const GBufferData data) {
  if (x.presampled_id == LIGHT_ID_NONE) {
    return 0.0f;
  }

  // We overwrite the local scope copy of the light sample
  x.weight = 1.0f;

  BRDFInstance brdf = brdf_get_instance(data.albedo, data.V, data.normal, data.roughness, data.metallic);
  BRDFInstance result;

  if (data.flags & G_BUFFER_VOLUME_HIT) {
    result = brdf_apply_sample_scattering(brdf, x, data.position, VOLUME_HIT_TYPE(data.hit_id));
  }
  else {
    result = brdf_apply_sample(brdf, x, data.position);
  }

  float value = luminance(result.term);

  if (isinf(value) || isnan(value)) {
    value = 0.0f;
  }

  switch (x.presampled_id) {
    case LIGHT_ID_SUN:
      value *= (2e+04f * device.scene.sky.sun_strength);
      break;
    case LIGHT_ID_TOY:
      value *= device.scene.toy.material.b * luminance(device.scene.toy.emission);
      break;
    default:
      value *= device.scene.material.default_material.b;
      break;
  }

  return value;
}

/*
 * There are three different MIS weights implemented:
 * Uniform, balance heuristic (power = 1) and power heuristic (power = 2).
 * In general, power heuristic is known to perform the best and uniform the worst.
 * However, since the multiple sampling functions domains form a partition of the total sampling
 * domain, and each sampling function is uniform over its domain, the uniform MIS weights perform the best.
 * The balance heuristic produces identical results but it is also slightly more expensive computationally.
 * However, I want to keep the code simple, the balance heuristic enforces a system of pdf tracking that
 * will make it easier for me if I ever decide to implement other sampling strategies.
 */
#define RESTIR_MIS_WEIGHT_METHOD 1

/**
 * Samples a light sample using WRS and MIS.
 * @param data Data used for evaluating a lights importance.
 * @param seed Seed used for random number generation, the seed is overwritten on use.
 * @result Sampled light sample.
 */
__device__ LightSample restir_sample_reservoir(GBufferData data, uint32_t& seed) {
  LightSample selected = restir_sample_empty();

  const vec3 sky_pos = world_to_sky_transform(data.position);

  const int sun_visible = !sph_ray_hit_p0(normalize_vector(sub_vector(device.sun_pos, sky_pos)), sky_pos, SKY_EARTH_RADIUS);
  const int toy_visible = (device.scene.toy.active && device.scene.toy.emissive);

  // Sampling pdf used when the selected light was selected, this is used for the MIS balance heuristic
  float selection_pdf        = 1.0f;
  float selection_target_pdf = 1.0f;

  // Importance sample the sun
  if (sun_visible) {
    LightSample sampled;
    sampled.seed          = random_uint32_t(seed++);
    sampled.presampled_id = LIGHT_ID_SUN;

    const float sampled_target_pdf = restir_sample_target_pdf(sampled, data);
    const float sampled_pdf        = 1.0f;

    const float weight = sampled_target_pdf / sampled_pdf;

    selected.weight += weight;
    if (white_noise_offset(seed++) * selected.weight < weight) {
      selected.id            = LIGHT_ID_SUN;
      selected.presampled_id = LIGHT_ID_SUN;
      selected.seed          = sampled.seed;
      selection_pdf          = sampled_pdf;
      selection_target_pdf   = sampled_target_pdf;
    }
  }

  // Importance sample the toy (but only if we are not originating from the toy)
  if (toy_visible && data.hit_id != HIT_TYPE_TOY) {
    LightSample sampled;
    sampled.seed          = random_uint32_t(seed++);
    sampled.presampled_id = LIGHT_ID_TOY;

    const float sampled_target_pdf = restir_sample_target_pdf(sampled, data);
    const float sampled_pdf        = 1.0f;

    const float weight = sampled_target_pdf / sampled_pdf;

    selected.weight += weight;
    if (white_noise_offset(seed++) * selected.weight < weight) {
      selected.id            = LIGHT_ID_TOY;
      selected.presampled_id = LIGHT_ID_TOY;
      selected.seed          = sampled.seed;
      selection_pdf          = sampled_pdf;
      selection_target_pdf   = sampled_target_pdf;
    }
  }

  const uint32_t light_count = (device.scene.material.lights_active) ? (1 << device.restir.light_candidate_pool_size_log2) : 0;
  const int reservoir_size   = (device.scene.triangle_lights_count > 0) ? min(device.restir.initial_reservoir_size, light_count) : 0;

  const float reservoir_sampling_pdf = (1.0f / device.scene.triangle_lights_count);

  // Don't allow triangles to sample themselves.
  uint32_t blocked_light_id = LIGHT_ID_TRIANGLE_ID_LIMIT + 1;
  if (data.hit_id <= LIGHT_ID_TRIANGLE_ID_LIMIT) {
    blocked_light_id = __ldg(&(device.scene.triangles[data.hit_id].light_id));
  }

  for (int i = 0; i < reservoir_size; i++) {
    LightSample sampled;
    sampled.seed          = random_uint32_t(seed++);
    sampled.presampled_id = random_uint32_t(seed++) & (light_count - 1);

    const uint32_t sampled_light_global_id = device.ptrs.light_candidates[sampled.presampled_id];

    if (sampled_light_global_id == blocked_light_id)
      continue;

    const float sampled_target_pdf = restir_sample_target_pdf(sampled, data);
    const float sampled_pdf        = reservoir_sampling_pdf;

    const float weight = sampled_target_pdf / sampled_pdf;

    selected.weight += weight;
    if (white_noise_offset(seed++) * selected.weight < weight) {
      selected.id            = sampled_light_global_id;
      selected.presampled_id = sampled.presampled_id;
      selected.seed          = sampled.seed;
      selection_pdf          = sampled_pdf;
      selection_target_pdf   = sampled_target_pdf;
    }
  }

  // Compute the shading weight of the selected light (Probability of selecting the light through WRS)
  if (selected.id == LIGHT_ID_NONE) {
    selected.weight = 0.0f;
  }
  else {
#if RESTIR_MIS_WEIGHT_METHOD == 0
    // Compute the sum of the sampling PDFs of the selected light for MIS uniform
    float sum_pdf = 0.0f;
    sum_pdf += (sun_visible && selected.id == LIGHT_ID_SUN) ? 1.0f : 0.0f;
    sum_pdf += (toy_visible && selected.id == LIGHT_ID_TOY) ? 1.0f : 0.0f;
    sum_pdf += (selected.id <= LIGHT_ID_TRIANGLE_ID_LIMIT) ? reservoir_size : 0.0f;

    const float mis_weight = 1.0f / sum_pdf;
#elif RESTIR_MIS_WEIGHT_METHOD == 1
    // Compute the sum of the sampling PDFs of the selected light for MIS balance heuristic
    float sum_pdf = 0.0f;
    sum_pdf += (sun_visible && selected.id == LIGHT_ID_SUN) ? 1.0f : 0.0f;
    sum_pdf += (toy_visible && selected.id == LIGHT_ID_TOY) ? 1.0f : 0.0f;
    sum_pdf += (selected.id <= LIGHT_ID_TRIANGLE_ID_LIMIT) ? reservoir_size * reservoir_sampling_pdf : 0.0f;

    const float mis_weight = selection_pdf / sum_pdf;
#else
    // Compute the sum of the sampling PDFs of the selected light for MIS power heuristic
    float sum_pdf = 0.0f;
    sum_pdf += (sun_visible && selected.id == LIGHT_ID_SUN) ? 1.0f : 0.0f;
    sum_pdf += (toy_visible && selected.id == LIGHT_ID_TOY) ? 1.0f : 0.0f;
    sum_pdf += (selected.id <= LIGHT_ID_TRIANGLE_ID_LIMIT)
                 ? reservoir_size * reservoir_sampling_pdf * reservoir_size * reservoir_sampling_pdf
                 : 0.0f;

    const float mis_weight = (selection_pdf * selection_pdf) / sum_pdf;
#endif

    selected.weight = mis_weight * selected.weight / selection_target_pdf;
  }

  return selected;
}

/**
 * Kernel that determines a light sample to be used in next event estimation
 *
 * Light sample is stored in device.ptrs.light_samples.
 *
 * Light samples are only generated for pixels for which the flag G_BUFFER_REQUIRES_SAMPLING is set.
 */
__global__ void restir_weighted_reservoir_sampling() {
  const int task_count = device.ptrs.task_counts[(threadIdx.x + blockIdx.x * blockDim.x) * 6 + 5];

  uint32_t seed = device.ptrs.randoms[threadIdx.x + blockIdx.x * blockDim.x];

  for (int i = 0; i < task_count; i++) {
    const int offset     = get_task_address(i);
    const ushort2 index  = __ldg((ushort2*) (device.trace_tasks + offset));
    const uint32_t pixel = get_pixel_id(index.x, index.y);

    const GBufferData data = load_g_buffer_data(pixel);

    LightSample sample = (data.flags & G_BUFFER_REQUIRES_SAMPLING) ? restir_sample_reservoir(data, seed) : restir_sample_empty();

    store_light_sample(device.ptrs.light_samples, sample, pixel);

    if (device.iteration_type != TYPE_CAMERA) {
      device.ptrs.g_buffer[pixel].flags = data.flags & (~G_BUFFER_REQUIRES_SAMPLING);
    }
  }

  device.ptrs.randoms[threadIdx.x + blockIdx.x * blockDim.x] = seed;
}

__global__ void restir_candidates_pool_generation() {
  int id = threadIdx.x + blockIdx.x * blockDim.x;

  uint32_t seed = device.ptrs.randoms[threadIdx.x + blockIdx.x * blockDim.x];

  const int light_sample_bin_count = 1 << device.restir.light_candidate_pool_size_log2;

  if (device.scene.triangle_lights_count == 0)
    return;

  while (id < light_sample_bin_count) {
    const uint32_t sampled_id = random_uint32_t(seed++) % device.scene.triangle_lights_count;

    device.ptrs.light_candidates[id]             = sampled_id;
    device.restir.presampled_triangle_lights[id] = device.scene.triangle_lights[sampled_id];

    id += blockDim.x * gridDim.x;
  }

  device.ptrs.randoms[threadIdx.x + blockIdx.x * blockDim.x] = seed;
}

#endif /* CU_RESTIR_H */
