#ifndef CU_RIS_H
#define CU_RIS_H

#include "light.cuh"
#include "math.cuh"
#include "random.cuh"
#include "utils.cuh"
#include "volume_utils.cuh"

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
//
// [Cik22]
// E. Ciklabakkal, A. Gruson, I. Georgiev, D. Nowrouzezahrai, T. Hachisuka, "Single-pass stratified importance resampling",
// Computer Graphics Forum (Proceedings of EGSR), 2022

// For light tree based technique, we stratify the RIS light tree random number dimensions using 1D latin hypercube sampling
// to achieve the desired ordered stratification. The other dimensions are entirely dependent on this first dimension so it makes no
// sense to apply any further stratification for those within RIS.
__device__ float ris_transform_stratum(const uint32_t index, const uint32_t num_samples, const float random) {
  const float section_length = 1.0f / num_samples;

  return (index + random) * section_length;
}

__device__ float2 ris_transform_stratum_2D(const uint32_t index, const uint32_t num_samples, const float2 random) {
  return make_float2(ris_transform_stratum(index, num_samples, random.x), random.y);
}

#if defined(SHADING_KERNEL) && !defined(VOLUME_KERNEL)

__device__ TriangleHandle ris_sample_light(
  const GBufferData data, const ushort2 pixel, const uint32_t bsdf_sample_light_key, const vec3 bsdf_sample_ray,
  const bool initial_is_refraction, vec3& selected_ray, RGBF& selected_light_color, float& selected_dist, bool& selected_is_refraction) {
  const uint32_t num_samples = (IS_PRIMARY_RAY) ? device.settings.light_num_ris_samples : 1;

  float sum_weights_front = 0.0f;
  float sum_weights_back  = 0.0f;

  selected_ray           = get_vector(0.0f, 0.0f, 1.0f);
  selected_light_color   = get_color(0.0f, 0.0f, 0.0f);
  selected_dist          = 1.0f;
  selected_is_refraction = false;

  TriangleHandle selected_handle = triangle_handle_get(LIGHT_ID_NONE, 0);

  // Don't allow triangles to sample themselves.
  const TriangleHandle blocked_handle = triangle_handle_get(data.instance_id, data.tri_id);

  ////////////////////////////////////////////////////////////////////
  // Initialize reservoir with an initial sample
  ////////////////////////////////////////////////////////////////////

#ifndef DL_GEO_NO_BSDF_SAMPLE
  TriangleHandle bsdf_sample_handle = blocked_handle;
  if (bsdf_sample_light_key != HIT_TYPE_LIGHT_BSDF_HINT) {
    bsdf_sample_handle = device.ptrs.light_tree_tri_handle_map[bsdf_sample_light_key];
  }

  if (!triangle_handle_equal(bsdf_sample_handle, blocked_handle)) {
    const DeviceTransform trans = load_transform(bsdf_sample_handle.instance_id);

    float dist;
    const TriangleLight triangle_light = light_load(bsdf_sample_handle, data.position, bsdf_sample_ray, trans, dist);
    const float solid_angle            = light_get_solid_angle(triangle_light, data.position);

    if (dist < FLT_MAX && solid_angle > 0.0f) {
      RGBF light_color = light_get_color(triangle_light);

      bool is_refraction;
      const RGBF bsdf_weight = bsdf_evaluate(data, bsdf_sample_ray, BSDF_SAMPLING_GENERAL, is_refraction);
      light_color            = mul_color(light_color, bsdf_weight);

      const float target_pdf = color_importance(light_color);

      const float bsdf_sample_pdf = bsdf_sample_for_light_pdf(data, bsdf_sample_ray);

      float mis_weight;
      if (num_samples > 0) {
        const float nee_light_tree_pdf      = light_tree_query_pdf(data, bsdf_sample_light_key);
        const float one_over_nee_sample_pdf = solid_angle / (nee_light_tree_pdf * num_samples);

        // MIS weight pre multiplied with inverse of pdf, little trick by using inverse of NEE pdf, this is fine because NEE pdf is never 0.
        mis_weight = bsdf_sample_pdf * one_over_nee_sample_pdf * one_over_nee_sample_pdf
                     / (bsdf_sample_pdf * bsdf_sample_pdf * one_over_nee_sample_pdf * one_over_nee_sample_pdf + 1.0f);
      }
      else {
        mis_weight = 1.0f / bsdf_sample_pdf;
      }

      const float weight = target_pdf * mis_weight;

      // First sample is trivially always the front sample
      sum_weights_front += weight;

      selected_handle        = bsdf_sample_handle;
      selected_light_color   = (target_pdf > 0.0f) ? scale_color(light_color, 1.0f / target_pdf) : get_color(0.0f, 0.0f, 0.0f);
      selected_ray           = bsdf_sample_ray;
      selected_dist          = dist;
      selected_is_refraction = initial_is_refraction;
    }
  }
#endif /* !DL_GEO_NO_BSDF_SAMPLE */

  ////////////////////////////////////////////////////////////////////
  // Sample light tree
  ////////////////////////////////////////////////////////////////////

  const float2 light_tree_random = quasirandom_sequence_2D(QUASI_RANDOM_TARGET_RIS_LIGHT_TREE, pixel);

  LightTreeStackEntry stack[LIGHT_TREE_STACK_SIZE];

  uint32_t num_tree_samples;
  float sum_importance_tree_samples;
  light_tree_query(data, light_tree_random, stack, num_tree_samples, sum_importance_tree_samples);

  ////////////////////////////////////////////////////////////////////
  // Resample NEE samples
  ////////////////////////////////////////////////////////////////////

#if 1
  const float random_light_id = quasirandom_sequence_1D(QUASI_RANDOM_TARGET_RIS_LIGHT_ID, pixel);

  const float target           = random_light_id * sum_importance_tree_samples;
  float accumulated_importance = 0.0f;

  uint32_t sample_id = 0;
  for (; sample_id < num_tree_samples; sample_id++) {
    const float sample_importance = light_tree_bfloat_to_float(stack[sample_id].importance);
    accumulated_importance += sample_importance;

    if (accumulated_importance >= target) {
      break;
    }
  }

  const float sample_importance = light_tree_bfloat_to_float(stack[sample_id].importance);

  const uint32_t light_id    = stack[sample_id].id;
  float selected_probability = sample_importance / sum_importance_tree_samples;
  selected_probability *= light_tree_unpack_probability(stack[sample_id].final_prob);

  DeviceTransform trans;
  const TriangleHandle light_handle = light_tree_get_light(light_id, trans);

  if (triangle_handle_equal(light_handle, blocked_handle))
    return triangle_handle_get(LIGHT_ID_NONE, 0);

  uint3 light_uv_packed;
  TriangleLight triangle_light = light_load_sample_init(light_handle, trans, light_uv_packed);

  const float2 ray_random = quasirandom_sequence_2D(QUASI_RANDOM_TARGET_RIS_RAY_DIR, pixel);

  vec3 ray;
  float dist, solid_angle;
  light_load_sample_finalize(triangle_light, light_uv_packed, data.position, ray_random, ray, dist, solid_angle);

  if (dist == FLT_MAX || solid_angle == 0.0f)
    return triangle_handle_get(LIGHT_ID_NONE, 0);

  RGBF light_color = light_get_color(triangle_light);

  bool is_refraction;
  const RGBF bsdf_weight = bsdf_evaluate(data, ray, BSDF_SAMPLING_GENERAL, is_refraction);
  light_color            = mul_color(light_color, bsdf_weight);

  const float one_over_nee_sample_pdf = solid_angle / selected_probability;

  light_color = scale_color(light_color, one_over_nee_sample_pdf);

  selected_light_color   = light_color;
  selected_ray           = ray;
  selected_dist          = dist;
  selected_is_refraction = is_refraction;

  return light_handle;

#else

  // The paper first computes the first and last candidate and then enters the common logic,
  // to simplify the code, we prepend a candidate to the front and back each and just pretend that they
  // are outside the support of the target PDF, this way, all the logic can be inside the loop.
  uint32_t index_front = (uint32_t) -1;
  uint32_t index_back  = num_samples;

  const float resampling_random = quasirandom_sequence_1D(QUASI_RANDOM_TARGET_RIS_RESAMPLING, pixel);

  for (uint32_t iteration = 0; iteration <= num_samples; iteration++) {
    const bool compute_front = (sum_weights_front <= resampling_random * (sum_weights_front + sum_weights_back));

    if (!compute_front && iteration == num_samples)
      break;

    uint32_t current_index;
    if (compute_front) {
      current_index = ++index_front;
    }
    else {
      current_index = --index_back;
    }

    // This happens if all samples had a weight of zero
    if (current_index == num_samples)
      break;

    const float random_light_id =
      ris_transform_stratum(current_index, num_samples, quasirandom_sequence_1D(QUASI_RANDOM_TARGET_RIS_LIGHT_ID + current_index, pixel));

    const uint32_t random_index = (uint32_t) ((num_tree_samples - 1) * random_light_id + 0.5f);

    LightTreeStackEntry entry = stack[random_index];

    // Entry is a node, that should never happen.
    if ((entry.id & LIGHT_TREE_STACK_FLAG_NODE) != 0)
      continue;

    DeviceTransform trans;
    const TriangleHandle light_handle = light_tree_get_light(entry.id, trans);

    if (triangle_handle_equal(light_handle, blocked_handle))
      continue;

    uint3 light_uv_packed;
    TriangleLight triangle_light = light_load_sample_init(light_handle, trans, light_uv_packed);

    const float2 ray_random = quasirandom_sequence_2D(QUASI_RANDOM_TARGET_RIS_RAY_DIR + current_index, pixel);

    vec3 ray;
    float dist, solid_angle;
    light_load_sample_finalize(triangle_light, light_uv_packed, data.position, ray_random, ray, dist, solid_angle);

    if (dist == FLT_MAX || solid_angle == 0.0f)
      continue;

    RGBF light_color = light_get_color(triangle_light);

    bool is_refraction;
    const RGBF bsdf_weight = bsdf_evaluate(data, ray, BSDF_SAMPLING_GENERAL, is_refraction);
    light_color            = mul_color(light_color, bsdf_weight);
    const float target_pdf = color_importance(light_color);

    if (target_pdf == 0.0f)
      continue;

    if (compute_front) {
      selected_handle        = light_handle;
      selected_light_color   = scale_color(light_color, 1.0f / target_pdf);
      selected_ray           = ray;
      selected_dist          = dist;
      selected_is_refraction = is_refraction;
    }

    // Last iteration cannot add to the weight sums to avoid double counting
    if (iteration == num_samples)
      break;

#ifndef DL_GEO_NO_BSDF_SAMPLE
    const float bsdf_sample_pdf         = bsdf_sample_for_light_pdf(data, ray);
    const float one_over_nee_sample_pdf = solid_angle / ((float) num_samples * light_tree_pdf);

    const float mis_weight =
      one_over_nee_sample_pdf / (bsdf_sample_pdf * bsdf_sample_pdf * one_over_nee_sample_pdf * one_over_nee_sample_pdf + 1.0f);
#else  /* !DL_GEO_NO_BSDF_SAMPLE */
    const float mis_weight = solid_angle * num_tree_samples / (num_samples * entry.pdf);
#endif /* DL_GEO_NO_BSDF_SAMPLE */

    const float weight = target_pdf * mis_weight;

    if (compute_front) {
      sum_weights_front += weight;
    }
    else {
      sum_weights_back += weight;
    }
  }

  ////////////////////////////////////////////////////////////////////
  // Compute the shading weight of the selected light (Probability of selecting the light through WRS)
  ////////////////////////////////////////////////////////////////////

  // Selected light color already includes 1 / target_pdf.
  selected_light_color = scale_color(selected_light_color, sum_weights_front + sum_weights_back);

  UTILS_CHECK_NANS(pixel, selected_light_color);

  return selected_handle;
#endif
}
#endif /* SHADING_KERNEL && !VOLUME_KERNEL */

#endif /* CU_RIS_H */
