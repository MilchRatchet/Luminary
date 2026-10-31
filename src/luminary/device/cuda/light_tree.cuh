#ifndef CU_LUMINARY_LIGHT_TREE_H
#define CU_LUMINARY_LIGHT_TREE_H

#include "material.cuh"
#include "math.cuh"
#include "memory.cuh"
#include "ris.cuh"
#include "utils.cuh"

#define LIGHT_TREE_NUM_OUTPUTS 8
#define LIGHT_TREE_INVALID_NODE 0xFFFFFF
#define LIGHT_TREE_SELECTED_IS_LIGHT 0x80000000
#define LIGHT_TREE_SELECTED_PTR_MASK LIGHT_TREE_INVALID_NODE
// #define LIGHT_TREE_DEBUG_TRAVERSAL

static_assert(LIGHT_TREE_NUM_OUTPUTS <= LIGHT_GEO_MAX_SAMPLES, "Update random allocations if you increase number of output samples.");

struct LightTreeContinuation {
  uint32_t is_light : 1, child_index : 8, probability : 20;  // 3 Spare
  static_assert((1 << 8) >= LIGHT_TREE_ROOT_MAX_CHILD_COUNT, "There are too many root children, increase the number of child index bits!");
} typedef LightTreeContinuation;
LUM_STATIC_SIZE_ASSERT(LightTreeContinuation, 0x04);

struct LightTreeResult {
  uint32_t light_id;
  float weight;
} typedef LightTreeResult;
LUM_STATIC_SIZE_ASSERT(LightTreeResult, 0x08);

struct LightTreeWork {
  LightTreeContinuation data[LIGHT_TREE_NUM_OUTPUTS];
  float root_sum;
} typedef LightTreeWork;

#ifdef LIGHT_TREE_DEBUG_TRAVERSAL
#define _LIGHT_TREE_DEBUG_SELECT_CHILD_TOKEN(__lane_id, __selected, __target) \
  if (is_center_pixel(pixel)) {                                               \
    printf("@%u CSEL 0x%08X [Weight:%f]\n", __lane_id, __selected, __target); \
  }
#define _LIGHT_TREE_DEBUG_STORE_CONTINUATION_TOKEN(__lane_id, __continuation)                                                           \
  if (is_center_pixel(pixel)) {                                                                                                         \
    printf(                                                                                                                             \
      "@%u ST.CONTINUATION.%s 0x%08X [Weight:%f]\n", __lane_id, __continuation.is_light ? "LIGHT" : "NODE", __continuation.child_index, \
      1.0f / _light_tree_continuation_unpack_prob(__continuation));                                                                     \
  }
#define _LIGHT_TREE_DEBUG_LOAD_CONTINUATION_TOKEN(__lane_id, __continuation)                                                            \
  if (is_center_pixel(pixel)) {                                                                                                         \
    printf(                                                                                                                             \
      "@%u LD.CONTINUATION.%s 0x%08X [Weight:%f]\n", __lane_id, __continuation.is_light ? "LIGHT" : "NODE", __continuation.child_index, \
      1.0f / _light_tree_continuation_unpack_prob(__continuation));                                                                     \
  }
#define _LIGHT_TREE_DEBUG_STORE_LIGHT_TOKEN(__lane_id, __selected, __weight)      \
  if (is_center_pixel(pixel)) {                                                   \
    printf("@%u ST.LIGHT 0x%08X [Weight:%f]\n", __lane_id, __selected, __weight); \
  }
#define _LIGHT_TREE_DEBUG_JUMP_NODE_TOKEN(__lane_id, __selected, __weight)        \
  if (is_center_pixel(pixel)) {                                                   \
    printf("@%u JMP.NODE 0x%08X [Weight:%f]\n", __lane_id, __selected, __weight); \
  }
#else
#define _LIGHT_TREE_DEBUG_SELECT_CHILD_TOKEN(__lane_id, __selected, __target)
#define _LIGHT_TREE_DEBUG_STORE_CONTINUATION_TOKEN(__lane_id, __continuation)
#define _LIGHT_TREE_DEBUG_LOAD_CONTINUATION_TOKEN(__lane_id, __continuation)
#define _LIGHT_TREE_DEBUG_STORE_LIGHT_TOKEN(__lane_id, __selected, __weight)
#define _LIGHT_TREE_DEBUG_JUMP_NODE_TOKEN(__lane_id, __selected, __weight)
#endif /* LIGHT_TREE_DEBUG_TRAVERSAL */

template <MaterialType TYPE>
struct LightTreeContext {
  MaterialContext<TYPE> ctx;
};

template <>
struct LightTreeContext<MATERIAL_GEOMETRY> {
  vec3 position;
  vec3 normal;
  vec3 spec_dir_scaled;
  float spec_offset;
};

template <MaterialType TYPE>
LUMINARY_FUNCTION LightTreeContext<TYPE> light_tree_context_init(const MaterialContext<TYPE> ctx) {
  return {ctx};
}

template <>
LUMINARY_FUNCTION LightTreeContext<MATERIAL_GEOMETRY> light_tree_context_init(const MaterialContextGeometry ctx) {
  LightTreeContext<MATERIAL_GEOMETRY> tree_ctx;
  tree_ctx.position = ctx.position;

  if (MATERIAL_IS_SUBSTRATE_TRANSLUCENT(ctx.params.flags)) {
    tree_ctx.normal          = get_vector(0.0f, 0.0f, 0.0f);
    tree_ctx.spec_dir_scaled = get_vector(0.0f, 0.0f, 0.0f);
    tree_ctx.spec_offset     = 1.0f;
    return tree_ctx;
  }

  const float roughness = material_get_float<MATERIAL_GEOMETRY_PARAM_ROUGHNESS>(ctx.params);
  const float alpha     = roughness * roughness;
  const float NdotV     = __saturatef(dot_product(ctx.normal, ctx.V));

  // Off-specular reflection peak direction: blend of R = 2 * NdotV * N - V and N controlled purely by roughness
  const float k       = (1.0f - alpha) * (1.0f - alpha);
  const float n_scale = fmaf(k, fmaf(2.0f, NdotV, -1.0f), 1.0f);
  const vec3 R_raw    = sub_vector(scale_vector(ctx.normal, n_scale), scale_vector(ctx.V, k));

  float diff_weight = 0.05f;
  float spec_weight = 0.95f;
  if ((ctx.params.flags & MATERIAL_FLAG_METALLIC) == 0) {
    const float t     = 1.0f - NdotV;
    const float t2    = t * t;
    const float F     = fmaf(0.96f, t2 * t2 * t, 0.04f);
    const RGBF albedo = material_get_color<MATERIAL_GEOMETRY_PARAM_ALBEDO>(ctx.params);
    const float lum   = (1.0f / 3.0f) * (albedo.r + albedo.g + albedo.b);
    const float denom = fmaf(1.0f - F, fmaxf(lum, 0.05f), F);

    spec_weight = (F / denom) * (1.0f - alpha);
    diff_weight = 1.0f - spec_weight;
  }

  // Microfacet cone cosine threshold c0 depends only on roughness; pre-bake sqrt(spec_weight) and diff_weight into lobes
  const float c0         = 0.85f * k;
  const float spec_scale = sqrtf(spec_weight) / (1.0f - c0);
  const float r_scale    = __rsqrtf(fmaxf(dot_product(R_raw, R_raw), eps)) * spec_scale;

  tree_ctx.normal          = scale_vector(ctx.normal, diff_weight);
  tree_ctx.spec_dir_scaled = scale_vector(R_raw, r_scale);
  tree_ctx.spec_offset     = -c0 * spec_scale;

  return tree_ctx;
}

template <MaterialType TYPE>
LUMINARY_FUNCTION float light_tree_importance(const LightTreeContext<TYPE> ctx, const float power, const vec3 mean, const float std_dev);

template <>
LUMINARY_FUNCTION float light_tree_importance<MATERIAL_GEOMETRY>(
  const LightTreeContext<MATERIAL_GEOMETRY> ctx, const float power, const vec3 PO, const float std_dev) {
  const float dist_sq  = dot_product(PO, PO);
  const float variance = std_dev * std_dev;

  // Single MUFU.RSQ instruction; __rsqrtf prevents cicc from splitting rsqrt^2 into RCP + SQRT
  const float rsqrt_dist        = __rsqrtf(fmaxf(dist_sq + variance, eps));
  const float inv_dist_sq       = rsqrt_dist * rsqrt_dist;
  const float power_inv_dist_sq = power * inv_dist_sq;

  // Diffuse cosine lobe + microfacet specular cone lobe (weights pre-baked into ctx)
  const float NdotL     = __saturatef(dot_product(PO, ctx.normal) * rsqrt_dist);
  const float spec_cone = __saturatef(fmaf(dot_product(PO, ctx.spec_dir_scaled), rsqrt_dist, ctx.spec_offset));
  const float bsdf      = fmaf(spec_cone, spec_cone, NdotL);

  // Common-denominator variance blend: inv_dist_sq * ((1 - t) * bsdf + t) == inv_dist_sq^2 * (bsdf * dist_sq + variance)
  return (power_inv_dist_sq * inv_dist_sq) * fmaf(bsdf, dist_sq, variance);
}

template <>
LUMINARY_FUNCTION float light_tree_importance<MATERIAL_VOLUME>(
  const LightTreeContext<MATERIAL_VOLUME> tree_ctx, const float power, const vec3 PO, const float std_dev) {
  const MaterialContextVolume ctx = tree_ctx.ctx;

  // Distance to the mean projected onto the ray
  const float dist_along_ray = -dot_product(PO, ctx.V);

  // Clamp the distance if the ray is shortened by geometry.
  const float clamped_dist_along_ray = clampf(dist_along_ray, ctx.path.start, ctx.path_end);

  // Squared distance to the closest point on the ray segment via Pythagorean identity:
  // ||PO + clamped * V||^2 = (||PO||^2 - dist_along_ray^2) + (dist_along_ray - clamped)^2
  const float clamp_diff      = dist_along_ray - clamped_dist_along_ray;
  const float closest_dist_sq = clamp_diff * clamp_diff - dist_along_ray * dist_along_ray + dot_product(PO, PO);

  // Effective squared distance accounting for cluster spatial variance
  const float eff_dist_sq = fmaxf(std_dev * std_dev + closest_dist_sq, eps);

  // Effective active scattering segment length
  const float scatter_dist = (clamped_dist_along_ray - ctx.path.start) + std_dev;

  // Rational approximation of inscattering buildup and extinction attenuation along the ray
  const float atten = ctx.descriptor.max_scattering * scatter_dist + 1.0f;

  return (power * scatter_dist) / (eff_dist_sq * atten * atten);
}

template <>
LUMINARY_FUNCTION float light_tree_importance<MATERIAL_PARTICLE>(
  const LightTreeContext<MATERIAL_PARTICLE> tree_ctx, const float power, const vec3 PO, const float std_dev) {
  const float dist_sq = fmaxf(dot_product(PO, PO) + std_dev * std_dev, eps);

  return power / dist_sq;
}

template <MaterialType TYPE>
LUMINARY_FUNCTION float light_tree_child_importance(
  const LightTreeContext<TYPE> ctx, const DeviceLightTreeRootSection section, const vec3 rel_base, const vec3 exp, const float exp_v,
  const uint32_t i) {
  if (section.rel_power[i] == 0)
    return 0.0f;

  const float power   = (float) section.rel_power[i];
  const float std_dev = section.rel_std_dev[i] * exp_v;

  const vec3 rel_mean = get_vector(section.rel_mean_x[i], section.rel_mean_y[i], section.rel_mean_z[i]);
  const vec3 mean     = add_vector(mul_vector(rel_mean, exp), rel_base);

  return light_tree_importance<TYPE>(ctx, power, mean, std_dev);
}

template <MaterialType TYPE>
LUMINARY_FUNCTION float light_tree_child_importance(
  const LightTreeContext<TYPE> ctx, const DeviceLightTreeNode node, const vec3 rel_base, const vec3 exp, const float exp_v,
  const uint32_t i) {
  if (node.rel_power[i] == 0)
    return 0.0f;

  const float power   = (float) node.rel_power[i];
  const float std_dev = node.rel_std_dev[i] * exp_v;

  const vec3 rel_mean = get_vector(node.rel_mean_x[i], node.rel_mean_y[i], node.rel_mean_z[i]);
  const vec3 mean     = add_vector(mul_vector(rel_mean, exp), rel_base);

  return light_tree_importance<TYPE>(ctx, power, mean, std_dev);
}

LUMINARY_FUNCTION uint32_t light_tree_get_write_ptr(uint32_t& inplace_output_ptr, uint32_t& queue_write_ptr) {
  uint32_t output_ptr;
  if (inplace_output_ptr != 0xFFFFFFFF) {
    output_ptr         = inplace_output_ptr;
    inplace_output_ptr = 0xFFFFFFFF;
  }
  else {
    output_ptr = queue_write_ptr++;
  }

  return output_ptr;
}

LUMINARY_FUNCTION LightTreeContinuation
  _light_tree_continuation_pack(const uint32_t child_index, const float probability, const bool is_light) {
  LightTreeContinuation continuation;

  continuation.is_light    = is_light ? 1 : 0;
  continuation.child_index = child_index;
  continuation.probability = (probability > 0.0f) ? max((uint32_t) ((0xFFFFF * probability) + 0.5f), 1) : 0;

  return continuation;
}

LUMINARY_FUNCTION float _light_tree_continuation_unpack_prob(const LightTreeContinuation continuation) {
  return continuation.probability * (1.0f / 0xFFFFF) * LIGHT_TREE_NUM_OUTPUTS;
}

template <MaterialType TYPE>
LUMINARY_FUNCTION LightTreeWork light_tree_traverse_prepass(const LightTreeContext<TYPE> tree_ctx, const PathID& path_id) {
  const DeviceLightTreeRootHeader header = load_light_tree_root();

  RISAggregator ris_aggregator = ris_aggregator_init();

  RISLane ris_lane[LIGHT_TREE_NUM_OUTPUTS];
#pragma unroll
  for (uint32_t lane_id = 0; lane_id < LIGHT_TREE_NUM_OUTPUTS; lane_id++) {
    ris_lane[lane_id] = ris_lane_init(random_1D(MaterialContext<TYPE>::RANDOM_DL_GEO::TREE_PREPASS + lane_id, path_id));
  }

  const vec3 base   = get_vector(bfloat_unpack(header.x), bfloat_unpack(header.y), bfloat_unpack(header.z));
  const vec3 exp    = get_vector(exp2f(header.exp_x), exp2f(header.exp_y), exp2f(header.exp_z));
  const float exp_v = exp2f(header.exp_std_dev);

  vec3 position;
  if constexpr (TYPE == MATERIAL_GEOMETRY)
    position = tree_ctx.position;
  else
    position = tree_ctx.ctx.position;

  const vec3 rel_base = sub_vector(base, position);

  uint32_t selected[LIGHT_TREE_NUM_OUTPUTS] = {};

  float sum = 0.0f;

  LUMINARY_ASSUME(header.num_sections <= LIGHT_TREE_ROOT_MAX_NUM_SECTIONS);

#pragma nounroll
  for (uint32_t section_id = 0; section_id < header.num_sections; section_id++) {
    const DeviceLightTreeRootSection section = load_light_tree_root_section(section_id);

    static_assert(
      sizeof(section.rel_power[0]) == 2,
      "The root sum is normalized based on the rel_power because we don't do it in the importance function.");

#pragma unroll
    for (uint32_t rel_child_id = 0; rel_child_id < LIGHT_TREE_MAX_CHILDREN_PER_SECTION; rel_child_id++) {
      const float target         = light_tree_child_importance<TYPE>(tree_ctx, section, rel_base, exp, exp_v, rel_child_id);
      RISSampleHandle ris_sample = ris_aggregator_add_sample(ris_aggregator, target, 1.0f);

      if (ris_sample.resampling_probability <= 0.0f)
        continue;

      ris_sample_handle_finalize(ris_sample);

      if constexpr (TYPE == MATERIAL_GEOMETRY)
        sum += target;

      const uint32_t child_id = section_id * LIGHT_TREE_MAX_CHILDREN_PER_SECTION + rel_child_id;

#pragma unroll
      for (uint32_t lane_id = 0; lane_id < LIGHT_TREE_NUM_OUTPUTS; lane_id++) {
        if (ris_lane_add_sample(ris_lane[lane_id], ris_sample)) {
          selected[lane_id] = child_id;

          _LIGHT_TREE_DEBUG_SELECT_CHILD_TOKEN(lane_id, selected[lane_id], target);
        }
      }
    }
  }

  LightTreeWork work;
  if constexpr (TYPE == MATERIAL_GEOMETRY)
    work.root_sum = sum * (bfloat_unpack(header.power_normalization) / 0xFFFF);
  else
    work.root_sum = sum;

#pragma unroll
  for (uint32_t lane_id = 0; lane_id < LIGHT_TREE_NUM_OUTPUTS; lane_id++) {
    const bool is_light  = selected[lane_id] < header.num_root_lights;
    const uint32_t index = (is_light) ? selected[lane_id] : selected[lane_id] - header.num_root_lights;
    work.data[lane_id]   = _light_tree_continuation_pack(index, ris_lane_get_sampling_prob(ris_lane[lane_id], ris_aggregator), is_light);

    _LIGHT_TREE_DEBUG_STORE_CONTINUATION_TOKEN(lane_id, work.data[lane_id]);
  }

  return work;
}

template <MaterialType TYPE>
LUMINARY_FUNCTION LightTreeWork light_tree_traverse_prepass(const MaterialContext<TYPE> ctx, const PathID& path_id) {
  return light_tree_traverse_prepass<TYPE>(light_tree_context_init<TYPE>(ctx), path_id);
}

template <MaterialType TYPE>
LUMINARY_FUNCTION LightTreeResult light_tree_traverse_postpass(
  const LightTreeContext<TYPE> tree_ctx, const PathID& path_id, const uint32_t lane_id, const LightTreeWork& work) {
  const LightTreeContinuation continuation = work.data[lane_id];

  _LIGHT_TREE_DEBUG_LOAD_CONTINUATION_TOKEN(lane_id, continuation);

  const float continuation_probability = _light_tree_continuation_unpack_prob(continuation);

  LightTreeResult result;
  result.light_id = 0xFFFFFFFF;
  result.weight   = (continuation_probability > 0.0f) ? 1.0f / continuation_probability : 0.0f;

  if (continuation_probability == 0.0f)
    return result;

  if (continuation.is_light) {
    result.light_id = continuation.child_index;
    return result;
  }

  DeviceLightTreeNode node = load_light_tree_node(continuation.child_index);

  const float random     = random_1D(MaterialContext<TYPE>::RANDOM_DL_GEO::TREE_POSTPASS + lane_id, path_id);
  RISReservoir reservoir = ris_reservoir_init(random);

  vec3 position;
  if constexpr (TYPE == MATERIAL_GEOMETRY)
    position = tree_ctx.position;
  else
    position = tree_ctx.ctx.position;

#pragma nounroll
  while (true) {
    const vec3 base   = get_vector(bfloat_unpack(node.x), bfloat_unpack(node.y), bfloat_unpack(node.z));
    const vec3 exp    = get_vector(exp2f(node.exp_x), exp2f(node.exp_y), exp2f(node.exp_z));
    const float exp_v = exp2f(node.exp_std_dev);

    const vec3 rel_base = sub_vector(base, position);

    uint32_t selected_child = 0xFF;

#pragma unroll
    for (uint32_t rel_child_id = 0; rel_child_id < LIGHT_TREE_CHILDREN_PER_NODE; rel_child_id++) {
      const float target = light_tree_child_importance<TYPE>(tree_ctx, node, rel_base, exp, exp_v, rel_child_id);

      if (ris_reservoir_add_sample(reservoir, target, 1.0f)) {
        selected_child = rel_child_id;
      }
    }

    if (selected_child == 0xFF)
      break;

    result.weight *= ris_reservoir_get_sampling_weight(reservoir);

    if (selected_child < node.num_lights) {
      result.light_id = node.light_ptr + selected_child;
      _LIGHT_TREE_DEBUG_STORE_LIGHT_TOKEN(lane_id, result.light_id, result.weight);
      break;
    }

    _LIGHT_TREE_DEBUG_JUMP_NODE_TOKEN(lane_id, node.child_ptr + (selected_child - node.num_lights), result.weight);

    node = load_light_tree_node(node.child_ptr + (selected_child - node.num_lights));

    ris_reservoir_reset(reservoir);
  }

  return result;
}

template <MaterialType TYPE>
LUMINARY_FUNCTION LightTreeResult
  light_tree_traverse_postpass(const MaterialContext<TYPE> ctx, const PathID& path_id, const uint32_t lane_id, const LightTreeWork& work) {
  return light_tree_traverse_postpass<TYPE>(light_tree_context_init<TYPE>(ctx), path_id, lane_id, work);
}

LUMINARY_FUNCTION TriangleHandle light_tree_get_light(const uint32_t id, DeviceTransform& transform) {
  const TriangleHandle handle = device.ptrs.light_tree_tri_handle_map[id];

  transform = load_transform(handle.instance_id);

  return handle;
}

#endif /* CU_LUMINARY_LIGHT_TREE_H */
