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
  // 3 bits spare
  uint32_t is_light : 1, child_index : 8, probability : 20;
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
LUMINARY_FUNCTION float light_tree_importance(const MaterialContext<TYPE> ctx, const float power, const vec3 mean, const float std_dev);

template <>
LUMINARY_FUNCTION float light_tree_importance<MATERIAL_GEOMETRY>(
  const MaterialContextGeometry ctx, const float power, const vec3 mean, const float std_dev) {
  const vec3 PO       = sub_vector(mean, ctx.position);
  const float dist_sq = dot_product(PO, PO);

  const float variance    = std_dev * std_dev;
  const float inv_dist_sq = 1.0f / (dist_sq + variance);

  float result = power * inv_dist_sq;
  if (MATERIAL_IS_SUBSTRATE_TRANSLUCENT(ctx.params.flags))
    return result;

  const float t     = variance * inv_dist_sq;
  const float NdotL = __saturatef(dot_product(PO, ctx.normal) * sqrtf(inv_dist_sq));
  result            = result * fmaf(NdotL, 1.0f - t, t);

  return result;
}

template <>
LUMINARY_FUNCTION float light_tree_importance<MATERIAL_VOLUME>(
  const MaterialContextVolume ctx, const float power, const vec3 mean, const float std_dev) {
  const vec3 PO = sub_vector(mean, ctx.position);

  // Distance to the mean projected onto the ray
  const float dist_along_ray = -dot_product(PO, ctx.V);

  // Clamp the distance if the ray is shortened by geometry.
  const float clamped_dist_along_ray = clampf(dist_along_ray, 0.0f, ctx.max_dist);

  // This is only perpendicular if we didn't actually clamp
  const vec3 perpendicular_vector   = sub_vector(PO, scale_vector(ctx.V, clamped_dist_along_ray));
  const float perpendicular_dist_sq = dot_product(perpendicular_vector, perpendicular_vector);

  // Account for quadratic falloff
  const float falloff = 1.0f / (perpendicular_dist_sq + std_dev);

  const float variance = std_dev * std_dev;

  // Don't use direct path to save on computations by using distances that we already computed.
  const float transmittance_depth = fmaxf(perpendicular_dist_sq + clamped_dist_along_ray * clamped_dist_along_ray - variance, 0.0f);

  // Account for energy loss due to absorption
  const float transmittance = expf(-ctx.descriptor.max_absorption * transmittance_depth);

  // Account for limited amount of energy that is scattered along the ray before passing the mean
  const float scattering = 1.0f - expf(-ctx.descriptor.max_scattering * (variance + clamped_dist_along_ray));

  return power * falloff * transmittance * scattering;
}

template <>
LUMINARY_FUNCTION float light_tree_importance<MATERIAL_PARTICLE>(
  const MaterialContextParticle ctx, const float power, const vec3 mean, const float std_dev) {
  const vec3 PO       = sub_vector(mean, ctx.position);
  const float dist_sq = dot_product(PO, PO) + std_dev * std_dev;

  return power / dist_sq;
}

template <MaterialType TYPE>
LUMINARY_FUNCTION float light_tree_child_importance(
  const MaterialContext<TYPE> ctx, const DeviceLightTreeRootSection section, const vec3 base, const vec3 exp, const float exp_v,
  const uint32_t i) {
  if (section.rel_power[i] == 0)
    return 0.0f;

  const float power   = (float) section.rel_power[i];
  const float std_dev = section.rel_std_dev[i] * exp_v;

  const vec3 rel_mean = get_vector(section.rel_mean_x[i], section.rel_mean_y[i], section.rel_mean_z[i]);
  const vec3 mean     = add_vector(mul_vector(rel_mean, exp), base);

  return fmaxf(light_tree_importance<TYPE>(ctx, power, mean, std_dev), 0.0f);
}

template <MaterialType TYPE>
LUMINARY_FUNCTION float light_tree_child_importance(
  const MaterialContext<TYPE> ctx, const DeviceLightTreeNode node, const vec3 base, const vec3 exp, const float exp_v, const uint32_t i) {
  if (node.rel_power[i] == 0)
    return 0.0f;

  const float power   = (float) node.rel_power[i];
  const float std_dev = node.rel_std_dev[i] * exp_v;

  const vec3 rel_mean = get_vector(node.rel_mean_x[i], node.rel_mean_y[i], node.rel_mean_z[i]);
  const vec3 mean     = add_vector(mul_vector(rel_mean, exp), base);

  return fmaxf(light_tree_importance<TYPE>(ctx, power, mean, std_dev), 0.0f);
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
  _light_tree_continuation_pack(const uint8_t child_index, const float probability, const bool is_light) {
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
LUMINARY_FUNCTION LightTreeWork light_tree_traverse_prepass(const MaterialContext<TYPE> ctx, const PathID& path_id) {
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

  uint8_t selected[LIGHT_TREE_NUM_OUTPUTS];
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
      const float target               = light_tree_child_importance<TYPE>(ctx, section, base, exp, exp_v, rel_child_id);
      const RISSampleHandle ris_sample = ris_aggregator_add_sample(ris_aggregator, target, 1.0f);

      if (ris_sample.resampling_probability == 0.0f)
        continue;

      if constexpr (TYPE == MATERIAL_GEOMETRY)
        sum += target;

#pragma unroll
      for (uint32_t lane_id = 0; lane_id < LIGHT_TREE_NUM_OUTPUTS; lane_id++) {
        if (ris_lane_add_sample(ris_lane[lane_id], ris_sample)) {
          selected[lane_id] = section_id * LIGHT_TREE_MAX_CHILDREN_PER_SECTION + rel_child_id;

          _LIGHT_TREE_DEBUG_SELECT_CHILD_TOKEN(lane_id, selected[lane_id], target);
        }
      }
    }
  }

  LightTreeWork work;
  work.root_sum = sum * (bfloat_unpack(header.power_normalization) / 0xFFFF);

#pragma unroll
  for (uint32_t lane_id = 0; lane_id < LIGHT_TREE_NUM_OUTPUTS; lane_id++) {
    const bool is_light = selected[lane_id] < header.num_root_lights;
    const uint8_t index = (is_light) ? selected[lane_id] : selected[lane_id] - header.num_root_lights;
    work.data[lane_id]  = _light_tree_continuation_pack(index, ris_lane_get_sampling_prob(ris_lane[lane_id], ris_aggregator), is_light);

    _LIGHT_TREE_DEBUG_STORE_CONTINUATION_TOKEN(lane_id, work.data[lane_id]);
  }

  return work;
}

template <MaterialType TYPE>
LUMINARY_FUNCTION LightTreeResult
  light_tree_traverse_postpass(const MaterialContext<TYPE> ctx, const PathID& path_id, const uint32_t lane_id, const LightTreeWork& work) {
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

#pragma nounroll
  while (result.light_id == 0xFFFFFFFF) {
    const vec3 base   = get_vector(bfloat_unpack(node.x), bfloat_unpack(node.y), bfloat_unpack(node.z));
    const vec3 exp    = get_vector(exp2f(node.exp_x), exp2f(node.exp_y), exp2f(node.exp_z));
    const float exp_v = exp2f(node.exp_std_dev);

    uint8_t selected_child = 0xFF;

#pragma unroll
    for (uint32_t rel_child_id = 0; rel_child_id < LIGHT_TREE_CHILDREN_PER_NODE; rel_child_id++) {
      const float target = light_tree_child_importance<TYPE>(ctx, node, base, exp, exp_v, rel_child_id);

      if (ris_reservoir_add_sample(reservoir, target, 1.0f)) {
        selected_child = rel_child_id;
      }
    }

    if (selected_child == 0xFF) {
      break;
    }

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

LUMINARY_FUNCTION TriangleHandle light_tree_get_light(const uint32_t id, DeviceTransform& transform) {
  const TriangleHandle handle = device.ptrs.light_tree_tri_handle_map[id];

  transform = load_transform(handle.instance_id);

  return handle;
}

#endif /* CU_LUMINARY_LIGHT_TREE_H */
