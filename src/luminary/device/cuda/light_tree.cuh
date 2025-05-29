#ifndef CU_LUMINARY_LIGHT_TREE_H
#define CU_LUMINARY_LIGHT_TREE_H

#include "light_sg.cuh"
#include "math.cuh"
#include "memory.cuh"
#include "ris.cuh"
#include "splitting.cuh"
#include "utils.cuh"

#define LIGHT_TREE_NUM_OUTPUTS 4
#define LIGHT_TREE_INVALID_NODE 0xFFFFFF
#define LIGHT_TREE_SELECTED_IS_LIGHT 0x80000000
#define LIGHT_TREE_SELECTED_PTR_MASK LIGHT_TREE_INVALID_NODE
// #define LIGHT_TREE_DEBUG_TRAVERSAL

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

template <MaterialLayerType TYPE>
__device__ float light_tree_importance(const GBufferData data, const float power, const vec3 mean, const float radius);

template <>
__device__ float light_tree_importance<MATERIAL_LAYER_TYPE_DIFFUSE>(
  const GBufferData data, const float power, const vec3 mean, const float radius) {
  const vec3 PO = sub_vector(mean, data.position);

#if 0
  const vec3 D      = normalize_vector(cross_product(cross_product(PO, data.normal), PO));
  const vec3 v      = normalize_vector(PO);
  const float theta = fminf(asinf(radius / get_length(PO)), acosf(dot_product(data.normal, v)));
  const vec3 L = normalize_vector(add_vector(scale_vector(v, cosf(theta)), scale_vector(D, sinf(theta))));
#else
  const vec3 L = normalize_vector(sub_vector(add_vector(mean, scale_vector(data.normal, radius)), data.position));
#endif

  const float dist_sq = fmaxf(dot_product(PO, PO), radius * radius);

  return power * __saturatef(dot_product(L, data.normal)) / dist_sq;
}

template <>
__device__ float light_tree_importance<MATERIAL_LAYER_TYPE_MICROFACET_REFLECTION>(
  const GBufferData data, const float power, const vec3 mean, const float radius) {
  const vec3 PO = sub_vector(mean, data.position);

  const vec3 R          = reflect_vector(data.V, data.normal);
  const vec3 target_dir = normalize_vector(add_vector(scale_vector(data.normal, data.roughness), scale_vector(R, 1.0f - data.roughness)));
  const vec3 v          = normalize_vector(PO);
  const float length_PO = get_length(PO);
  const float R_dot_V   = dot_product(target_dir, v);

  float N_dot_H;
  if (R_dot_V > 1.0f - eps || length_PO < radius) {
    N_dot_H = 1.0f;
  }
  else {
    const vec3 D      = normalize_vector(cross_product(cross_product(PO, target_dir), PO));
    const float theta = fminf(asinf(radius / length_PO), acosf(R_dot_V));
    const vec3 u      = normalize_vector(add_vector(scale_vector(v, cosf(theta)), scale_vector(D, sinf(theta))));
    const vec3 H      = normalize_vector(add_vector(data.V, u));

    N_dot_H = dot_product(data.normal, H);
  }

  const float roughness2 = data.roughness * data.roughness;
  // TODO: Why is roughness2 giving correct values? This seems like there is something wrong in the evaluation but I can't find anything.
  const float D = bsdf_microfacet_evaluate_D_GGX(N_dot_H, roughness2);

  const float dist_sq    = fmaxf(dot_product(PO, PO), radius * radius);
  const float power_term = lerp(1.0f, power / dist_sq, data.roughness);

  return power_term * fmaxf(D, 0.0f);
}

template <>
__device__ float light_tree_importance<MATERIAL_LAYER_TYPE_MICROFACET_REFRACTION>(
  const GBufferData data, const float power, const vec3 mean, const float radius) {
  return 0.0f;
}

template <MaterialLayerType TYPE>
__device__ float light_tree_child_importance(
  const GBufferData data, const DeviceLightTreeRootSection section, const vec3 base, const vec3 exp, const float exp_v, const uint32_t i) {
  if (section.rel_power[i] == 0)
    return 0.0f;

  const float power    = (float) section.rel_power[i];
  const float variance = section.rel_variance[i] * exp_v;

  const vec3 mean = add_vector(mul_vector(get_vector(section.rel_mean_x[i], section.rel_mean_y[i], section.rel_mean_z[i]), exp), base);

  return fmaxf(light_tree_importance<TYPE>(data, power, mean, variance), 0.0f);
}

template <MaterialLayerType TYPE>
__device__ float light_tree_child_importance(
  const GBufferData data, const DeviceLightTreeNode node, const vec3 base, const vec3 exp, const float exp_v, const uint32_t i) {
  if (node.rel_power[i] == 0)
    return 0.0f;

  const float power    = (float) node.rel_power[i];
  const float variance = node.rel_variance[i] * exp_v;

  const vec3 mean = add_vector(mul_vector(get_vector(node.rel_mean_x[i], node.rel_mean_y[i], node.rel_mean_z[i]), exp), base);

  return fmaxf(light_tree_importance<TYPE>(data, power, mean, variance), 0.0f);
}

__device__ uint32_t light_tree_get_write_ptr(uint32_t& inplace_output_ptr, uint32_t& queue_write_ptr) {
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

__device__ LightTreeContinuation _light_tree_continuation_pack(const uint8_t child_index, const float probability, const bool is_light) {
  LightTreeContinuation continuation;

  continuation.is_light    = is_light ? 1 : 0;
  continuation.child_index = child_index;
  continuation.probability = (uint32_t) ((0xFFFFF * probability) + 0.5f);

  return continuation;
}

__device__ float _light_tree_continuation_unpack_prob(const LightTreeContinuation continuation) {
  return continuation.probability * (1.0f / 0xFFFFF) * LIGHT_TREE_NUM_OUTPUTS;
}

template <MaterialLayerType TYPE>
__device__ LightTreeWork light_tree_traverse_prepass(const GBufferData data, const ushort2 pixel) {
  const DeviceLightTreeRootHeader header = load_light_tree_root();

  RISAggregator ris_aggregator = ris_aggregator_init();

  RISLane ris_lane[LIGHT_TREE_NUM_OUTPUTS];
#pragma unroll
  for (uint32_t lane_id = 0; lane_id < LIGHT_TREE_NUM_OUTPUTS; lane_id++) {
    ris_lane[lane_id] = ris_lane_init(quasirandom_sequence_1D(QUASI_RANDOM_TARGET_RIS_LIGHT_TREE + lane_id, pixel));
  }

  uint8_t selected[LIGHT_TREE_NUM_OUTPUTS];
#pragma unroll
  for (uint32_t lane_id = 0; lane_id < LIGHT_TREE_NUM_OUTPUTS; lane_id++) {
    selected[lane_id] = 0xFF;
  }

  const vec3 base   = get_vector(bfloat_unpack(header.x), bfloat_unpack(header.y), bfloat_unpack(header.z));
  const vec3 exp    = get_vector(exp2f(header.exp_x), exp2f(header.exp_y), exp2f(header.exp_z));
  const float exp_v = exp2f(header.exp_variance);

  for (uint32_t section_id = 0; section_id < header.num_sections; section_id++) {
    const DeviceLightTreeRootSection section = load_light_tree_root_section(section_id);

    for (uint32_t rel_child_id = 0; rel_child_id < LIGHT_TREE_MAX_CHILDREN_PER_SECTION; rel_child_id++) {
      const float target               = light_tree_child_importance<TYPE>(data, section, base, exp, exp_v, rel_child_id);
      const RISSampleHandle ris_sample = ris_aggregator_add_sample(ris_aggregator, target, 1.0f);

      for (uint32_t lane_id = 0; lane_id < LIGHT_TREE_NUM_OUTPUTS; lane_id++) {
        if (ris_lane_add_sample(ris_lane[lane_id], ris_sample)) {
          selected[lane_id] = section_id * LIGHT_TREE_MAX_CHILDREN_PER_SECTION + rel_child_id;

          _LIGHT_TREE_DEBUG_SELECT_CHILD_TOKEN(lane_id, selected[lane_id], target);
        }
      }
    }
  }

  LightTreeWork work;

#pragma unroll
  for (uint32_t lane_id = 0; lane_id < LIGHT_TREE_NUM_OUTPUTS; lane_id++) {
    const bool is_light  = selected[lane_id] < header.num_root_lights;
    const uint32_t index = is_light ? selected[lane_id] : selected[lane_id] - header.num_root_lights;
    work.data[lane_id]   = _light_tree_continuation_pack(index, ris_lane_get_sampling_prob(ris_lane[lane_id], ris_aggregator), is_light);

    _LIGHT_TREE_DEBUG_STORE_CONTINUATION_TOKEN(lane_id, work.data[lane_id]);
  }

  return work;
}

template <MaterialLayerType TYPE>
__device__ LightTreeResult
  light_tree_traverse_postpass(const GBufferData data, const ushort2 pixel, const uint32_t index, const LightTreeWork work) {
  const LightTreeContinuation continuation = work.data[index];

  _LIGHT_TREE_DEBUG_LOAD_CONTINUATION_TOKEN(index, continuation);

  LightTreeResult result;
  result.light_id = 0xFFFFFFFF;
  result.weight   = 1.0f / _light_tree_continuation_unpack_prob(continuation);

  if (continuation.child_index == 0xFF) {
    return result;
  }

  if (continuation.is_light) {
    result.light_id = continuation.child_index;
    return result;
  }

  DeviceLightTreeNode node = load_light_tree_node(continuation.child_index);

  const float random     = quasirandom_sequence_1D(QUASI_RANDOM_TARGET_RIS_LIGHT_TREE + index + LIGHT_TREE_NUM_OUTPUTS, pixel);
  RISReservoir reservoir = ris_reservoir_init(random);

  while (result.light_id == 0xFFFFFFFF) {
    const vec3 base   = get_vector(bfloat_unpack(node.x), bfloat_unpack(node.y), bfloat_unpack(node.z));
    const vec3 exp    = get_vector(exp2f(node.exp_x), exp2f(node.exp_y), exp2f(node.exp_z));
    const float exp_v = exp2f(node.exp_variance);

    uint8_t selected_child = 0xFF;

    for (uint32_t rel_child_id = 0; rel_child_id < LIGHT_TREE_CHILDREN_PER_NODE; rel_child_id++) {
      const float target = light_tree_child_importance<TYPE>(data, node, base, exp, exp_v, rel_child_id);

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
      _LIGHT_TREE_DEBUG_STORE_LIGHT_TOKEN(index, result.light_id, result.weight);
      break;
    }

    _LIGHT_TREE_DEBUG_JUMP_NODE_TOKEN(index, node.child_ptr + (selected_child - node.num_lights), result.weight);

    node = load_light_tree_node(node.child_ptr + (selected_child - node.num_lights));

    ris_reservoir_reset(reservoir);
  }

  return result;
}

__device__ TriangleHandle light_tree_get_light(const uint32_t id, DeviceTransform& transform) {
  const TriangleHandle handle = device.ptrs.light_tree_tri_handle_map[id];

  transform = load_transform(handle.instance_id);

  return handle;
}

#endif /* CU_LUMINARY_LIGHT_TREE_H */
