#ifndef CU_LUMINARY_LIGHT_TREE_H
#define CU_LUMINARY_LIGHT_TREE_H

#include "light_linked_list.cuh"
#include "light_sg.cuh"
#include "math.cuh"
#include "memory.cuh"
#include "ris.cuh"
#include "utils.cuh"

typedef uint16_t PackedProb;
typedef uint16_t BFloat16;

__device__ float light_tree_unpack_probability(const PackedProb p) {
  const uint32_t data = p;

  // 5 bits exponent to reach an exponent range of -31 to 0
  // 11 bits mantissa
  return __uint_as_float(0x30000000 | (data << 12));
}

__device__ PackedProb light_tree_pack_probability(const float p) {
  // Add this term to round to nearest
  const uint32_t data = __float_as_uint(p) + (1u << 11);

  // The 2 bits in the exponent will automatically be truncated
  return (PackedProb) (data >> 12);
}

__device__ float light_tree_bfloat_to_float(const BFloat16 val) {
  const uint32_t data = val;

  return __uint_as_float(data << 16);
}

__device__ BFloat16 light_tree_float_to_bfloat(const float val) {
  // Add this term to round to nearest
  const uint32_t data = __float_as_uint(val) + (1u << 15);

  return (BFloat16) (data >> 16);
}

__device__ void light_tree_child_importance(
  const LightSGData data, const vec3 position, const vec3 normal, const DeviceLightTreeNode node, const vec3 exp, const float exp_v,
  float importance[8], const uint32_t i) {
  const bool lower_data = (i < 4);
  const uint32_t shift  = (lower_data ? i : (i - 4)) << 3;

  const uint32_t rel_power = lower_data ? node.rel_power[0] : node.rel_power[1];

  float power = (float) ((rel_power >> shift) & 0xFF);

  // This means this is a NULL node.
  if (power == 0.0f) {
    importance[i] = 0.0f;
    return;
  }

  const uint32_t rel_variance = lower_data ? node.rel_variance_leaf[0] : node.rel_variance_leaf[1];

  float variance;
  variance = (rel_variance >> shift) & 0xFF;
  variance = variance * exp_v;

  const uint32_t rel_mean_x = lower_data ? node.rel_mean_x[0] : node.rel_mean_x[1];
  const uint32_t rel_mean_y = lower_data ? node.rel_mean_y[0] : node.rel_mean_y[1];
  const uint32_t rel_mean_z = lower_data ? node.rel_mean_z[0] : node.rel_mean_z[1];

  vec3 mean;
  mean = get_vector((rel_mean_x >> shift) & 0xFF, (rel_mean_y >> shift) & 0xFF, (rel_mean_z >> shift) & 0xFF);
  mean = add_vector(mul_vector(mean, exp), node.base_mean);

  importance[i] = fmaxf(light_sg_evaluate(data, position, normal, mean, variance, power), 0.0f);
}

__device__ void light_tree_traverse(
  const LightSGData data, const vec3 position, const vec3 normal, const ushort2 pixel, float random,
  LightLinkedListReference stack[LIGHT_LINKED_LIST_MAX_REFERENCES], uint32_t& stack_ptr) {
  random = random_saturate(random);

  float importance[8];

  DeviceLightTreeNode node = load_light_tree_node(0);
  float probability        = 1.0f;

  while (node.child_ptr != 0xFFFFFFFF) {
    // Only push if this node references a valid light list
    if (node.light_ptr != 0xFFFFFFFF) {
      LightLinkedListReference ref;
      ref.id          = node.light_ptr;
      ref.probability = probability;

      stack[stack_ptr++] = ref;
    }

    const vec3 exp    = get_vector(exp2f(node.exp_x), exp2f(node.exp_y), exp2f(node.exp_z));
    const float exp_v = exp2f(node.exp_variance);

    float sum_importance = 0.0f;

#pragma unroll
    for (uint32_t i = 0; i < 8; i++) {
      light_tree_child_importance(data, position, normal, node, exp, exp_v, importance, i);
      sum_importance += importance[i];
    }

    float accumulated_importance = 0.0f;

    uint32_t selected_child   = 0xFFFFFFFF;
    float selected_importance = 0.0f;
    float random_shift        = 0.0f;

    const float importance_target = random * sum_importance;

#pragma unroll
    for (uint32_t i = 0; i < 8; i++) {
      const float child_importance = importance[i];

      accumulated_importance += child_importance;

      // Select node, never select nodes with 0 importance (this is also important for NULL nodes)
      if (child_importance > 0.0f && accumulated_importance >= importance_target) {
        selected_child      = i;
        selected_importance = child_importance;

        random_shift = accumulated_importance - child_importance;

        // No control flow, we always loop over all children.
        accumulated_importance = -FLT_MAX;
      }
    }

    // This can only happen if all children were leaves
    if (selected_child == 0xFFFFFFFF) {
      node.child_ptr = 0xFFFFFFFF;
      break;
    }

    const float selection_probability = selected_importance / sum_importance;

    // Rescale random number
    random = (sum_importance > 0.0f) ? random_saturate((random * sum_importance - random_shift) / selected_importance) : random;

    probability *= selection_probability;

    node = load_light_tree_node(node.child_ptr + selected_child);
  }
}

__device__ uint32_t light_tree_query(
  const GBufferData data, const float random, const ushort2 pixel, LightLinkedListReference stack[LIGHT_LINKED_LIST_MAX_REFERENCES]) {
  const LightSGData sg_data = light_sg_prepare(data);

  uint32_t num_lists = 0;
  light_tree_traverse(sg_data, data.position, data.normal, pixel, random, stack, num_lists);

  return num_lists;
}

__device__ TriangleHandle light_tree_get_light(const uint32_t id, DeviceTransform& transform) {
  const TriangleHandle handle = device.ptrs.light_tree_tri_handle_map[id];

  transform = load_transform(handle.instance_id);

  return handle;
}

#endif /* CU_LUMINARY_LIGHT_TREE_H */
