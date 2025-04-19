#ifndef CU_LUMINARY_LIGHT_TREE_H
#define CU_LUMINARY_LIGHT_TREE_H

#include "light_sg.cuh"
#include "math.cuh"
#include "memory.cuh"
#include "ris.cuh"
#include "utils.cuh"

#define LIGHT_TREE_STACK_SIZE 64

#define LIGHT_TREE_STACK_POP(__macro_internal_stack, __macro_internal_ptr, __macro_internal_entry) \
  (__macro_internal_entry) = (__macro_internal_stack)[--(__macro_internal_ptr)]

#define LIGHT_TREE_STACK_PUSH(__macro_internal_stack, __macro_internal_ptr, __macro_internal_entry) \
  (__macro_internal_stack)[(__macro_internal_ptr)++] = (__macro_internal_entry)

typedef uint16_t PackedProb;
typedef uint16_t BFloat16;

struct LightTreeStackEntry {
  uint32_t id;
  PackedProb T;
  PackedProb parent_split;
} typedef LightTreeStackEntry;
LUM_STATIC_SIZE_ASSERT(LightTreeStackEntry, 0x08);

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
  float importance[8], float splitting_prob[8], const uint32_t i, const uint32_t child_light_id, bool& is_leaf) {
  const bool lower_data = (i < 4);
  const uint32_t shift  = (lower_data ? i : (i - 4)) << 3;

  const uint32_t rel_power = lower_data ? node.rel_power[0] : node.rel_power[1];

  float power = (float) ((rel_power >> shift) & 0xFF);

  // This means this is a NULL node.
  if (power == 0.0f) {
    importance[i]     = 0.0f;
    splitting_prob[i] = 0.0f;
    return;
  }
  const uint32_t rel_variance_leaf = lower_data ? node.rel_variance_leaf[0] : node.rel_variance_leaf[1];

  is_leaf = ((rel_variance_leaf >> shift) & 0x1) != 0;

  DeviceLightTreeLeaf leaf_data;
  if (is_leaf) {
    // TODO: Single load instruction
    leaf_data = device.ptrs.light_tree_leaves[node.light_ptr + child_light_id];
  }

  float variance;
  variance = (rel_variance_leaf >> (shift + 1)) & 0x7F;
  variance = variance * exp_v;

  const uint32_t rel_mean_x = lower_data ? node.rel_mean_x[0] : node.rel_mean_x[1];
  const uint32_t rel_mean_y = lower_data ? node.rel_mean_y[0] : node.rel_mean_y[1];
  const uint32_t rel_mean_z = lower_data ? node.rel_mean_z[0] : node.rel_mean_z[1];

  vec3 mean;
  mean = get_vector((rel_mean_x >> shift) & 0xFF, (rel_mean_y >> shift) & 0xFF, (rel_mean_z >> shift) & 0xFF);
  mean = add_vector(mul_vector(mean, exp), node.base_mean);

  vec3 leaf_normal = get_vector(0.0f, 0.0f, 0.0f);
  if (is_leaf) {
    leaf_normal = normal_unpack(leaf_data.packed_normal);
    power       = leaf_data.power;
  }

  float uncertainty;
  const float approx_importance = light_sg_evaluate(data, position, normal, mean, variance, power, leaf_normal, is_leaf, uncertainty);

  // Uncertainty is in the range [0,1] and also acts as the probability for splitting.
  // For non leaf nodes we reduce the importance based on the uncertainty. We cannot do that for
  // leaves because we will resample later based on the importance.
  importance[i]     = fmaxf((is_leaf) ? approx_importance : approx_importance * (1.0f - uncertainty), 0.0f);
  splitting_prob[i] = uncertainty;
}

__device__ void light_tree_traverse(
  const LightSGData data, const vec3 position, const vec3 normal, const ushort2 pixel, float2 random,
  LightTreeStackEntry stack[LIGHT_TREE_STACK_SIZE], uint32_t& stack_ptr, RISReservoir& reservoir, uint32_t& light_id) {
  random.x = random_saturate(random.x);

  float importance[8];
  float split_probability[8];

  while (stack_ptr > 0) {
    LightTreeStackEntry entry;
    LIGHT_TREE_STACK_POP(stack, stack_ptr, entry);

    DeviceLightTreeNode node = load_light_tree_node(entry.id);
    float T                  = light_tree_unpack_probability(entry.T);
    float parent_split       = light_tree_unpack_probability(entry.parent_split);

    while (node.child_ptr != 0xFFFFFFFF) {
      const vec3 exp    = get_vector(exp2f(node.exp_x), exp2f(node.exp_y), exp2f(node.exp_z));
      const float exp_v = exp2f(node.exp_variance);

      float sum_importance  = 0.0f;
      uint32_t child_lights = 0;

#pragma unroll
      for (uint32_t i = 0; i < 8; i++) {
        bool is_leaf;
        light_tree_child_importance(data, position, normal, node, exp, exp_v, importance, split_probability, i, child_lights, is_leaf);

        child_lights += (is_leaf) ? 1 : 0;

        // Leaves are always send to the reservoir, we don't select them
        if (is_leaf == false) {
          sum_importance += importance[i];
          split_probability[i] *= parent_split;
        }
      }

      float accumulated_importance = 0.0f;

      uint32_t selected_child   = 0xFFFFFFFF;
      float selected_importance = 0.0f;
      float selected_split_prob = 0.0f;
      float random_shift        = 0.0f;

      const float importance_target = random.x * sum_importance;

      child_lights = 0;

#pragma unroll
      for (uint32_t i = 0; i < 8; i++) {
        const float child_importance = importance[i];

        const bool lower_data             = (i < 4);
        const uint32_t variance_leaf_data = lower_data ? node.rel_variance_leaf[0] : node.rel_variance_leaf[1];
        const uint32_t shift              = (lower_data ? i : (i - 4)) << 3;

        const bool is_leaf = ((variance_leaf_data >> shift) & 0x1) != 0;

        // if (is_center_pixel(pixel) && IS_PRIMARY_RAY) {
        //   printf("CHILD[%u]: %f %f %u\n", i, child_importance, split_probability[i], is_leaf ? 1 : 0);
        // }

        if (is_leaf) {
          const float leaf_probability = T * (1.0f - parent_split) + parent_split;

          // if (is_center_pixel(pixel) && IS_PRIMARY_RAY) {
          // printf("RESERVOIR: %u %f %f\n", node.light_ptr + current_child_light_offset, child_importance, leaf_probability);
          //}

          if (ris_reservoir_add_sample(reservoir, child_importance, 1.0f / leaf_probability)) {
            light_id = node.light_ptr + child_lights;
          }
        }
        else {
          accumulated_importance += child_importance;

          const float split_prob = split_probability[i];

          // Select node, never select nodes with 0 importance (this is also important for NULL nodes)
          if (child_importance > 0.0f && accumulated_importance >= importance_target) {
            selected_child      = i;
            selected_importance = child_importance;
            selected_split_prob = split_prob;

            random_shift = accumulated_importance - child_importance;

            // if (is_center_pixel(pixel) && IS_PRIMARY_RAY) {
            // printf("SELECTED: %u %f %f\n", i, child_importance, sum_importance);
            //}

            // No control flow, we always loop over all children.
            accumulated_importance = -FLT_MAX;
          }
          // Split node
          else if (split_prob > random.y) {
            // if (is_center_pixel(pixel) && IS_PRIMARY_RAY) {
            // printf("SPLIT: %u %f\n", node.child_ptr + i, split_prob);
            //}

            const float prob_parent_split_given_no_split_child = (parent_split - split_prob) / fmaxf(1.0f - split_prob, eps);
            const float selection_prob_child                   = fabsf(importance[i]) / sum_importance;

            const float T_child =
              selection_prob_child * (prob_parent_split_given_no_split_child + (1.0f - prob_parent_split_given_no_split_child) * T);
            const float parent_split_child = split_prob;

            // Store split node
            LightTreeStackEntry entry;
            entry.id           = node.child_ptr + i;
            entry.T            = light_tree_pack_probability(T_child);
            entry.parent_split = light_tree_pack_probability(parent_split_child);

            LIGHT_TREE_STACK_PUSH(stack, stack_ptr, entry);
          }
        }

        child_lights += (is_leaf) ? 1 : 0;
      }

      // This can only happen if all children were leaves
      if (selected_child == 0xFFFFFFFF) {
        node.child_ptr = 0xFFFFFFFF;
        break;
      }

      const float selection_probability = selected_importance / sum_importance;

      if (is_center_pixel(pixel) && IS_PRIMARY_RAY) {
        // printf(
        //   "RANDOM: %f %f %f => %f\n", random.x, random_shift, selected_importance,
        //   (random.x * sum_importance - random_shift) / selected_importance);
      }

      // Rescale random number
      random.x = (sum_importance > 0.0f) ? random_saturate((random.x * sum_importance - random_shift) / selected_importance) : random.x;

      const float prob_parent_split_given_no_split = (parent_split - selected_split_prob) / fmaxf(1.0f - selected_split_prob, eps);

      T            = selection_probability * (prob_parent_split_given_no_split + (1.0f - prob_parent_split_given_no_split) * T);
      parent_split = selected_split_prob;

      node = load_light_tree_node(node.child_ptr + selected_child);
    }
  }
}

__device__ uint32_t light_tree_query(const GBufferData data, const float2 random, const ushort2 pixel, RISReservoir& reservoir) {
  const LightSGData sg_data = light_sg_prepare(data);

  LightTreeStackEntry stack[LIGHT_TREE_STACK_SIZE];
  uint32_t stack_ptr = 0;

  LightTreeStackEntry root;
  root.id           = 0;
  root.T            = light_tree_pack_probability(1.0f);
  root.parent_split = light_tree_pack_probability(1.0f);

  LIGHT_TREE_STACK_PUSH(stack, stack_ptr, root);

  uint32_t light_id = 0xFFFFFFFF;

  light_tree_traverse(sg_data, data.position, data.normal, pixel, random, stack, stack_ptr, reservoir, light_id);

  return light_id;
}

__device__ TriangleHandle light_tree_get_light(const uint32_t id, DeviceTransform& transform) {
  const TriangleHandle handle = device.ptrs.light_tree_tri_handle_map[id];

  transform = load_transform(handle.instance_id);

  return handle;
}

#endif /* CU_LUMINARY_LIGHT_TREE_H */