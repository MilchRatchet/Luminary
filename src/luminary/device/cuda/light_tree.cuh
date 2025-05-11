#ifndef CU_LUMINARY_LIGHT_TREE_H
#define CU_LUMINARY_LIGHT_TREE_H

#include "light_sg.cuh"
#include "math.cuh"
#include "memory.cuh"
#include "ris.cuh"
#include "utils.cuh"

#define LIGHT_TREE_NUM_TARGETS 2
#define LIGHT_TREE_NUM_STD_SAMPLES 3
#define LIGHT_TREE_NUM_DEFENSIVE_SAMPLES 1

struct LightSubsetReference {
  uint32_t subset_id;
  float sampling_weight;
} typedef LightSubsetReference;
LUM_STATIC_SIZE_ASSERT(LightSubsetReference, 0x08);

struct LightTreeNodeReference {
  uint32_t ptr;
  float sampling_weight;
} typedef LightTreeNodeReference;
LUM_STATIC_SIZE_ASSERT(LightTreeNodeReference, 0x08);

struct LightTreeOutput {
  RISReservoir reservoir;
  uint32_t light_id;
} typedef LightTreeOutput;

#define LIGHT_TREE_NUM_OUTPUTS 16
#define LIGHT_TREE_MAX_NODE_REFERENCES 64

typedef uint16_t PackedProb;

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

__device__ void light_tree_child_importance(
  const LightSGData data, const vec3 position, const vec3 normal, const DeviceLightTreeNodeSection section, const vec3 base, const vec3 exp,
  const float exp_v, const uint32_t i, float target[LIGHT_TREE_NUM_TARGETS]) {
  const float power    = (float) section.rel_power[i];
  const float variance = section.rel_variance[i] * exp_v;

  const vec3 mean = add_vector(mul_vector(get_vector(section.rel_mean_x[i], section.rel_mean_y[i], section.rel_mean_z[i]), exp), base);

  target[0] = fmaxf(light_sg_evaluate(data, position, normal, mean, variance, power), 0.0f);
  target[1] = 1.0f;  // TODO
}

__device__ void light_tree_traverse(
  const LightSGData data, const vec3 position, const vec3 normal, const ushort2 pixel, LightTreeOutput outputs[LIGHT_TREE_NUM_OUTPUTS]) {
  MultiRISAggregator<LIGHT_TREE_NUM_TARGETS> aggregator = multi_ris_aggregator_init<LIGHT_TREE_NUM_TARGETS>();

  // TODO: Fix random targets
  MultiRISLane<LIGHT_TREE_NUM_TARGETS, 0> lanes_std[LIGHT_TREE_NUM_STD_SAMPLES];
  for (uint32_t lane_id = 0; lane_id < LIGHT_TREE_NUM_STD_SAMPLES; lane_id++) {
    const float lane_random = quasirandom_sequence_1D(QUASI_RANDOM_TARGET_RIS_LIGHT_TREE + lane_id, pixel);
    lanes_std[lane_id]      = multi_ris_lane_init<LIGHT_TREE_NUM_TARGETS, 0>(lane_random);
  }

  MultiRISLane<LIGHT_TREE_NUM_TARGETS, 1> lanes_defensive[LIGHT_TREE_NUM_DEFENSIVE_SAMPLES];
  for (uint32_t lane_id = 0; lane_id < LIGHT_TREE_NUM_DEFENSIVE_SAMPLES; lane_id++) {
    const float lane_random  = quasirandom_sequence_1D(QUASI_RANDOM_TARGET_RIS_LIGHT_TREE + lane_id, pixel);
    lanes_defensive[lane_id] = multi_ris_lane_init<LIGHT_TREE_NUM_TARGETS, 1>(lane_random);
  }

  uint32_t output_ptr = 0;

  // TODO: Turn this into a queue to better handle degenerate tree where we would easily overrun memory (i.e. completely unbalanced tree)
  LightTreeNodeReference node_stack[LIGHT_TREE_MAX_NODE_REFERENCES];
  uint32_t node_stack_ptr = 0;

  LightTreeNodeReference root_node;
  root_node.ptr             = 0;
  root_node.sampling_weight = 1.0f;

  node_stack[node_stack_ptr++] = root_node;

  const uint32_t num_lanes[LIGHT_TREE_NUM_TARGETS] = {LIGHT_TREE_NUM_STD_SAMPLES, LIGHT_TREE_NUM_DEFENSIVE_SAMPLES};

  while (node_stack_ptr) {
    const LightTreeNodeReference node_reference = node_stack[--node_stack_ptr];

    uint32_t node_ptr   = 0xFFFFFFFF;
    uint32_t section_id = 0;
    uint32_t light_ptr  = LIGHT_TREE_LIGHT_SUBSET_ID_NULL;
    uint32_t child_ptr  = 0xFFFFFFFF;

    uint32_t selected_ptr_std[LIGHT_TREE_NUM_STD_SAMPLES];
    for (uint32_t lane_id = 0; lane_id < LIGHT_TREE_NUM_STD_SAMPLES; lane_id++) {
      selected_ptr_std[lane_id] = 0xFFFFFFFF;
    }

    uint32_t selected_ptr_defensive[LIGHT_TREE_NUM_DEFENSIVE_SAMPLES];
    for (uint32_t lane_id = 0; lane_id < LIGHT_TREE_NUM_DEFENSIVE_SAMPLES; lane_id++) {
      selected_ptr_defensive[lane_id] = 0xFFFFFFFF;
    }

    selected_ptr_std[0] = node_reference.ptr;

    vec3 base   = get_vector(0.0f, 0.0f, 0.0f);
    vec3 exp    = get_vector(0.0f, 0.0f, 0.0f);
    float exp_v = 0.0f;

    float sampling_weight = node_reference.sampling_weight;

    // TODO: Hack
    for (uint32_t target_id = 0; target_id < LIGHT_TREE_NUM_TARGETS; target_id++) {
      lanes_std->selected_target[target_id] = 1.0f;
      aggregator.sum_weight[target_id]      = 1.0f;
    }

    bool has_next = false;

    bool has_reached_end = false;

    while (!has_reached_end) {
      if (has_next == false) {
        if (selected_ptr_std[0] != 0xFFFFFFFF) {
          // Push all non-active nodes onto the stack
#pragma unroll
          for (uint32_t lane_id = 1; lane_id < LIGHT_TREE_NUM_STD_SAMPLES; lane_id++) {
            const uint32_t selected_ptr = selected_ptr_std[lane_id];

            // Ignore cases where no child was selected.
            bool already_selected = (selected_ptr == 0xFFFFFFFF);

#pragma unroll
            for (uint32_t previous_lane_id = 0; previous_lane_id < lane_id; previous_lane_id++) {
              already_selected |= (selected_ptr == selected_ptr_std[previous_lane_id]);
            }

            if (already_selected)
              continue;

            LightTreeNodeReference deferred_node;
            deferred_node.ptr             = selected_ptr;
            deferred_node.sampling_weight = sampling_weight * multi_ris_lane_get_sampling_weight(lanes_std[lane_id], aggregator, num_lanes);

            node_stack[node_stack_ptr++] = deferred_node;
          }

#pragma unroll
          for (uint32_t lane_id = 0; lane_id < LIGHT_TREE_NUM_DEFENSIVE_SAMPLES; lane_id++) {
            const uint32_t selected_ptr = selected_ptr_defensive[lane_id];

            // Ignore cases where no child was selected.
            bool already_selected = (selected_ptr == 0xFFFFFFFF);

#pragma unroll
            for (uint32_t previous_lane_id = 0; previous_lane_id < LIGHT_TREE_NUM_STD_SAMPLES; previous_lane_id++) {
              already_selected |= (selected_ptr == selected_ptr_std[previous_lane_id]);
            }

#pragma unroll
            for (uint32_t previous_lane_id = 0; previous_lane_id < lane_id; previous_lane_id++) {
              already_selected |= (selected_ptr == selected_ptr_defensive[previous_lane_id]);
            }

            if (already_selected)
              continue;

            LightTreeNodeReference deferred_node;
            deferred_node.ptr = selected_ptr;
            deferred_node.sampling_weight =
              sampling_weight * multi_ris_lane_get_sampling_weight(lanes_defensive[lane_id], aggregator, num_lanes);

            node_stack[node_stack_ptr++] = deferred_node;
          }

          // Update active node
          node_ptr = selected_ptr_std[0];
          sampling_weight *= multi_ris_lane_get_sampling_weight(lanes_std[0], aggregator, num_lanes);

          // Reset aggregator and lanes
          for (uint32_t lane_id = 0; lane_id < LIGHT_TREE_NUM_STD_SAMPLES; lane_id++) {
            selected_ptr_std[lane_id] = 0xFFFFFFFF;
            multi_ris_lane_reset(lanes_std[lane_id]);
          }

          for (uint32_t lane_id = 0; lane_id < LIGHT_TREE_NUM_DEFENSIVE_SAMPLES; lane_id++) {
            selected_ptr_defensive[lane_id] = 0xFFFFFFFF;
            multi_ris_lane_reset(lanes_defensive[lane_id]);
          }

          multi_ris_aggregator_reset(aggregator);
        }
        else {
          has_reached_end = true;
          break;
        }

        if (node_ptr == 0xFFFFFFFF) {
          has_reached_end = true;
          break;
        }

        const DeviceLightTreeNodeHeader header = load_light_tree_node_header(node_ptr);

        base       = get_vector(bfloat_unpack(header.x), bfloat_unpack(header.y), bfloat_unpack(header.z));
        exp        = get_vector(exp2f(header.exp_x), exp2f(header.exp_y), exp2f(header.exp_z));
        exp_v      = exp2f(header.exp_variance);
        child_ptr  = ((uint32_t) header.child_and_light_ptr[0]) | (((uint32_t) header.child_and_light_ptr[1] & 0x00FF) << 16);
        light_ptr  = ((uint32_t) header.child_and_light_ptr[2]) | (((uint32_t) header.child_and_light_ptr[1] & 0xFF00) << 8);
        section_id = 0;
      }

      if (light_ptr != LIGHT_TREE_LIGHT_SUBSET_ID_NULL) {
        if (ris_reservoir_add_sample(outputs[output_ptr].reservoir, 1.0f, sampling_weight)) {
          outputs[output_ptr].light_id = light_ptr;
        }

        output_ptr = (output_ptr + 1) & (LIGHT_TREE_NUM_OUTPUTS - 1);
        light_ptr  = LIGHT_TREE_LIGHT_SUBSET_ID_NULL;
      }

      // This indicates that we have no children
      if (child_ptr == 0) {
        has_reached_end = true;
        break;
      }

      const DeviceLightTreeNodeSection section = load_light_tree_node_section(node_ptr + 1 + section_id);
      section_id++;

      has_next = (section.meta & LIGHT_TREE_META_HAS_NEXT) != 0;

#pragma unroll
      for (uint32_t rel_child_id = 0; rel_child_id < 3; rel_child_id++) {
        float target[LIGHT_TREE_NUM_TARGETS] = {0.0f, 0.0f};
        uint32_t offset_this_child           = 0;

        if (section.rel_power[rel_child_id] > 0) {
          light_tree_child_importance(data, position, normal, section, base, exp, exp_v, rel_child_id, target);
          offset_this_child = (((section.meta >> (rel_child_id * 2)) & 0x3) << LIGHT_TREE_CHILD_OFFSET_STRIDE_LOG) + 1;
        }

        multi_ris_aggregator_add_sample(aggregator, target, 1.0f);

#pragma unroll
        for (uint32_t lane_id = 0; lane_id < LIGHT_TREE_NUM_STD_SAMPLES; lane_id++) {
          if (multi_ris_lane_add_sample(lanes_std[lane_id], aggregator, target)) {
            selected_ptr_std[lane_id] = child_ptr;
          }
        }

#pragma unroll
        for (uint32_t lane_id = 0; lane_id < LIGHT_TREE_NUM_DEFENSIVE_SAMPLES; lane_id++) {
          if (multi_ris_lane_add_sample(lanes_defensive[lane_id], aggregator, target)) {
            selected_ptr_defensive[lane_id] = child_ptr;
          }
        }

        child_ptr += offset_this_child;
      }
    }
  }
}

__device__ void light_tree_query(const GBufferData data, const ushort2 pixel, LightTreeOutput light_tree_outputs[LIGHT_TREE_NUM_OUTPUTS]) {
  const LightSGData sg_data = light_sg_prepare(data);

  light_tree_traverse(sg_data, data.position, data.normal, pixel, light_tree_outputs);
}

__device__ TriangleHandle light_tree_get_light(const uint32_t id, DeviceTransform& transform) {
  const TriangleHandle handle = device.ptrs.light_tree_tri_handle_map[id];

  transform = load_transform(handle.instance_id);

  return handle;
}

#endif /* CU_LUMINARY_LIGHT_TREE_H */
