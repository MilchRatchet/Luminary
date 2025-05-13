#ifndef CU_LUMINARY_LIGHT_TREE_H
#define CU_LUMINARY_LIGHT_TREE_H

#include "light_sg.cuh"
#include "math.cuh"
#include "memory.cuh"
#include "ris.cuh"
#include "utils.cuh"

#define LIGHT_TREE_NUM_TARGETS 2
#define LIGHT_TREE_NUM_STD_SAMPLES 2
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

#define LIGHT_TREE_NODE_REFERENCE_BUDGET_SHIFT 24
#define LIGHT_TREE_NODE_REFERENCE_OFFSET_MASK ((1 << LIGHT_TREE_NODE_REFERENCE_BUDGET_SHIFT) - 1)

struct LightTreeOutput {
  float sampling_weight;
  uint32_t light_id;
} typedef LightTreeOutput;

#define LIGHT_TREE_NUM_OUTPUTS 32
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

  target[0] = fmaxf(light_sg_evaluate(data, position, normal, mean, variance, power, target[1]), 0.0f);
}

__device__ void light_tree_traverse(
  const LightSGData data, const vec3 position, const vec3 normal, const ushort2 pixel, LightTreeOutput outputs[LIGHT_TREE_NUM_OUTPUTS]) {
  RISReservoir reservoir1 = ris_reservoir_init(quasirandom_sequence_1D(QUASI_RANDOM_TARGET_RIS_LIGHT_TREE + 0, pixel));
  RISReservoir reservoir2 = ris_reservoir_init(quasirandom_sequence_1D(QUASI_RANDOM_TARGET_RIS_LIGHT_TREE + 1, pixel));

  uint32_t output_ptr = 0;

  // TODO: Turn this into a queue to better handle degenerate tree where we would easily overrun memory (i.e. completely unbalanced tree)
  LightTreeNodeReference node_stack[LIGHT_TREE_MAX_NODE_REFERENCES];
  uint32_t node_stack_ptr = 0;

  LightTreeNodeReference root_node;
  root_node.ptr             = 0 | (LIGHT_TREE_NUM_OUTPUTS << LIGHT_TREE_NODE_REFERENCE_BUDGET_SHIFT);
  root_node.sampling_weight = 1.0f;

  node_stack[node_stack_ptr++] = root_node;

  const uint32_t num_lanes[LIGHT_TREE_NUM_TARGETS] = {LIGHT_TREE_NUM_STD_SAMPLES, LIGHT_TREE_NUM_DEFENSIVE_SAMPLES};

  while (node_stack_ptr) {
    const LightTreeNodeReference node_reference = node_stack[--node_stack_ptr];

    // if (is_center_pixel(pixel)) {
    //   printf("POP: %u %f\n", node_reference.ptr, node_reference.sampling_weight);
    // }

    uint32_t node_ptr     = 0xFFFFFFFF;
    uint32_t section_id   = 0;
    uint32_t light_ptr    = LIGHT_TREE_LIGHT_SUBSET_ID_NULL;
    uint32_t child_ptr    = 0xFFFFFFFF;
    uint32_t split_budget = node_reference.ptr >> LIGHT_TREE_NODE_REFERENCE_BUDGET_SHIFT;

    uint32_t selected1 = node_reference.ptr & LIGHT_TREE_NODE_REFERENCE_OFFSET_MASK;
    uint32_t selected2 = 0xFFFFFFFF;

    vec3 base   = get_vector(0.0f, 0.0f, 0.0f);
    vec3 exp    = get_vector(0.0f, 0.0f, 0.0f);
    float exp_v = 0.0f;

    float sampling_weight = node_reference.sampling_weight;

    reservoir1.selected_target = 1.0f;
    reservoir1.sum_weight      = 1.0f;

    bool has_next = false;

    bool has_reached_end = false;

    while (!has_reached_end) {
      if (has_next == false) {
        if (selected1 != 0xFFFFFFFF || selected2 != 0xFFFFFFFF) {
          uint32_t split_factor = 0;
          split_factor += (selected1 != 0xFFFFFFFF) ? 1 : 0;
          split_factor += (selected2 != 0xFFFFFFFF && selected2 != selected1) ? 1 : 0;

          LightTreeNodeReference new_node;

          if (selected2 != 0xFFFFFFFF && selected2 != selected1) {
            uint32_t budget_this_node = (selected1 != 0xFFFFFFFF) ? split_budget >> 1 : split_budget;

            split_budget -= budget_this_node;

            new_node.ptr             = selected2 | (budget_this_node << LIGHT_TREE_NODE_REFERENCE_BUDGET_SHIFT);
            new_node.sampling_weight = sampling_weight * ris_reservoir_get_sampling_weight(reservoir2) * (1.0f / split_factor);

            if (selected1 != 0xFFFFFFFF) {
              node_stack[node_stack_ptr++] = new_node;
            }
          }

          if (selected1 != 0xFFFFFFFF) {
            uint32_t budget_this_node = split_budget;

            split_budget -= budget_this_node;

            new_node.ptr             = selected1 | (budget_this_node << LIGHT_TREE_NODE_REFERENCE_BUDGET_SHIFT);
            new_node.sampling_weight = sampling_weight * ris_reservoir_get_sampling_weight(reservoir1) * (1.0f / split_factor);
          }

          node_ptr        = new_node.ptr & LIGHT_TREE_NODE_REFERENCE_OFFSET_MASK;
          split_budget    = new_node.ptr >> LIGHT_TREE_NODE_REFERENCE_BUDGET_SHIFT;
          sampling_weight = new_node.sampling_weight;

          ris_reservoir_reset(reservoir1);
          ris_reservoir_reset(reservoir2);

          selected1 = 0xFFFFFFFF;
          selected2 = 0xFFFFFFFF;
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
        LightTreeOutput output;
        output.sampling_weight = sampling_weight;
        output.light_id        = light_ptr;

        outputs[output_ptr++] = output;
        light_ptr             = LIGHT_TREE_LIGHT_SUBSET_ID_NULL;
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

        if (ris_reservoir_add_sample(reservoir1, target[0], 1.0f)) {
          selected1 = child_ptr;
        }

        if (split_budget > 1 && ris_reservoir_add_sample(reservoir2, target[0], 1.0f)) {
          selected2 = child_ptr;
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
