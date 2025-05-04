#ifndef CU_LUMINARY_LIGHT_TREE_H
#define CU_LUMINARY_LIGHT_TREE_H

#include "light_linked_list.cuh"
#include "light_sg.cuh"
#include "math.cuh"
#include "memory.cuh"
#include "ris.cuh"
#include "utils.cuh"

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

__device__ float light_tree_child_importance(
  const LightSGData data, const vec3 position, const vec3 normal, const DeviceLightTreeNodeSection section, const vec3 base, const vec3 exp,
  const float exp_v, const uint32_t i) {
  const float power    = (float) section.rel_power[i];
  const float variance = section.rel_variance[i] * exp_v;

  const vec3 mean = add_vector(mul_vector(get_vector(section.rel_mean_x[i], section.rel_mean_y[i], section.rel_mean_z[i]), exp), base);

  return fmaxf(light_sg_evaluate(data, position, normal, mean, variance, power), 0.0f);
}

__device__ void light_tree_traverse(
  const LightSGData data, const vec3 position, const vec3 normal, const ushort2 pixel, float random,
  LightLinkedListReference stack[LIGHT_LINKED_LIST_MAX_REFERENCES], uint32_t& stack_ptr) {
  random = random_saturate(random);

  uint32_t node_ptr     = 0xFFFFFFFF;
  uint32_t section_id   = 0;
  uint32_t light_ptr    = LIGHT_TREE_LINKED_LIST_NULL;
  uint32_t child_ptr    = 0xFFFFFFFF;
  uint32_t selected_ptr = 0;

  vec3 base   = get_vector(0.0f, 0.0f, 0.0f);
  vec3 exp    = get_vector(0.0f, 0.0f, 0.0f);
  float exp_v = 0.0f;

  float sampling_weight = 1.0f;

  bool has_next = false;

  bool has_reached_end = false;

  RISReservoir reservoir = ris_reservoir_init(random);

  while (!has_reached_end) {
    if (has_next == false) {
      if (selected_ptr != 0xFFFFFFFF) {
        node_ptr = selected_ptr;

        selected_ptr = 0xFFFFFFFF;

        sampling_weight *= ris_reservoir_get_sampling_weight(reservoir);
        ris_reservoir_reset(reservoir);
      }
      else {
        has_reached_end = true;
        break;
      }

      if (node_ptr == 0xFFFFFFFF) {
        has_reached_end = true;
        break;
      }

      const DeviceLightTreeNodeHeader header = device.ptrs.light_tree_nodes[node_ptr];

      base       = get_vector(bfloat_unpack(header.x), bfloat_unpack(header.y), bfloat_unpack(header.z));
      exp        = get_vector(exp2f(header.exp_x), exp2f(header.exp_y), exp2f(header.exp_z));
      exp_v      = exp2f(header.exp_variance);
      child_ptr  = ((uint32_t) header.child_and_light_ptr[0]) | (((uint32_t) header.child_and_light_ptr[1] & 0x00FF) << 16);
      light_ptr  = ((uint32_t) header.child_and_light_ptr[2]) | (((uint32_t) header.child_and_light_ptr[1] & 0xFF00) << 8);
      section_id = 0;
    }

    if (light_ptr != LIGHT_TREE_LINKED_LIST_NULL) {
      LightLinkedListReference ref;
      ref.id              = light_ptr;
      ref.sampling_weight = sampling_weight;

      stack[stack_ptr++] = ref;

      light_ptr = LIGHT_TREE_LINKED_LIST_NULL;
    }

    // This indicates that we have no children
    if (child_ptr == 0) {
      has_reached_end = true;
      break;
    }

    // TODO: Apply a prefetch hint based on LIGHT_TREE_CHILD_OFFSET_STRIDE
    const DeviceLightTreeNodeSection section = ((DeviceLightTreeNodeSection*) device.ptrs.light_tree_nodes)[node_ptr + 1 + section_id];
    section_id++;

    has_next = (section.meta & LIGHT_TREE_META_HAS_NEXT) != 0;

#pragma unroll
    for (uint32_t rel_child_id = 0; rel_child_id < 3; rel_child_id++) {
      float target               = 0.0f;
      uint32_t offset_this_child = 0;

      if (section.rel_power[rel_child_id] > 0) {
        target            = light_tree_child_importance(data, position, normal, section, base, exp, exp_v, rel_child_id);
        offset_this_child = (((section.meta >> (rel_child_id * 2)) & 0x3) << LIGHT_TREE_CHILD_OFFSET_STRIDE_LOG) + 1;
      }

      if (ris_reservoir_add_sample(reservoir, target, 1.0f)) {
        selected_ptr = child_ptr;
      }

      child_ptr += offset_this_child;
    }
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
