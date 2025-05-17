#ifndef CU_LUMINARY_LIGHT_TREE_H
#define CU_LUMINARY_LIGHT_TREE_H

#include "light_sg.cuh"
#include "math.cuh"
#include "memory.cuh"
#include "ris.cuh"
#include "utils.cuh"

#define LIGHT_TREE_NUM_OUTPUTS 32
#define LIGHT_TREE_INVALID_NODE 0xFFFFFF

struct LightTreeWorkEntry {
  union {
    struct {
      uint32_t num_samples : 8, node_ptr : 24;
    };
    struct {
      uint32_t material_layer_mask : 8, light_ptr : 24;
    };
  };
  float weight;
} typedef LightTreeWorkEntry;
LUM_STATIC_SIZE_ASSERT(LightTreeWorkEntry, 0x08);

struct LightTreeWork {
  LightTreeWorkEntry data[LIGHT_TREE_NUM_OUTPUTS];
  uint32_t num_outputs;
} typedef LightTreeWork;

__device__ float light_tree_child_importance_diffuse(const GBufferData data, const float power, const vec3 mean, const float variance) {
  const vec3 PO = sub_vector(mean, data.position);

#if 0
  const vec3 D      = normalize_vector(cross_product(cross_product(PO, data.normal), PO));
  const vec3 v      = normalize_vector(PO);
  const float theta = fminf(asinf(variance / get_length(PO)), acosf(dot_product(data.normal, v)));
  const vec3 L = normalize_vector(add_vector(scale_vector(v, cosf(theta)), scale_vector(D, sinf(theta))));
#else
  const vec3 L = normalize_vector(sub_vector(add_vector(mean, scale_vector(data.normal, variance)), data.position));
#endif

  const float dist_sq = fmaxf(dot_product(PO, PO), variance * variance);

  return power * __saturatef(dot_product(L, data.normal)) / dist_sq;
}

__device__ float light_tree_child_importance_microfacet_reflection(
  const GBufferData data, const float power, const vec3 mean, const float variance) {
  const vec3 PO = sub_vector(mean, data.position);

  const vec3 R          = reflect_vector(data.V, data.normal);
  const vec3 target_dir = normalize_vector(add_vector(scale_vector(data.normal, data.roughness), scale_vector(R, 1.0f - data.roughness)));
  const vec3 v          = normalize_vector(PO);
  const float length_PO = get_length(PO);
  const float R_dot_V   = dot_product(target_dir, v);

  float N_dot_H;
  if (R_dot_V > 1.0f - eps || length_PO < variance) {
    N_dot_H = 1.0f;
  }
  else {
    const vec3 D      = normalize_vector(cross_product(cross_product(PO, target_dir), PO));
    const float theta = fminf(asinf(variance / length_PO), acosf(R_dot_V));
    const vec3 u      = normalize_vector(add_vector(scale_vector(v, cosf(theta)), scale_vector(D, sinf(theta))));
    const vec3 H      = normalize_vector(add_vector(data.V, u));

    N_dot_H = dot_product(data.normal, H);
  }

  const float roughness2 = data.roughness * data.roughness;
  const float roughness4 = roughness2 * roughness2;
  // TODO: Why is roughness2 giving correct values? This seems like there is something wrong in the evaluation but I can't find anything.
  const float D = bsdf_microfacet_evaluate_D_GGX(N_dot_H, roughness2);

  const float dist_sq    = fmaxf(dot_product(PO, PO), variance * variance);
  const float power_term = lerp(1.0f, power / dist_sq, data.roughness);

  return power_term * fmaxf(D, 0.0f);
}

__device__ float light_tree_child_importance(
  const GBufferData data, const vec3 position, const vec3 normal, const DeviceLightTreeNodeSection section, const vec3 base, const vec3 exp,
  const float exp_v, const uint32_t i) {
  const float power    = (float) section.rel_power[i];
  const float variance = section.rel_variance[i] * exp_v;

  const vec3 mean = add_vector(mul_vector(get_vector(section.rel_mean_x[i], section.rel_mean_y[i], section.rel_mean_z[i]), exp), base);

  return fmaxf(light_tree_child_importance_diffuse(data, power, mean, variance), 0.0f);
}

__device__ void light_tree_traverse(
  const GBufferData data, const vec3 position, const vec3 normal, const ushort2 pixel, LightTreeWork& work) {
  RISReservoir reservoir1 = ris_reservoir_init(quasirandom_sequence_1D(QUASI_RANDOM_TARGET_RIS_LIGHT_TREE + 0, pixel));
  RISReservoir reservoir2 = ris_reservoir_init(quasirandom_sequence_1D(QUASI_RANDOM_TARGET_RIS_LIGHT_TREE + 1, pixel));

  uint32_t queue_read_ptr  = work.num_outputs;
  uint32_t queue_write_ptr = work.num_outputs;

  LightTreeWorkEntry root_node;
  root_node.num_samples = LIGHT_TREE_NUM_OUTPUTS - work.num_outputs;
  root_node.node_ptr    = 0;
  root_node.weight      = 1.0f;

  work.data[queue_write_ptr++] = root_node;

  while (queue_read_ptr < queue_write_ptr) {
    const LightTreeWorkEntry entry = work.data[queue_read_ptr++];

    // if (is_center_pixel(pixel)) {
    //   printf("POP: %u %f\n", node_reference.ptr, node_reference.sampling_weight);
    // }

    uint32_t node_ptr     = LIGHT_TREE_INVALID_NODE;
    uint32_t section_id   = 0;
    uint32_t light_ptr    = LIGHT_TREE_LIGHT_SUBSET_ID_NULL;
    uint32_t child_ptr    = LIGHT_TREE_INVALID_NODE;
    uint32_t split_budget = entry.num_samples;
    uint32_t selected1    = entry.node_ptr;
    uint32_t selected2    = LIGHT_TREE_INVALID_NODE;

    vec3 base   = get_vector(0.0f, 0.0f, 0.0f);
    vec3 exp    = get_vector(0.0f, 0.0f, 0.0f);
    float exp_v = 0.0f;

    float sampling_weight = entry.weight;

    reservoir1.selected_target = 1.0f;
    reservoir1.sum_weight      = 1.0f;

    bool has_next = false;

    bool has_reached_end = false;

    while (!has_reached_end) {
      if (has_next == false) {
        if (selected1 != LIGHT_TREE_INVALID_NODE || selected2 != LIGHT_TREE_INVALID_NODE) {
          uint32_t split_factor = 0;
          split_factor += (selected1 != LIGHT_TREE_INVALID_NODE) ? 1 : 0;
          split_factor += (selected2 != LIGHT_TREE_INVALID_NODE && selected2 != selected1) ? 1 : 0;

          LightTreeWorkEntry new_entry;

          if (selected2 != LIGHT_TREE_INVALID_NODE && selected2 != selected1) {
            uint32_t budget_this_node = (selected1 != LIGHT_TREE_INVALID_NODE) ? split_budget >> 1 : split_budget;

            split_budget -= budget_this_node;

            new_entry.num_samples = budget_this_node;
            new_entry.node_ptr    = selected2;
            new_entry.weight      = sampling_weight * ris_reservoir_get_sampling_weight(reservoir2) * (1.0f / split_factor);

            if (selected1 != LIGHT_TREE_INVALID_NODE) {
              work.data[queue_write_ptr++] = new_entry;
            }
          }

          if (selected1 != LIGHT_TREE_INVALID_NODE) {
            uint32_t budget_this_node = split_budget;

            split_budget -= budget_this_node;

            new_entry.num_samples = budget_this_node;
            new_entry.node_ptr    = selected1;
            new_entry.weight      = sampling_weight * ris_reservoir_get_sampling_weight(reservoir1) * (1.0f / split_factor);
          }

          node_ptr        = new_entry.node_ptr;
          split_budget    = new_entry.num_samples;
          sampling_weight = new_entry.weight;

          ris_reservoir_reset(reservoir1);
          ris_reservoir_reset(reservoir2);

          selected1 = LIGHT_TREE_INVALID_NODE;
          selected2 = LIGHT_TREE_INVALID_NODE;
        }
        else {
          has_reached_end = true;
          break;
        }

        if (node_ptr == LIGHT_TREE_INVALID_NODE) {
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
        LightTreeWorkEntry output_entry;
        output_entry.weight              = sampling_weight;
        output_entry.light_ptr           = light_ptr;
        output_entry.material_layer_mask = 0xFF;

        work.data[work.num_outputs++] = output_entry;
        light_ptr                     = LIGHT_TREE_LIGHT_SUBSET_ID_NULL;
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
        float target               = 0.0f;
        uint32_t offset_this_child = 0;

        if (section.rel_power[rel_child_id] > 0) {
          target            = light_tree_child_importance(data, position, normal, section, base, exp, exp_v, rel_child_id);
          offset_this_child = (((section.meta >> (rel_child_id * 2)) & 0x3) << LIGHT_TREE_CHILD_OFFSET_STRIDE_LOG) + 1;
        }

        if (ris_reservoir_add_sample(reservoir1, target, 1.0f)) {
          selected1 = child_ptr;
        }

        if (split_budget > 1 && ris_reservoir_add_sample(reservoir2, target, 1.0f)) {
          selected2 = child_ptr;
        }

        child_ptr += offset_this_child;
      }
    }
  }
}

__device__ void light_tree_work_init(LightTreeWork& work) {
  work.num_outputs = 0;
}

__device__ void light_tree_query(const GBufferData data, const ushort2 pixel, LightTreeWork& work) {
  light_tree_traverse(data, data.position, data.normal, pixel, work);
}

__device__ TriangleHandle light_tree_get_light(const uint32_t id, DeviceTransform& transform) {
  const TriangleHandle handle = device.ptrs.light_tree_tri_handle_map[id];

  transform = load_transform(handle.instance_id);

  return handle;
}

#endif /* CU_LUMINARY_LIGHT_TREE_H */
