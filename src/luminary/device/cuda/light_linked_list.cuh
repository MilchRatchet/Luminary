#ifndef CU_LUMINARY_LIGHT_LINKED_LIST_H
#define CU_LUMINARY_LIGHT_LINKED_LIST_H

#include "light_ltc.cuh"
#include "ris.cuh"
#include "utils.cuh"

#define LIGHT_LINKED_LIST_MAX_REFERENCES 32

struct LightLinkedListReference {
  uint32_t id;
  float probability;
} typedef LightLinkedListReference;
LUM_STATIC_SIZE_ASSERT(LightLinkedListReference, 0x08);

typedef uint16_t BFloat16;

__device__ float _bfloat_to_float(const BFloat16 val) {
  const uint32_t data = val;

  return __uint_as_float(data << 16);
}

__device__ BFloat16 _float_to_bfloat(const float val) {
  // Add this term to round to nearest
  const uint32_t data = __float_as_uint(val) + (1u << 15);

  return (BFloat16) (data >> 16);
}

__device__ uint32_t light_linked_list_resample(
  const GBufferData data, const LightLinkedListReference stack[LIGHT_LINKED_LIST_MAX_REFERENCES], const uint32_t num_references,
  RISReservoir& reservoir) {
  const Quaternion local_space = quaternion_rotation_to_z_canonical(data.normal);
  const vec3 V_local           = quaternion_apply(local_space, data.V);
  const LTCMatrix ltc_matrix   = light_ltc_load(V_local, data.roughness, data.roughness);

  const bool include_diffuse = ((data.flags & G_BUFFER_FLAG_METALLIC) == 0) && (GBUFFER_IS_SUBSTRATE_OPAQUE(data.flags));

  uint32_t reference_ptr   = 0;
  uint32_t linked_list_ptr = 0;
  uint32_t section_id      = 0;

  float linked_list_sampling_weight = 1.0f;

  bool reached_end = false;

  DeviceLightLinkedListHeader header;
  header.meta = 0;

  vec3 base_point;
  vec3 exp;
  float max_intensity;

  uint32_t selected_light_id = 0xFFFFFFFF;

  while (!reached_end) {
    if (section_id >= (header.meta & (LIGHT_LINKED_LIST_META_HAS_NEXT - 1))) {
      if ((header.meta & LIGHT_LINKED_LIST_META_HAS_NEXT) == 0) {
        if (reference_ptr < num_references) {
          const LightLinkedListReference reference = stack[reference_ptr++];

          linked_list_ptr             = reference.id;
          linked_list_sampling_weight = 1.0f / reference.probability;
        }
        else {
          reached_end = true;
          break;
        }
      }

      header = load_light_linked_list_header(linked_list_ptr);
      linked_list_ptr += (sizeof(DeviceLightLinkedListHeader) / sizeof(float4));

      base_point    = get_vector(_bfloat_to_float(header.x), _bfloat_to_float(header.y), _bfloat_to_float(header.z));
      exp           = get_vector(exp2f(header.exp_x), exp2f(header.exp_y), exp2f(header.exp_z));
      max_intensity = _bfloat_to_float(header.intensity);
      section_id    = 0;
    }

    const DeviceLightLinkedListSection section = load_light_linked_list_section(linked_list_ptr);
    linked_list_ptr += (sizeof(DeviceLightLinkedListSection) / sizeof(float4));

#pragma unroll
    for (uint32_t tri_id = 0; tri_id < 4; tri_id++) {
      if (section.intensity == 0)
        continue;

      vec3 v0 = get_vector(section.v0_x[tri_id], section.v0_y[tri_id], section.v0_z[tri_id]);
      v0      = add_vector(mul_vector(v0, exp), base_point);

      vec3 v1 = get_vector(section.v1_x[tri_id], section.v1_y[tri_id], section.v1_z[tri_id]);
      v1      = add_vector(mul_vector(v1, exp), base_point);

      vec3 v2 = get_vector(section.v2_x[tri_id], section.v2_y[tri_id], section.v2_z[tri_id]);
      v2      = add_vector(mul_vector(v2, exp), base_point);

      float intensity = section.intensity[tri_id] * max_intensity;

      intensity *= light_ltc_triangle_integral(ltc_matrix, include_diffuse, data.position, local_space, v0, v1, v2);

      if (ris_reservoir_add_sample(reservoir, intensity, linked_list_sampling_weight)) {
        selected_light_id = header.light_id + section_id * 4 + tri_id;
      }
    }

    section_id++;
  }

  return selected_light_id;
}

#endif /* CU_LUMINARY_LIGHT_LINKED_LIST_H */
