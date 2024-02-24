#ifndef CU_BSDF_H
#define CU_BSDF_H

#include "bsdf_utils.cuh"
#include "random.cuh"
#include "utils.cuh"

__device__ RGBF bsdf_evaluate(const GBufferData data, const vec3 L) {
  float NdotL       = dot_product(data.normal, L);
  const float NdotV = dot_product(data.normal, data.V);

  // Refraction rays can leave the surface and reflections could enter the surface, we hack here and simply swap their meaning around based
  // on if they enter or leave, otherwise it is not clear which of the two a ray should be.
  const bool is_refraction = (NdotL < 0.0f);

  const float ambient_ior = bsdf_refraction_index_ambient(data);
  const float ior_in      = (data.flags & G_BUFFER_REFRACTION_IS_INSIDE) ? ambient_ior : data.refraction_index;
  const float ior_out     = (data.flags & G_BUFFER_REFRACTION_IS_INSIDE) ? data.refraction_index : ambient_ior;

  vec3 reflection_vector, refraction_vector, H;

  // For every refraction exists a corresponding reflection vector, we always compute the BSDF based on the reflection.
  if (is_refraction) {
    NdotL = fabsf(NdotL);

    H = bsdf_refraction_normal_from_pair(L, data.V, ior_out, ior_in);

    reflection_vector = reflect_vector(scale_vector(data.V, -1.0f), H);
    refraction_vector = L;
  }
  else {
    H = normalize_vector(add_vector(data.V, L));

    reflection_vector = L;
    refraction_vector = refract_ray(scale_vector(data.V, -1.0f), H, ior_in / ior_out);
  }

  const float NdotH = dot_product(data.normal, H);
  const float HdotV = __saturatef(dot_product(H, data.V));

  const RGBF f0      = bsdf_fresnel_normal_incidence(data);
  const RGBF fresnel = bsdf_fresnel_composite(data, reflection_vector, refraction_vector, ior_in, ior_out, f0, HdotV);
  const RGBF diffuse = bsdf_diffuse_color(data);

  return bsdf_multiscattering_evaluate(data, fresnel, f0, diffuse, opaque_color(data.albedo), NdotH, NdotL, NdotV, is_refraction);
}

__device__ vec3 bsdf_sample(const GBufferData data, const ushort2 pixel, RGBF& weight) {
  // Transformation to +Z-Up
  const Quaternion rotation_to_z = get_rotation_to_z_canonical(data.normal);
  const vec3 V_local             = rotate_vector_by_quaternion(data.V, rotation_to_z);

  // G Buffer Data (+Z-Up)
  GBufferData data_local = data;

  data_local.V      = V_local;
  data_local.normal = get_vector(0.0f, 0.0f, 1.0f);

  // Microfacet Normal (+Z-Up)
  vec3 microfacet_normal_local = get_vector(0.0f, 0.0f, 1.0f);
  if (data_local.roughness > 0.0f) {
    const float2 microfacet_normal_random = quasirandom_sequence_2D(QUASI_RANDOM_TARGET_BSDF_NORMAL, pixel);
    microfacet_normal_local               = bsdf_microfacet_sample_normal(data_local, V_local, microfacet_normal_random);
  }

  // Specular Reflection Vector (+Z-Up)
  const vec3 spec_reflection_local = reflect_vector(scale_vector(V_local, -1.0f), microfacet_normal_local);

  // Refraction Vector (+Z-Up)
  const float refraction_index = bsdf_refraction_index(data_local);
  const vec3 refraction_local  = refract_ray(scale_vector(V_local, -1.0f), microfacet_normal_local, refraction_index);

  // Diffuse Vector (+Z-Up)
  const float2 diffuse_random = quasirandom_sequence_2D(QUASI_RANDOM_TARGET_BSDF_DIFFUSE, pixel);
  const vec3 diffuse_local    = bsdf_diffuse_sample(diffuse_random);
  const float diffuse_pdf     = bsdf_diffuse_pdf(data_local.normal, diffuse_local);

  // Evaluate BSDF for each sample
  const RGBF spec_reflection_bsdf = bsdf_evaluate(data_local, spec_reflection_local);
  const RGBF refraction_bsdf      = bsdf_evaluate(data_local, refraction_local);
  const RGBF diffuse_bsdf         = bsdf_evaluate(data_local, diffuse_local);

  // Compute weight for each sample
  const float spec_reflection_weight = luminance(spec_reflection_bsdf);
  const float refraction_weight      = luminance(refraction_bsdf);
  const float diffuse_weight         = luminance(diffuse_bsdf);

  const float weight_normalization_term = 1.0f / (spec_reflection_weight + refraction_weight + diffuse_weight);

  // Compute probability of choosing each sample
  const float spec_reflection_probability = spec_reflection_weight * weight_normalization_term;
  const float refraction_probability      = refraction_weight * weight_normalization_term;
  const float diffuse_probability         = diffuse_weight * weight_normalization_term;

  const float choice_random = quasirandom_sequence_1D(QUASI_RANDOM_TARGET_BSDF_CHOICE, pixel);

  vec3 choice_local;
  float choice_pdf;
  RGBF choice_bsdf;
  float choice_probability;

  if (choice_random < spec_reflection_probability) {
    choice_local       = spec_reflection_local;
    choice_pdf         = spec_reflection_pdf;
    choice_bsdf        = spec_reflection_bsdf;
    choice_probability = spec_reflection_probability;
  }
  else if (choice_random < spec_reflection_probability + refraction_probability) {
    choice_local       = refraction_local;
    choice_pdf         = refraction_pdf;
    choice_bsdf        = refraction_bsdf;
    choice_probability = refraction_probability;
  }
  else {
    choice_local       = diffuse_local;
    choice_pdf         = diffuse_pdf;
    choice_bsdf        = diffuse_bsdf;
    choice_probability = diffuse_probability;
  }

  vec3 choice = normalize_vector(rotate_vector_by_quaternion(choice_local, inverse_quaternion(rotation_to_z)));

  weight = scale_color(choice_bsdf, 1.0f / (choice_probability * choice_pdf));

  return choice;
}

#endif /* CU_BSDF_H */
