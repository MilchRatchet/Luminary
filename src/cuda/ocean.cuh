#ifndef CU_OCEAN_H
#define CU_OCEAN_H

#include "brdf.cuh"
#include "math.cuh"
#include "memory.cuh"
#include "ocean_utils.cuh"

//
// In this ocean implementation the surface shape is defined by a function based on the shadertoy by
// Alexander Alekseev aka TDM (https://www.shadertoy.com/view/Ms2SD1).
// The intersection of the ray with the surface is handled through a ray marcher that uses an
// approximate Lipschitz factor of the surface function to obtain a function similar to an SDF.
// The shading of the ocean and the water beneath is based on
// M. Droske, J. Hanika, J. Vorba, A. Weidlich, M. Sabbadin, "Path Tracing in Production: The Path of Water", ACM SIGGRAPH 2023 Courses,
// 2023.
// The water is handled by the volume implementation.
//

LUM_DEVICE_FUNC vec3 ocean_get_normal(const vec3 p) {
  const float d = (OCEAN_LIPSCHITZ + get_length(p)) * eps;

  // Sobel filter
  float h[8];
  h[0] = ocean_get_height(add_vector(p, get_vector(-d, 0.0f, d)), OCEAN_ITERATIONS_NORMAL);
  h[1] = ocean_get_height(add_vector(p, get_vector(0.0f, 0.0f, d)), OCEAN_ITERATIONS_NORMAL);
  h[2] = ocean_get_height(add_vector(p, get_vector(d, 0.0f, d)), OCEAN_ITERATIONS_NORMAL);
  h[3] = ocean_get_height(add_vector(p, get_vector(-d, 0.0f, 0.0f)), OCEAN_ITERATIONS_NORMAL);
  h[4] = ocean_get_height(add_vector(p, get_vector(d, 0.0f, 0.0f)), OCEAN_ITERATIONS_NORMAL);
  h[5] = ocean_get_height(add_vector(p, get_vector(-d, 0.0f, -d)), OCEAN_ITERATIONS_NORMAL);
  h[6] = ocean_get_height(add_vector(p, get_vector(0.0f, 0.0f, -d)), OCEAN_ITERATIONS_NORMAL);
  h[7] = ocean_get_height(add_vector(p, get_vector(d, 0.0f, -d)), OCEAN_ITERATIONS_NORMAL);

  vec3 normal;
  normal.x = -((h[7] + 2.0f * h[4] + h[2]) - (h[5] + 2.0f * h[3] + h[0])) / 8.0f;
  normal.y = d;
  normal.z = -((h[0] + 2.0f * h[1] + h[2]) - (h[5] + 2.0f * h[6] + h[7])) / 8.0f;

  return normalize_vector(normal);
}

/*
 * This uses the actual Fresnel equations to compute the reflection coefficient under the following assumptions:
 *  - The media are not magnetic.
 *  - The light is not polarized.
 *  - The IORs are wavelength independent.
 */
LUM_DEVICE_FUNC float ocean_reflection_coefficient(
  const vec3 normal, const vec3 ray, const vec3 refraction, const float index_in, const float index_out) {
  const float NdotV = -dot_product(ray, normal);
  const float NdotT = -dot_product(refraction, normal);

  const float s_pol_term1 = index_in * NdotV;
  const float s_pol_term2 = index_out * NdotT;

  const float p_pol_term1 = index_in * NdotT;
  const float p_pol_term2 = index_out * NdotV;

  float reflection_s_pol = (s_pol_term1 - s_pol_term2) / (s_pol_term1 + s_pol_term2);
  float reflection_p_pol = (p_pol_term1 - p_pol_term2) / (p_pol_term1 + p_pol_term2);

  reflection_s_pol *= reflection_s_pol;
  reflection_p_pol *= reflection_p_pol;

  return __saturatef(0.5f * (reflection_s_pol + reflection_p_pol));
}

#endif /* CU_OCEAN_H */
