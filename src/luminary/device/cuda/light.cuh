#ifndef CU_LIGHT_H
#define CU_LIGHT_H

#if defined(SHADING_KERNEL)

#include "hashmap.cuh"
#include "intrinsics.cuh"
#include "memory.cuh"
#include "sky_utils.cuh"
#include "texture_utils.cuh"
#include "utils.cuh"
#include "volume_utils.cuh"

////////////////////////////////////////////////////////////////////
// Literature
////////////////////////////////////////////////////////////////////

// [Est24]
// A. C. Estevez and P. Lecocq and C. Hellmuth, "A Resampled Tree for Many Lights Rendering",
// ACM SIGGRAPH 2024 Talks, 2024

// [Tok24]
// Y. Tokuyoshi and S. Ikeda and P. Kulkarni and T. Harada, "Hierarchical Light Sampling with Accurate Spherical Gaussian Lighting",
// SIGGRAPH Asia 2024 Conference Papers, 2024

////////////////////////////////////////////////////////////////////
// SG Lighting
////////////////////////////////////////////////////////////////////

// Code taken from the SG lighting demo distributed with [Tok24].

#define LIGHT_SG_VARIANCE_THRESHOLD (0x1.0p-31f)
#define LIGHT_SG_INV_SQRTPI (0.56418958354775628694807945156077f)
#define LIGHT_SG_UNCERTAINTY_AGGRESSIVENESS (0.7f)

__device__ float light_sg_integral(const float sharpness) {
  return 4.0f * PI * expm1_over_x(-2.0f * sharpness);
}

// Approximate hemispherical integral for a vMF distribution (i.e. normalized SG).
__device__ float light_vmf_integral(const float cosine, const float sharpness) {
  // Interpolation factor [Tokuyoshi 2022].
  const float A          = 0.6517328826907056171791055021459f;
  const float B          = 1.3418280033141287699294252888649f;
  const float C          = 7.2216687798956709087860872386955f;
  const float steepness  = sharpness * sqrtf((0.5f * sharpness + A) / ((sharpness + B) * sharpness + C));
  const float lerpFactor = __saturatef(0.5f + 0.5f * (erff(steepness * clampf(cosine, -1.0f, 1.0f)) / erff(steepness)));

  // Interpolation between upper and lower hemispherical integrals .
  const float e = expf(-sharpness);

  return lerp(e, 1.0f, lerpFactor) / (e + 1.0f);
}

__device__ float light_sg_cosine_integral_upper_hemisphere(const float sharpness) {
  if (sharpness <= 0.5) {
    // Taylor-series approximation for the numerical stability.
    return (((((((-1.0f / 362880.0f) * sharpness + 1.0f / 40320.0f) * sharpness - 1.0f / 5040.0f) * sharpness + 1.0f / 720.0f) * sharpness
              - 1.0f / 120.0f)
               * sharpness
             + 1.0f / 24.0f)
              * sharpness
            - 1.0f / 6.0f)
             * sharpness
           + 0.5f;
  }

  return (expm1f(-sharpness) + sharpness) / (sharpness * sharpness);
}

__device__ float light_sg_cosine_integral_lower_hemisphere(const float sharpness) {
  const float e = expf(-sharpness);

  if (sharpness <= 0.5f) {
    // Taylor-series approximation for the numerical stability.
    return e
           * (((((((((1.0f / 403200.0f) * sharpness - 1.0f / 45360.0f) * sharpness + 1.0f / 5760.0f) * sharpness - 1.0f / 840.0f) * sharpness + 1.0f / 144.0f) * sharpness - 1.0f / 30.0f) * sharpness + 1.0f / 8.0f) * sharpness - 1.0f / 3.0f) * sharpness + 0.5f);
  }

  return e * (-expm1f(-sharpness) - sharpness * e) / (sharpness * sharpness);
}

__device__ float light_sg_cosine_integral(const float z, const float sharpness) {
  // Fitted approximation for t(sharpness).
  const float A  = 2.7360831611272558028247203765204f;
  const float B  = 17.02129778174187535455530451145f;
  const float C  = 4.0100826728510421403939290030394f;
  const float D  = 15.219156263147210594866010069381f;
  const float E  = 76.087896272360737270901154261082f;
  const float t  = sharpness * sqrtf(0.5f * ((sharpness + A) * sharpness + B) / (((sharpness + C) * sharpness + D) * sharpness + E));
  const float tz = t * z;

  const float lerp_factor = __saturatef(
    FLT_EPSILON * 0.5f + 0.5f * (z * erfcf(-tz) + erfcf(t))
    - 0.5f * LIGHT_SG_INV_SQRTPI * expf(-tz * tz) * expm1f(t * t * (z * z - 1.0f)) / t);

  // Interpolation between upper and lower hemispherical integrals.
  const float lowerIntegral = light_sg_cosine_integral_lower_hemisphere(sharpness);
  const float upperIntegral = light_sg_cosine_integral_upper_hemisphere(sharpness);

  return 2.0f * lerp(lowerIntegral, upperIntegral, lerp_factor);
}

__device__ float light_sg_ggx(const vec3 m, const float mat00, const float mat01, const float mat11) {
  const float det = fmaxf(mat00 * mat11 - mat01 * mat01, eps);

  const float m_roughness_x = mat11 * m.x - mat01 * m.y;
  const float m_roughness_y = -mat01 * m.x + mat00 * m.y;

  const float dot = m.x * m_roughness_x + m.y * m_roughness_y;

  const float length2 = dot / det + m.z * m.z;

  return 1.0f / (PI * sqrtf(det) * length2 * length2);
}

// Reflection lobe based the symmetric GGX VNDF.
__device__ float light_sg_ggx_reflection_pdf(const vec3 wi, const vec3 m, const float mat00, const float mat01, const float mat11) {
  const float ggx = light_sg_ggx(m, mat00, mat01, mat11);

  const float wi_roughness_x = mat00 * wi.x + mat01 * wi.y;
  const float wi_roughness_y = mat01 * wi.x + mat11 * wi.y;

  const float dot = wi.x * wi_roughness_x + wi.y * wi_roughness_y;

  return ggx / (4.0f * sqrtf(dot + wi.z * wi.z));
}

struct LightTreeRuntimeData {
  float diffuse_weight;
  float reflection_weight;
  Quaternion rotation_to_z;
  float proj_roughness_u2;
  float proj_roughness_v2;
  vec3 V_local;
  float JJT00;
  float JJT01;
  float JJT11;
  float det_JJT4;
  vec3 reflection_vec;
} typedef LightTreeRuntimeData;

__device__ LightTreeRuntimeData light_sg_prepare(const GBufferData data) {
  const Quaternion rotation_to_z = quaternion_rotation_to_z_canonical(data.normal);

  // This is already accounting for future anisotropy support.
  const float roughness_u = data.roughness;
  const float roughness_v = data.roughness;

  // Convert the roughness from slope space to projected space.
  const float roughness_u2      = roughness_u * roughness_u;
  const float roughness_v2      = roughness_v * roughness_v;
  const float proj_roughness_u2 = roughness_u2 / fmaxf(1.0f - roughness_u2, eps);
  const float proj_roughness_v2 = roughness_v2 / fmaxf(1.0f - roughness_v2, eps);

  // Compute the Jacobian J for the transformation between halfvetors and reflection vectors at halfvector = normal.
  const vec3 V_local         = quaternion_apply(rotation_to_z, data.V);
  const float V_local_length = sqrtf(V_local.x * V_local.x + V_local.y * V_local.y);
  const float view_x         = (V_local_length != 0.0f) ? V_local.x / V_local_length : 1.0f;
  const float view_y         = (V_local_length != 0.0f) ? V_local.y / V_local_length : 0.0f;

  const float reflection_jacobian00 = 0.5f * view_x;
  const float reflection_jacobian01 = -0.5f * view_y / V_local.z;
  const float reflection_jacobian10 = 0.5f * view_y;
  const float reflection_jacobian11 = 0.5f * view_x / V_local.z;

  // Compute JJ^T matrix.
  const float JJT00 = reflection_jacobian00 * reflection_jacobian00 + reflection_jacobian01 * reflection_jacobian01;
  const float JJT01 = reflection_jacobian00 * reflection_jacobian10 + reflection_jacobian01 * reflection_jacobian11;
  const float JJT11 = reflection_jacobian10 * reflection_jacobian10 + reflection_jacobian11 * reflection_jacobian11;

  const float det_JJT4 = 1.0f / (4.0f * V_local.z * V_local.z);  // = 4 * determiant(JJ^T).

  // Preprocess for the lobe visibility.
  // Approximate the reflection lobe with an SG whose axis is the perfect specular reflection vector.
  // We use a conservative sharpness to filter the visibility.
  const float roughness_max2       = fmaxf(roughness_u2, roughness_v2);
  const float reflection_sharpness = (1.0f - roughness_max2) / fmaxf(2.0f * roughness_max2, eps);
  const vec3 reflection_vec        = scale_vector(reflect_vector(data.V, data.normal), reflection_sharpness);

  LightTreeRuntimeData runtime_data;
  runtime_data.diffuse_weight = (data.flags & (G_BUFFER_FLAG_METALLIC | G_BUFFER_FLAG_BASE_SUBSTRATE_TRANSLUCENT)) ? 0.0f : 1.0f;  // TODO
  runtime_data.reflection_weight = 1.0f;
  runtime_data.rotation_to_z     = rotation_to_z;
  runtime_data.proj_roughness_u2 = proj_roughness_u2;
  runtime_data.proj_roughness_v2 = proj_roughness_v2;
  runtime_data.V_local           = V_local;
  runtime_data.JJT00             = JJT00;
  runtime_data.JJT01             = JJT01;
  runtime_data.JJT11             = JJT11;
  runtime_data.det_JJT4          = det_JJT4;
  runtime_data.reflection_vec    = reflection_vec;

  return runtime_data;
}

// In our case, all light clusters are considered omnidirectional, i.e. sharpness = 0.
__device__ float light_sg_evaluate(
  const LightTreeRuntimeData data, const vec3 position, const vec3 normal, const vec3 mean, const float variance, const float power,
  const vec3 light_normal, const bool apply_light_normal, float& uncertainty) {
  const vec3 light_vec    = sub_vector(mean, position);
  const float distance_sq = dot_product(light_vec, light_vec);
  const vec3 light_dir    = scale_vector(light_vec, rsqrt(distance_sq));

  // Clamp the variance for the numerical stability.
  const float light_variance = fmaxf(variance, LIGHT_SG_VARIANCE_THRESHOLD * distance_sq);

  float emissive = power / light_variance;

  if (apply_light_normal) {
    emissive *= fabsf(dot_product(light_normal, light_dir));
  }

  // Compute SG sharpness for a light distribution viewed from the shading point.
  const float light_sharpness = distance_sq / light_variance;

  uncertainty = expf(-light_sharpness * device.settings.light_num_ris_samples);

  // Axis of the SG product lobe.
  const vec3 product_vec          = add_vector(data.reflection_vec, scale_vector(light_dir, light_sharpness));
  const float product_sharpness   = get_length(product_vec);
  const vec3 product_dir          = scale_vector(product_vec, 1.0f / product_sharpness);
  const float light_lobe_variance = 1.0f / light_sharpness;

  const float filtered_proj_roughness00 = data.proj_roughness_u2 + 2.0f * light_lobe_variance * data.JJT00;
  const float filtered_proj_roughness01 = 2.0f * light_lobe_variance * data.JJT01;
  const float filtered_proj_roughness11 = data.proj_roughness_v2 + 2.0f * light_lobe_variance * data.JJT11;

  // Compute determinant(filteredProjRoughnessMat) in a numerically stable manner.
  const float det = data.proj_roughness_u2 * data.proj_roughness_v2
                    + 2.0f * light_lobe_variance * (data.proj_roughness_u2 * data.JJT00 + data.proj_roughness_v2 * data.JJT11)
                    + light_lobe_variance * light_lobe_variance * data.det_JJT4;

  // NDF filtering in a numerically stable manner.
  const float tr = filtered_proj_roughness00 + filtered_proj_roughness11;

  const float denom       = 1.0f / (1.0f + tr + det);
  const bool denom_finite = is_non_finite(denom) == false;

  const float filtered_roughness_mat00 = (denom_finite)
                                           ? fminf(filtered_proj_roughness00 + det, FLT_MAX) * denom
                                           : fminf(filtered_proj_roughness00, FLT_MAX) / fminf(filtered_proj_roughness00 + 1.0f, FLT_MAX);
  const float filtered_roughness_mat01 = (denom_finite) ? fminf(filtered_proj_roughness01, FLT_MAX) * denom : 0.0f;
  const float filtered_roughness_mat11 = (denom_finite)
                                           ? fminf(filtered_proj_roughness11 + det, FLT_MAX) * denom
                                           : fminf(filtered_proj_roughness11, FLT_MAX) / fminf(filtered_proj_roughness11 + 1.0f, FLT_MAX);

  // Evaluate the filtered distribution.
  const vec3 H            = add_vector(data.V_local, quaternion_apply(data.rotation_to_z, light_dir));
  const vec3 H_normalized = scale_vector(H, 1.0f / fmaxf(get_length(H), eps));
  const float pdf =
    light_sg_ggx_reflection_pdf(data.V_local, H_normalized, filtered_roughness_mat00, filtered_roughness_mat01, filtered_roughness_mat11);

  // Microfacet reflection importance.
  const float visibility            = light_vmf_integral(dot_product(product_dir, normal), product_sharpness);
  const float reflection_importance = visibility * pdf * light_sg_integral(light_sharpness);

  // Diffuse importance.
  const float cosine             = clampf(dot_product(light_dir, normal), -1.0f, 1.0f);
  const float diffuse_importance = light_sg_cosine_integral(cosine, light_sharpness);

  return emissive * (data.diffuse_weight * diffuse_importance + data.reflection_weight * reflection_importance);
}

////////////////////////////////////////////////////////////////////
// Light Tree
////////////////////////////////////////////////////////////////////

struct LightTreeReservoir {
  float sum_weight;
  uint32_t light_id;
  float target_pdf;
  float random;
} typedef LightTreeReservoir;

#define LIGHT_TREE_STACK_SIZE 64

#define LIGHT_TREE_STACK_POP(__macro_internal_stack, __macro_internal_ptr, __macro_internal_entry) \
  (__macro_internal_entry) = (__macro_internal_stack)[--(__macro_internal_ptr)]

#define LIGHT_TREE_STACK_PUSH(__macro_internal_stack, __macro_internal_ptr, __macro_internal_entry) \
  (__macro_internal_stack)[(__macro_internal_ptr)++] = (__macro_internal_entry)

typedef uint16_t PackedProb;
typedef uint16_t BFloat16;

struct LightTreeStackEntry {
  uint32_t id;  // Either the index to a node in the tree or the index of a light, first bit set implies that its a node
  union {
    // Node
    struct {
      PackedProb T;
      PackedProb parent_split;
    };
    // Light
    struct {
      PackedProb final_prob;
      BFloat16 importance;
    };
  };
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

__device__ float light_triangle_intersection_uv(const TriangleLight triangle, const vec3 origin, const vec3 ray, float2& coords) {
  const vec3 h  = cross_product(ray, triangle.edge2);
  const float a = dot_product(triangle.edge1, h);

  const float f = 1.0f / a;
  const vec3 s  = sub_vector(origin, triangle.vertex);
  const float u = f * dot_product(s, h);

  const vec3 q  = cross_product(s, triangle.edge1);
  const float v = f * dot_product(ray, q);

  coords = make_float2(u, v);

  //  The third check is inverted to catch NaNs since NaNs always return false, the not will turn it into a true
  if (v < 0.0f || u < 0.0f || !(u + v <= 1.0f))
    return FLT_MAX;

  const float t = f * dot_product(triangle.edge2, q);

  return __fslctf(t, FLT_MAX, t);
}

__device__ TriangleLight
  light_load(const TriangleHandle handle, const vec3 origin, const vec3 ray, const DeviceTransform trans, float& dist) {
  const uint32_t mesh_id = mesh_id_load(handle.instance_id);

  const DeviceTriangle* tri_ptr = device.ptrs.triangles[mesh_id];
  const uint32_t triangle_count = __ldg(device.ptrs.triangle_counts + mesh_id);

  const float4 v0   = __ldg((float4*) triangle_get_entry_address(tri_ptr, 0, 0, handle.tri_id, triangle_count));
  const float4 v1   = __ldg((float4*) triangle_get_entry_address(tri_ptr, 1, 0, handle.tri_id, triangle_count));
  const float4 v2   = __ldg((float4*) triangle_get_entry_address(tri_ptr, 2, 0, handle.tri_id, triangle_count));
  const uint32_t v3 = __ldg((uint32_t*) triangle_get_entry_address(tri_ptr, 3, 3, handle.tri_id, triangle_count));

  TriangleLight triangle;
  triangle.vertex = get_vector(v0.x, v0.y, v0.z);
  triangle.edge1  = get_vector(v0.w, v1.x, v1.y);
  triangle.edge2  = get_vector(v1.z, v1.w, v2.x);

  triangle.vertex = transform_apply(trans, triangle.vertex);
  triangle.edge1  = transform_apply_relative(trans, triangle.edge1);
  triangle.edge2  = transform_apply_relative(trans, triangle.edge2);

  const UV vertex_texture  = uv_unpack(__float_as_uint(v2.y));
  const UV vertex1_texture = uv_unpack(__float_as_uint(v2.z));
  const UV vertex2_texture = uv_unpack(__float_as_uint(v2.w));

  float2 coords;
  dist = light_triangle_intersection_uv(triangle, origin, ray, coords);

  triangle.tex_coords  = lerp_uv(vertex_texture, vertex1_texture, vertex2_texture, coords);
  triangle.material_id = v3 & 0xFFFF;

  return triangle;
}

__device__ TriangleLight light_load_sample_init(const TriangleHandle handle, const DeviceTransform trans, uint3& packed_light_data) {
  const uint32_t mesh_id = mesh_id_load(handle.instance_id);

  const DeviceTriangle* tri_ptr = device.ptrs.triangles[mesh_id];
  const uint32_t triangle_count = __ldg(device.ptrs.triangle_counts + mesh_id);

  const float4 v0   = __ldg((float4*) triangle_get_entry_address(tri_ptr, 0, 0, handle.tri_id, triangle_count));
  const float4 v1   = __ldg((float4*) triangle_get_entry_address(tri_ptr, 1, 0, handle.tri_id, triangle_count));
  const float4 v2   = __ldg((float4*) triangle_get_entry_address(tri_ptr, 2, 0, handle.tri_id, triangle_count));
  const uint32_t v3 = __ldg((uint32_t*) triangle_get_entry_address(tri_ptr, 3, 3, handle.tri_id, triangle_count));

  TriangleLight triangle;
  triangle.vertex = get_vector(v0.x, v0.y, v0.z);
  triangle.edge1  = get_vector(v0.w, v1.x, v1.y);
  triangle.edge2  = get_vector(v1.z, v1.w, v2.x);

  triangle.vertex = transform_apply(trans, triangle.vertex);
  triangle.edge1  = transform_apply_relative(trans, triangle.edge1);
  triangle.edge2  = transform_apply_relative(trans, triangle.edge2);

  packed_light_data.x = __float_as_uint(v2.y);
  packed_light_data.y = __float_as_uint(v2.z);
  packed_light_data.z = __float_as_uint(v2.w);

  triangle.material_id = v3 & 0xFFFF;

  return triangle;
}

/*
 * Robust solid angle sampling method from
 * C. Peters, "BRDF Importance Sampling for Linear Lights", Computer Graphics Forum (Proc. HPG) 40, 8, 2021.
 */
__device__ void light_load_sample_finalize(
  TriangleLight& triangle, const uint3 packed_light_data, const vec3 origin, const float2 random, vec3& ray, float& dist,
  float& solid_angle) {
  const vec3 v0 = normalize_vector(sub_vector(triangle.vertex, origin));
  const vec3 v1 = normalize_vector(sub_vector(add_vector(triangle.vertex, triangle.edge1), origin));
  const vec3 v2 = normalize_vector(sub_vector(add_vector(triangle.vertex, triangle.edge2), origin));

  const float G0 = fabsf(dot_product(cross_product(v0, v1), v2));
  const float G1 = dot_product(v0, v2) + dot_product(v1, v2);
  const float G2 = 1.0f + dot_product(v0, v1);

  solid_angle = 2.0f * atan2f(G0, G1 + G2);

  if (is_non_finite(solid_angle) || solid_angle < 1e-7f) {
    solid_angle = 0.0f;
    dist        = 1.0f;
    return;
  }

  const float sampled_solid_angle = random.x * solid_angle;

  const vec3 r = add_vector(
    scale_vector(v0, G0 * cosf(0.5f * sampled_solid_angle) - G1 * sinf(0.5f * sampled_solid_angle)),
    scale_vector(v2, G2 * sinf(0.5f * sampled_solid_angle)));

  const vec3 v2_t = sub_vector(scale_vector(r, 2.0f * dot_product(v0, r) / dot_product(r, r)), v0);

  const float s2 = dot_product(v1, v2_t);
  const float s  = (1.0f - random.y) + random.y * s2;
  const float t  = sqrtf(fmaxf((1.0f - s * s) / (1.0f - s2 * s2), 0.0f));

  ray = normalize_vector(add_vector(scale_vector(v1, s - t * s2), scale_vector(v2_t, t)));

  if (is_non_finite(ray.x) || is_non_finite(ray.y) || is_non_finite(ray.z)) {
    solid_angle = 0.0f;
    dist        = FLT_MAX;
    return;
  }

  float2 coords;
  dist = light_triangle_intersection_uv(triangle, origin, ray, coords);

  // Our ray does not actually hit the light, abort.
  if (dist == FLT_MAX) {
    solid_angle = 0.0f;
    dist        = 1.0f;
    return;
  }

  const UV vertex_texture  = uv_unpack(packed_light_data.x);
  const UV vertex1_texture = uv_unpack(packed_light_data.y);
  const UV vertex2_texture = uv_unpack(packed_light_data.z);
  triangle.tex_coords      = lerp_uv(vertex_texture, vertex1_texture, vertex2_texture, coords);
}

__device__ void light_load_sample_finalize_bridges(
  TriangleLight& triangle, const uint3 packed_light_data, const vec3 origin, const float2 random, vec3& ray, float& dist, float& area) {
  const float r1 = sqrtf(random.x);
  const float r2 = random.y;

  float2 uv;
  uv.x = 1.0f - r1;
  uv.y = r1 * r2;

  const vec3 point_on_light =
    add_vector(triangle.vertex, add_vector(scale_vector(triangle.edge1, uv.x), scale_vector(triangle.edge2, uv.y)));

  area = get_length(cross_product(triangle.edge1, triangle.edge2)) * 0.5f;

  ray = normalize_vector(sub_vector(point_on_light, origin));

  float2 coords;
  dist = light_triangle_intersection_uv(triangle, origin, ray, coords);

  // Our ray does not actually hit the light, abort.
  if (dist == FLT_MAX) {
    area = 0.0f;
    dist = 1.0f;
    return;
  }

  const UV vertex_texture  = uv_unpack(packed_light_data.x);
  const UV vertex1_texture = uv_unpack(packed_light_data.y);
  const UV vertex2_texture = uv_unpack(packed_light_data.z);
  triangle.tex_coords      = lerp_uv(vertex_texture, vertex1_texture, vertex2_texture, coords);
}

__device__ RGBF light_get_color(const TriangleLight triangle) {
  RGBF color = splat_color(0.0f);

  const DeviceMaterial mat = load_material(device.ptrs.materials, triangle.material_id);

  if (mat.luminance_tex != TEXTURE_NONE) {
    const float4 emission = texture_load(load_texture_object(mat.luminance_tex), triangle.tex_coords);

    color = scale_color(get_color(emission.x, emission.y, emission.z), mat.emission_scale * emission.w);
  }
  else {
    color = mat.emission;
  }

  if (color_importance(color) > 0.0f) {
    float alpha;
    if (mat.albedo_tex != TEXTURE_NONE) {
      alpha = texture_load(load_texture_object(mat.albedo_tex), triangle.tex_coords).w;
    }
    else {
      alpha = mat.albedo.a;
    }

    color = scale_color(color, alpha);
  }

  return color;
}

#ifdef VOLUME_KERNEL
#if 0
__device__ float light_tree_child_importance(
  const float transmittance_importance, const vec3 origin, const vec3 ray, const DeviceLightTreeNode node, const vec3 exp,
  const float exp_c, const uint32_t i) {
  const bool lower_data = (i < 4);
  const uint32_t shift  = (lower_data ? i : (i - 4)) << 3;

  const uint32_t rel_energy = lower_data ? node.rel_energy[0] : node.rel_energy[1];

  vec3 point;
  const float energy = (float) ((rel_energy >> shift) & 0xFF);

  if (energy == 0.0f)
    return 0.0f;

  const uint32_t rel_point_x = lower_data ? node.rel_point_x[0] : node.rel_point_x[1];
  const uint32_t rel_point_y = lower_data ? node.rel_point_y[0] : node.rel_point_y[1];
  const uint32_t rel_point_z = lower_data ? node.rel_point_z[0] : node.rel_point_z[1];

  point = get_vector((rel_point_x >> shift) & 0xFF, (rel_point_y >> shift) & 0xFF, (rel_point_z >> shift) & 0xFF);
  point = mul_vector(point, exp);
  point = add_vector(point, node.base_point);

  const vec3 diff = sub_vector(point, origin);

  // Compute the point along our ray that is closest to the child point.
  const float t            = fmaxf(dot_product(diff, ray), 0.0f);
  const vec3 closest_point = add_vector(origin, scale_vector(ray, t));

  const float dist = sqrtf(dot_product(diff, diff));

  const vec3 shift_vector = normalize_vector(sub_vector(closest_point, point));

  const uint32_t confidence_light = lower_data ? node.confidence_light[0] : node.confidence_light[1];

  float confidence;
  confidence = (confidence_light >> (shift + 2)) & 0x3F;
  confidence = confidence * exp_c;

  const float dist_clamped = fmaxf(dist, confidence);

  // We shift the center of the child towards and along the ray based on the confidence.
  const vec3 reference_point = add_vector(scale_vector(add_vector(shift_vector, ray), confidence), point);

  const float angle_term = (1.0f + dot_product(ray, normalize_vector(sub_vector(reference_point, origin))));

  return energy * angle_term / dist_clamped;
}

__device__ uint32_t light_tree_traverse(const VolumeDescriptor volume, const vec3 origin, const vec3 ray, float& random, float& pdf) {
  pdf = 1.0f;

  DeviceLightTreeNode node = load_light_tree_node(0);

  const float transmittance_importance = color_importance(add_color(volume.scattering, volume.absorption));

  uint32_t subset_ptr = 0xFFFFFFFFu;

  random = random_saturate(random);

  while (subset_ptr == 0xFFFFFFFFu) {
    const vec3 exp    = get_vector(exp2f(node.exp_x), exp2f(node.exp_y), exp2f(node.exp_z));
    const float exp_c = exp2f(node.exp_confidence);

    float importance[8];

    importance[0] = light_tree_child_importance(transmittance_importance, origin, ray, node, exp, exp_c, 0);
    importance[1] = light_tree_child_importance(transmittance_importance, origin, ray, node, exp, exp_c, 1);
    importance[2] = light_tree_child_importance(transmittance_importance, origin, ray, node, exp, exp_c, 2);
    importance[3] = light_tree_child_importance(transmittance_importance, origin, ray, node, exp, exp_c, 3);
    importance[4] = light_tree_child_importance(transmittance_importance, origin, ray, node, exp, exp_c, 4);
    importance[5] = light_tree_child_importance(transmittance_importance, origin, ray, node, exp, exp_c, 5);
    importance[6] = light_tree_child_importance(transmittance_importance, origin, ray, node, exp, exp_c, 6);
    importance[7] = light_tree_child_importance(transmittance_importance, origin, ray, node, exp, exp_c, 7);

    float sum_importance = 0.0f;
    for (uint32_t i = 0; i < 8; i++) {
      sum_importance += importance[i];
    }

    float accumulated_importance = 0.0f;

    uint32_t selected_child             = 0xFFFFFFFF;
    uint32_t selected_child_light_ptr   = 0;
    uint32_t selected_child_light_count = 0;
    uint32_t sum_lights                 = 0;
    float selected_importance           = 0.0f;
    float random_shift                  = 0.0f;

    random *= sum_importance;

    for (uint32_t i = 0; i < 8; i++) {
      const float child_importance = importance[i];
      accumulated_importance += child_importance;

      const bool lower_data                 = (i < 4);
      const uint32_t child_light_count_data = lower_data ? node.confidence_light[0] : node.confidence_light[1];
      const uint32_t shift                  = (lower_data ? i : (i - 4)) << 3;

      uint32_t child_light_count = (child_light_count_data >> shift) & 0x3;
      sum_lights += child_light_count;

      if (accumulated_importance > random) {
        selected_child             = i;
        selected_child_light_count = child_light_count;
        selected_child_light_ptr   = sum_lights - child_light_count;
        selected_importance        = child_importance;

        random_shift = accumulated_importance - child_importance;

        // No control flow, we always loop over all children.
        accumulated_importance = -FLT_MAX;
      }
    }

    if (selected_child == 0xFFFFFFFF) {
      subset_ptr = 0;
      break;
    }

    pdf *= selected_importance / sum_importance;

    // Rescale random number
    random = random_saturate((random - random_shift) / selected_importance);

    if (selected_child_light_count > 0) {
      subset_ptr = node.light_ptr + selected_child_light_ptr;
      break;
    }

    node = load_light_tree_node(node.child_ptr + selected_child);
  }

  return subset_ptr;
}
#endif

// TODO: Support light trees for volumes.
__device__ TriangleHandle
  light_tree_query(const VolumeDescriptor volume, const vec3 origin, const vec3 ray, float random, float& pdf, DeviceTransform& trans) {
  pdf = 1.0f;

#if 0
  const uint32_t light_tree_handle_key = light_tree_traverse(volume, origin, ray, random, pdf);
#else
  const uint32_t light_tree_handle_key = 0;
#endif

  const TriangleHandle handle = device.ptrs.light_tree_tri_handle_map[light_tree_handle_key];

  trans = load_transform(handle.instance_id);

  return handle;
}

#else /* VOLUME_KERNEL */

__device__ void light_tree_child_importance(
  const LightTreeRuntimeData data, const vec3 position, const vec3 normal, const DeviceLightTreeNode node, const vec3 exp,
  const float exp_v, float importance[8], float splitting_prob[8], const uint32_t i, const uint32_t child_light_id, bool& is_leaf) {
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
    leaf_data = device.ptrs.light_tree_leafs[node.child_ptr + child_light_id];
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
  // leafs because we will resample later based on the importance.
  importance[i]     = fmaxf((is_leaf) ? approx_importance : approx_importance * (1.0f - uncertainty), 0.0f);
  splitting_prob[i] = uncertainty;
}

__device__ void light_tree_reservoir_add_light(uint32_t light_id, float importance, float pdf, LightTreeReservoir& reservoir) {
  const float weight = importance / pdf;

  reservoir.sum_weight += weight;

  const float resampling_probability = weight / reservoir.sum_weight;

  if (reservoir.random <= resampling_probability) {
    reservoir.light_id   = light_id;
    reservoir.target_pdf = importance;

    reservoir.random = reservoir.random / resampling_probability;
  }
  else {
    reservoir.random = (reservoir.random - resampling_probability) / (1.0f - resampling_probability);
  }
}

__device__ void light_tree_traverse(
  const LightTreeRuntimeData data, const vec3 position, const vec3 normal, const ushort2 pixel, float2 random,
  LightTreeStackEntry stack[LIGHT_TREE_STACK_SIZE], uint32_t& stack_ptr, LightTreeReservoir& reservoir) {
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

        // Leafs are always send to the reservoir, we don't select them
        if (is_leaf == false) {
          sum_importance += importance[i];
          split_probability[i] *= parent_split;
        }
      }

      float accumulated_importance = 0.0f;

      uint32_t selected_child     = 0xFFFFFFFF;
      bool selected_child_is_leaf = false;
      float selected_importance   = 0.0f;
      float selected_split_prob   = 0.0f;
      float random_shift          = 0.0f;

      const float importance_target = random.x * sum_importance;

      child_lights = 0;

#pragma unroll
      for (uint32_t i = 0; i < 8; i++) {
        const float child_importance = importance[i];

        const bool lower_data             = (i < 4);
        const uint32_t variance_leaf_data = lower_data ? node.rel_variance_leaf[0] : node.rel_variance_leaf[1];
        const uint32_t shift              = (lower_data ? i : (i - 4)) << 3;

        const bool is_leaf = ((variance_leaf_data >> shift) & 0x1) != 0;

        if (is_leaf) {
          const float leaf_probability = T * (1.0f - parent_split) + parent_split;

          // if (is_center_pixel(pixel) && IS_PRIMARY_RAY) {
          // printf("RESERVOIR: %u %f %f\n", node.light_ptr + current_child_light_offset, child_importance, leaf_probability);
          //}

          light_tree_reservoir_add_light(node.light_ptr + child_lights, child_importance, leaf_probability, reservoir);
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

      // This can only happen if all children were leafs
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

#if 0
__device__ float light_tree_traverse_pdf(
  const LightTreeRuntimeData data, const vec3 position, const vec3 normal, const uint32_t primitive_id) {
  float pdf = 1.0f;

  const uint2 light_paths = __ldg(device.ptrs.light_tree_paths + primitive_id);

  uint32_t current_light_path = light_paths.x;
  uint32_t current_depth      = 0;

  DeviceLightTreeNode node = load_light_tree_node(0);

  while (true) {
    const vec3 exp    = get_vector(exp2f(node.exp_x), exp2f(node.exp_y), exp2f(node.exp_z));
    const float exp_v = exp2f(node.exp_variance);

    float importance[8];
    importance[0] = light_tree_child_importance(data, position, normal, node, exp, exp_v, 0);
    importance[1] = light_tree_child_importance(data, position, normal, node, exp, exp_v, 1);
    importance[2] = light_tree_child_importance(data, position, normal, node, exp, exp_v, 2);
    importance[3] = light_tree_child_importance(data, position, normal, node, exp, exp_v, 3);
    importance[4] = light_tree_child_importance(data, position, normal, node, exp, exp_v, 4);
    importance[5] = light_tree_child_importance(data, position, normal, node, exp, exp_v, 5);
    importance[6] = light_tree_child_importance(data, position, normal, node, exp, exp_v, 6);
    importance[7] = light_tree_child_importance(data, position, normal, node, exp, exp_v, 7);

    float sum_importance = 0.0f;
    for (uint32_t i = 0; i < 8; i++) {
      sum_importance += importance[i];
    }

    const float one_over_sum = 1.0f / sum_importance;

    uint32_t selected_child     = 0xFFFFFFFF;
    bool selected_child_is_leaf = false;
    float child_pdf             = 0.0f;

    for (uint32_t i = 0; i < 8; i++) {
      const float child_importance = importance[i];

      if ((current_light_path & 0x7) == i) {
        selected_child         = i;
        selected_child_is_leaf = ((i < 4) ? (node.rel_variance_leaf[0] >> (i * 8)) : (node.rel_variance_leaf[1] >> ((i - 4) * 8))) & 0x1;
        child_pdf              = child_importance * one_over_sum;
      }
    }

    if (selected_child == 0xFFFFFFFF) {
      break;
    }

    pdf *= child_pdf;

    if (selected_child_is_leaf) {
      break;
    }

    current_light_path = current_light_path >> 3;
    current_depth++;

    if (current_depth == 10) {
      current_light_path = light_paths.y;
    }

    node = load_light_tree_node(node.child_ptr + selected_child);
  }

  return pdf;
}
#endif

__device__ void light_tree_query(const GBufferData data, const float2 random, const ushort2 pixel, LightTreeReservoir& reservoir) {
  const LightTreeRuntimeData runtime_data = light_sg_prepare(data);

  LightTreeStackEntry stack[LIGHT_TREE_STACK_SIZE];
  uint32_t stack_ptr = 0;

  LightTreeStackEntry root;
  root.id           = 0;
  root.T            = light_tree_pack_probability(1.0f);
  root.parent_split = light_tree_pack_probability(1.0f);

  LIGHT_TREE_STACK_PUSH(stack, stack_ptr, root);

  light_tree_traverse(runtime_data, data.position, data.normal, pixel, random, stack, stack_ptr, reservoir);
}

__device__ TriangleHandle light_tree_get_light(const uint32_t id, DeviceTransform& transform) {
  const TriangleHandle handle = device.ptrs.light_tree_tri_handle_map[id];

  transform = load_transform(handle.instance_id);

  return handle;
}

#if 0
__device__ float light_tree_query_pdf(const GBufferData data, const uint32_t light_tree_handle_key) {
  const LightTreeRuntimeData runtime_data = light_sg_prepare(data);

  return light_tree_traverse_pdf(runtime_data, data.position, data.normal, light_tree_handle_key);
}
#endif

#endif /* !VOLUME_KERNEL */

#else /* SHADING_KERNEL */

////////////////////////////////////////////////////////////////////
// Light Processing
////////////////////////////////////////////////////////////////////

__device__ float lights_integrate_emission(const DeviceMaterial material, const UV vertex, const UV edge1, const UV edge2) {
  const DeviceTextureObject tex = load_texture_object(material.luminance_tex);

  // Super crude way of determining the number of texel fetches I will need. If performance of this becomes an issue
  // then I will have to rethink this here.
  const float texel_steps_u = fmaxf(fabsf(edge1.u), fabsf(edge2.u)) * tex.width;
  const float texel_steps_v = fmaxf(fabsf(edge1.v), fabsf(edge2.v)) * tex.height;

  const float steps = ceilf(fmaxf(texel_steps_u, texel_steps_v));

  const float step_size = 4.0f / steps;

  RGBF accumulator  = get_color(0.0f, 0.0f, 0.0f);
  float texel_count = 0.0f;

  for (float a = 0.0f; a < 1.0f; a += step_size) {
    for (float b = 0.0f; a + b < 1.0f; b += step_size) {
      const float u = vertex.u + a * edge1.u + b * edge2.u;
      const float v = vertex.v + a * edge1.v + b * edge2.v;

      const float4 texel = texture_load(tex, get_uv(u, v));

      const RGBF color = scale_color(get_color(texel.x, texel.y, texel.z), texel.w);

      accumulator = add_color(accumulator, color);
      texel_count += 1.0f;
    }
  }

  return color_importance(accumulator) / texel_count;
}

LUMINARY_KERNEL void light_compute_intensity(const KernelArgsLightComputeIntensity args) {
  const uint32_t light = threadIdx.x + blockIdx.x * blockDim.x;

  if (light >= args.lights_count)
    return;

  const uint32_t mesh_id     = args.mesh_ids[light];
  const uint32_t triangle_id = args.triangle_ids[light];

  const DeviceTriangle* tri_ptr = device.ptrs.triangles[mesh_id];
  const uint32_t triangle_count = device.ptrs.triangle_counts[mesh_id];

  const float4 t2 = __ldg((float4*) triangle_get_entry_address(tri_ptr, 2, 0, triangle_id, triangle_count));
  const float4 t3 = __ldg((float4*) triangle_get_entry_address(tri_ptr, 3, 0, triangle_id, triangle_count));

  const UV vertex_texture  = uv_unpack(__float_as_uint(t2.y));
  const UV vertex1_texture = uv_unpack(__float_as_uint(t2.z));
  const UV vertex2_texture = uv_unpack(__float_as_uint(t2.w));

  const UV edge1_texture = uv_sub(vertex1_texture, vertex_texture);
  const UV edge2_texture = uv_sub(vertex2_texture, vertex_texture);

  const uint16_t material_id    = __float_as_uint(t3.w) & 0xFFFF;
  const DeviceMaterial material = load_material(device.ptrs.materials, material_id);

  args.dst_average_intensities[light] = lights_integrate_emission(material, vertex_texture, edge1_texture, edge2_texture);
}

#endif /* !SHADING_KERNEL */

#endif /* CU_LIGHT_H */
