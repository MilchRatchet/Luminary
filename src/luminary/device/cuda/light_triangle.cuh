#ifndef CU_LUMINARY_LIGHT_TRIANGLE_H
#define CU_LUMINARY_LIGHT_TRIANGLE_H

#include "light_common.cuh"
#include "math.cuh"
#include "memory.cuh"
#include "texture_utils.cuh"
#include "utils.cuh"

__device__ float light_triangle_intersection_uv_generic(
  const vec3 vertex, const vec3 edge1, const vec3 edge2, const vec3 origin, const vec3 ray, float2& coords) {
  const vec3 h  = cross_product(ray, edge2);
  const float a = dot_product(edge1, h);

  const float f = 1.0f / a;
  const vec3 s  = sub_vector(origin, vertex);
  const float u = f * dot_product(s, h);

  const vec3 q  = cross_product(s, edge1);
  const float v = f * dot_product(ray, q);

  coords = make_float2(u, v);

  //  The third check is inverted to catch NaNs since NaNs always return false, the not will turn it into a true
  if (v < 0.0f || u < 0.0f || !(u + v <= 1.0f))
    return FLT_MAX;

  const float t = f * dot_product(edge2, q);

  return __fslctf(t, FLT_MAX, t);
}

__device__ float light_triangle_intersection_uv(const TriangleLight triangle, const vec3 origin, const vec3 ray, float2& coords) {
  return light_triangle_intersection_uv_generic(triangle.vertex, triangle.edge1, triangle.edge2, origin, ray, coords);
}

__device__ TriangleLight
  light_triangle_load(const TriangleHandle handle, const vec3 origin, const vec3 ray, const DeviceTransform trans, float& dist) {
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

__device__ TriangleLight light_triangle_sample_init(const TriangleHandle handle, const DeviceTransform trans, uint3& packed_light_data) {
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

__device__ bool light_triangle_sample_finalize_dist_and_uvs(
  TriangleLight& triangle, const uint3 packed_light_data, const vec3 origin, const vec3 ray, float& dist) {
  float2 coords;
  dist = light_triangle_intersection_uv(triangle, origin, ray, coords);

  // Our ray does not actually hit the light, abort.
  if (dist == FLT_MAX) {
    return false;
  }

  const UV vertex_texture  = uv_unpack(packed_light_data.x);
  const UV vertex1_texture = uv_unpack(packed_light_data.y);
  const UV vertex2_texture = uv_unpack(packed_light_data.z);
  triangle.tex_coords      = lerp_uv(vertex_texture, vertex1_texture, vertex2_texture, coords);

  return true;
}

__device__ float light_triangle_get_solid_angle_generic(const vec3 vertex, const vec3 edge1, const vec3 edge2, const vec3 origin) {
  const vec3 v0 = normalize_vector(sub_vector(vertex, origin));
  const vec3 v1 = normalize_vector(sub_vector(add_vector(vertex, edge1), origin));
  const vec3 v2 = normalize_vector(sub_vector(add_vector(vertex, edge2), origin));

  const float G0 = fabsf(dot_product(cross_product(v0, v1), v2));
  const float G1 = dot_product(v0, v2) + dot_product(v1, v2);
  const float G2 = 1.0f + dot_product(v0, v1);

  return 2.0f * atan2f(G0, G1 + G2);
}

__device__ float light_triangle_get_solid_angle(const TriangleLight triangle, const vec3 origin) {
  return light_triangle_get_solid_angle_generic(triangle.vertex, triangle.edge1, triangle.edge2, origin);
}

__device__ bool light_triangle_sample_solid_angle(
  const vec3 origin, const vec3 vertex, const vec3 edge1, const vec3 edge2, const float2 random, vec3& ray, float& solid_angle) {
  const vec3 v0 = normalize_vector(sub_vector(vertex, origin));
  const vec3 v1 = normalize_vector(sub_vector(add_vector(vertex, edge1), origin));
  const vec3 v2 = normalize_vector(sub_vector(add_vector(vertex, edge2), origin));

  const float G0 = fabsf(dot_product(cross_product(v0, v1), v2));
  const float G1 = dot_product(v0, v2) + dot_product(v1, v2);
  const float G2 = 1.0f + dot_product(v0, v1);

  solid_angle = 2.0f * atan2f(G0, G1 + G2);

  if (is_non_finite(solid_angle) || solid_angle < 1e-7f) {
    return false;
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
    return false;
  }

  return true;
}

/*
 * Robust solid angle sampling method from
 * C. Peters, "BRDF Importance Sampling for Linear Lights", Computer Graphics Forum (Proc. HPG) 40, 8, 2021.
 */
__device__ bool light_triangle_sample_finalize(
  TriangleLight& triangle, const uint3 packed_light_data, const vec3 origin, const float2 random, vec3& ray, float& dist,
  float& solid_angle) {
  bool success = true;

  success &= light_triangle_sample_solid_angle(origin, triangle.vertex, triangle.edge1, triangle.edge2, random, ray, solid_angle);
  success &= light_triangle_sample_finalize_dist_and_uvs(triangle, packed_light_data, origin, ray, dist);

  return success;
}

__device__ bool light_triangle_sample_microtriangle_finalize(
  TriangleLight& triangle, const float2 bary0, const float2 bary1, const float2 bary2, const uint3 packed_light_data, const vec3 origin,
  const float2 random, vec3& ray, float& dist, float& solid_angle, float2& coords) {
  const vec3 v0 = add_vector(triangle.vertex, add_vector(scale_vector(triangle.edge1, bary0.x), scale_vector(triangle.edge2, bary0.y)));
  const vec3 v1 = add_vector(triangle.vertex, add_vector(scale_vector(triangle.edge1, bary1.x), scale_vector(triangle.edge2, bary1.y)));
  const vec3 v2 = add_vector(triangle.vertex, add_vector(scale_vector(triangle.edge1, bary2.x), scale_vector(triangle.edge2, bary2.y)));

  const vec3 edge1 = sub_vector(v1, v0);
  const vec3 edge2 = sub_vector(v2, v0);

  bool success = true;

  success &= light_triangle_sample_solid_angle(origin, v0, edge1, edge2, random, ray, solid_angle);

  if (success) {
    dist = light_triangle_intersection_uv_generic(v0, edge1, edge2, origin, ray, coords);

    if (dist == FLT_MAX) {
      return false;
    }

    const UV vertex_texture  = uv_unpack(packed_light_data.x);
    const UV vertex1_texture = uv_unpack(packed_light_data.y);
    const UV vertex2_texture = uv_unpack(packed_light_data.z);

    const UV edge1_texture = uv_sub(vertex1_texture, vertex_texture);
    const UV edge2_texture = uv_sub(vertex2_texture, vertex_texture);

    const UV v0_texture = add_uv(vertex_texture, add_uv(uv_scale(edge1_texture, bary0.x), uv_scale(edge2_texture, bary0.y)));
    const UV v1_texture = add_uv(vertex_texture, add_uv(uv_scale(edge1_texture, bary1.x), uv_scale(edge2_texture, bary1.y)));
    const UV v2_texture = add_uv(vertex_texture, add_uv(uv_scale(edge1_texture, bary2.x), uv_scale(edge2_texture, bary2.y)));

    triangle.tex_coords = lerp_uv(v0_texture, v1_texture, v2_texture, coords);
  }

  return success;
}

__device__ vec3 light_triangle_sample_bridges(TriangleLight& triangle, const float2 random) {
  const float r1 = sqrtf(random.x);
  const float r2 = random.y;

  float2 uv;
  uv.x = 1.0f - r1;
  uv.y = r1 * r2;

  const vec3 point_on_light =
    add_vector(triangle.vertex, add_vector(scale_vector(triangle.edge1, uv.x), scale_vector(triangle.edge2, uv.y)));

  return point_on_light;
}

__device__ bool light_triangle_sample_finalize_bridges(
  TriangleLight& triangle, const uint3 packed_light_data, const vec3 origin, const vec3 point_on_light, vec3& ray, float& dist,
  float& area) {
  area = get_length(cross_product(triangle.edge1, triangle.edge2)) * 0.5f;

  ray = normalize_vector(sub_vector(point_on_light, origin));

  bool success = true;

  success &= light_triangle_sample_finalize_dist_and_uvs(triangle, packed_light_data, origin, ray, dist);

  return success;
}

__device__ RGBF light_get_color(TriangleLight& triangle) {
  RGBF color = splat_color(0.0f);

  // TODO: Implement a light weight DeviceLightMaterial type
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

#endif /* CU_LUMINARY_LIGHT_TRIANGLE_H */
