#ifndef CU_GEOMETRY_UTILS_H
#define CU_GEOMETRY_UTILS_H

#include "ior_stack.cuh"
#include "math.cuh"
#include "memory.cuh"
#include "texture_utils.cuh"
#include "utils.cuh"

__device__ vec3 geometry_compute_normal(
  vec3 v_normal, vec3 e1_normal, vec3 e2_normal, vec3 ray, vec3 e1, vec3 e2, UV e1_tex, UV e2_tex, uint16_t normal_tex, float2 coords,
  UV tex_coords, bool& is_inside) {
  vec3 face_normal = normalize_vector(cross_product(e1, e2));
  vec3 normal      = lerp_normals(v_normal, e1_normal, e2_normal, coords, face_normal);
  is_inside        = dot_product(face_normal, ray) > 0.0f;

  if (normal_tex != TEXTURE_NONE) {
    const float4 normal_f = texture_load(device.ptrs.textures[normal_tex], tex_coords);

    vec3 map_normal = get_vector(normal_f.x, normal_f.y, normal_f.z);

    map_normal = scale_vector(map_normal, 2.0f);
    map_normal = sub_vector(map_normal, get_vector(1.0f, 1.0f, 1.0f));

    Mat3x3 tangent_space = cotangent_frame(normal, e1, e2, e1_tex, e2_tex);

    normal = normalize_vector(transform_vec3(tangent_space, map_normal));

    if (dot_product(normal, ray) > 0.0f) {
      normal = scale_vector(normal, -1.0f);
    }
  }
  else {
    if (dot_product(face_normal, normal) < 0.0f) {
      normal = scale_vector(normal, -1.0f);
    }

    if (dot_product(face_normal, ray) > 0.0f) {
      face_normal = scale_vector(face_normal, -1.0f);
      normal      = scale_vector(normal, -1.0f);
    }

    /*
     * I came up with this quickly, problem is that we need to rotate the normal
     * towards the face normal with our ray is "behind" the shading normal
     */
    if (dot_product(normal, ray) > 0.0f) {
      const float a = sqrtf(1.0f - dot_product(face_normal, ray));
      const float b = dot_product(face_normal, normal);
      const float t = a / (a + b + eps);
      normal        = normalize_vector(add_vector(scale_vector(face_normal, t), scale_vector(normal, 1.0f - t)));
    }
  }

  return normal;
}

__device__ GBufferData geometry_generate_g_buffer(const DeviceTask task, const TriangleHandle triangle_handle, const int pixel) {
  const uint32_t mesh_id      = mesh_id_load(triangle_handle.instance_id);
  const DeviceTransform trans = load_transform(triangle_handle.instance_id);

  const DeviceTriangle* tri_ptr = device.ptrs.triangles[mesh_id];
  const uint32_t triangle_count = device.ptrs.triangle_counts[mesh_id];

  const float4 t0 = __ldg((float4*) triangle_get_entry_address(tri_ptr, 0, 0, triangle_handle.tri_id, triangle_count));
  const float4 t1 = __ldg((float4*) triangle_get_entry_address(tri_ptr, 1, 0, triangle_handle.tri_id, triangle_count));
  const float4 t2 = __ldg((float4*) triangle_get_entry_address(tri_ptr, 2, 0, triangle_handle.tri_id, triangle_count));
  const float4 t3 = __ldg((float4*) triangle_get_entry_address(tri_ptr, 3, 0, triangle_handle.tri_id, triangle_count));

  const vec3 vertex = transform_apply(trans, get_vector(t0.x, t0.y, t0.z));
  const vec3 edge1  = transform_apply(trans, get_vector(t0.w, t1.x, t1.y));
  const vec3 edge2  = transform_apply(trans, get_vector(t1.z, t1.w, t2.x));

  const float2 coords = get_coordinates_in_triangle(vertex, edge1, edge2, task.origin);

  const UV vertex_texture  = uv_unpack(__float_as_uint(t2.y));
  const UV vertex1_texture = uv_unpack(__float_as_uint(t2.z));
  const UV vertex2_texture = uv_unpack(__float_as_uint(t2.w));

  const UV tex_coords = lerp_uv(vertex_texture, vertex1_texture, vertex2_texture, coords);

  const uint16_t material_id = __float_as_uint(t3.w) & 0xFFFF;
  const DeviceMaterial mat   = load_material(device.ptrs.materials, material_id);

  const vec3 vertex_normal  = transform_apply_relative(trans, normal_unpack(__float_as_uint(t3.x)));
  const vec3 vertex1_normal = transform_apply_relative(trans, normal_unpack(__float_as_uint(t3.y)));
  const vec3 vertex2_normal = transform_apply_relative(trans, normal_unpack(__float_as_uint(t3.z)));

  const vec3 edge1_normal = sub_vector(vertex_normal, vertex1_normal);
  const vec3 edge2_normal = sub_vector(vertex_normal, vertex2_normal);

  bool is_inside;
  const vec3 normal = geometry_compute_normal(
    vertex_normal, edge1_normal, edge2_normal, task.ray, edge1, edge2, uv_sub(vertex_texture, vertex1_texture),
    uv_sub(vertex_texture, vertex2_texture), mat.normal_tex, coords, tex_coords, is_inside);

  RGBAF albedo = mat.albedo;
  if (mat.albedo_tex != TEXTURE_NONE) {
    const float4 albedo_f = texture_load(device.ptrs.textures[mat.albedo_tex], tex_coords);
    albedo.r              = albedo_f.x;
    albedo.g              = albedo_f.y;
    albedo.b              = albedo_f.z;
    albedo.a              = albedo_f.w;
  }

  const bool include_emission = (mat.flags & DEVICE_MATERIAL_FLAG_EMISSION) && (task.state & STATE_FLAG_CAMERA_DIRECTION);

  RGBF emission = (include_emission) ? mat.emission : get_color(0.0f, 0.0f, 0.0f);
  if (include_emission && (mat.luminance_tex != TEXTURE_NONE)) {
    const float4 luminance_f = texture_load(device.ptrs.textures[mat.luminance_tex], tex_coords);

    emission = get_color(luminance_f.x, luminance_f.y, luminance_f.z);
    emission = scale_color(emission, luminance_f.w * albedo.a);
  }

  float roughness = mat.roughness;
  float metallic  = mat.metallic;
  if (mat.material_tex != TEXTURE_NONE) {
    const float4 material_f = texture_load(device.ptrs.textures[mat.material_tex], tex_coords);

    roughness = material_f.x;
    metallic  = material_f.y;
  }

  // We clamp the roughness to avoid caustics which would never clean up.
  if (!(task.state & STATE_FLAG_DELTA_PATH)) {
    roughness = fmaxf(roughness, mat.roughness_clamp);
  }

  uint32_t flags = G_BUFFER_FLAG_USE_LIGHT_RAYS;

  if (is_inside) {
    flags |= G_BUFFER_FLAG_REFRACTION_IS_INSIDE;
  }

  const IORStackMethod ior_stack_method =
    (flags & G_BUFFER_FLAG_REFRACTION_IS_INSIDE) ? IOR_STACK_METHOD_PEEK_PREVIOUS : IOR_STACK_METHOD_PEEK_CURRENT;
  const float ray_ior = ior_stack_interact(mat.refraction_index, pixel, ior_stack_method);

  if (mat.flags & DEVICE_MATERIAL_FLAG_COLORED_TRANSPARENCY) {
    flags |= G_BUFFER_FLAG_COLORED_DIELECTRIC;
  }

  GBufferData data;
  data.instance_id = triangle_handle.instance_id;
  data.tri_id      = triangle_handle.tri_id;
  data.albedo      = albedo;
  data.emission    = emission;
  data.normal      = normal;
  data.position    = task.origin;
  data.V           = scale_vector(task.ray, -1.0f);
  data.roughness   = roughness;
  data.metallic    = metallic;
  data.state       = task.state;
  data.flags       = flags;
  data.ior_in      = (flags & G_BUFFER_FLAG_REFRACTION_IS_INSIDE) ? mat.refraction_index : ray_ior;
  data.ior_out     = (flags & G_BUFFER_FLAG_REFRACTION_IS_INSIDE) ? ray_ior : mat.refraction_index;

  return data;
}

#endif /* CU_GEOMETRY_UTILS_H */
