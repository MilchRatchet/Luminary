#ifndef CU_GEOMETRY_UTILS_H
#define CU_GEOMETRY_UTILS_H

#include "light_triangle.cuh"
#include "material.cuh"
#include "math.cuh"
#include "medium_stack.cuh"
#include "memory.cuh"
#include "mis.cuh"
#include "texture_utils.cuh"
#include "utils.cuh"

LUMINARY_FUNCTION vec3 geometry_compute_normal(
  vec3 v_normal, vec3 e1_normal, vec3 e2_normal, vec3 ray, UV e1_tex, UV e2_tex, uint16_t normal_tex, float2 coords, UV tex_coords,
  bool normal_map_is_compressed, vec3& face_normal, bool& is_inside) {
  is_inside = dot_product(face_normal, ray) > 0.0f;

  // TODO: Why do I not have a neg_vector function?
  // Convention is for the face normal to look towards the origin
  if (is_inside)
    face_normal = scale_vector(face_normal, -1.0f);

  vec3 normal = lerp_normals(v_normal, e1_normal, e2_normal, coords, face_normal);

  if (normal_tex != TEXTURE_NONE) {
    const DeviceTextureObject tex = load_texture_object(normal_tex);

    // TODO: Flip V based on a material flag that specifies if the texture is OpenGL or DirectX format.
    TextureLoadArgs tex_load_args = texture_get_default_args();
    tex_load_args.flip_v          = true;
    tex_load_args.apply_gamma     = false;
    tex_load_args.default_result  = make_float4(0.0f, 0.0f, 1.0f, 0.0f);

    const float4 normal_f = texture_load(tex, tex_coords, tex_load_args);

    vec3 map_normal = get_vector(normal_f.x, normal_f.y, normal_f.z);

    // Normal maps can be encoded in [-1,1]^3 or [0,1]^3.
    if (normal_map_is_compressed && texture_is_valid(tex)) {
      map_normal = scale_vector(map_normal, 2.0f);
      map_normal = sub_vector(map_normal, get_vector(1.0f, 1.0f, 1.0f));
    }

    map_normal = normalize_vector(map_normal);

    const Quaternion q = quaternion_rotation_to_z_canonical(normal);

    normal = quaternion_apply(quaternion_inverse(q), map_normal);
  }

  return normal_adaptation_apply(scale_vector(ray, -1.0f), normal, face_normal);
}

LUMINARY_FUNCTION MaterialContextGeometry
  geometry_get_context(const DeviceTask& task, const DeviceTaskTrace& trace, const DeviceTaskMediumStack& medium) {
  const uint32_t mesh_id      = mesh_id_load(trace.handle.instance_id);
  const DeviceTransform trans = load_transform(trace.handle.instance_id);

  const DeviceTriangleVertex* vertex_ptr = (const DeviceTriangleVertex*) __ldg((uint64_t*) (device.ptrs.vertices + mesh_id));
  const DeviceTriangleTexture* tri_ptr   = (const DeviceTriangleTexture*) __ldg((uint64_t*) (device.ptrs.texture_triangles + mesh_id));

  const float4 v0 = __ldg((float4*) (vertex_ptr + trace.handle.tri_id * 3 + 0));
  const float4 v1 = __ldg((float4*) (vertex_ptr + trace.handle.tri_id * 3 + 1));
  const float4 v2 = __ldg((float4*) (vertex_ptr + trace.handle.tri_id * 3 + 2));

  const float4 tri = __ldg((float4*) (tri_ptr + trace.handle.tri_id));

  vec3 position  = transform_apply_inv(trans, task.origin);
  const vec3 ray = transform_apply_rotation_inv(trans, task.ray);

  const vec3 vertex  = get_vector(v0.x, v0.y, v0.z);
  const vec3 vertex1 = get_vector(v1.x, v1.y, v1.z);
  const vec3 vertex2 = get_vector(v2.x, v2.y, v2.z);

  const vec3 edge1 = sub_vector(vertex1, vertex);
  const vec3 edge2 = sub_vector(vertex2, vertex);

  vec3 face_normal = normalize_vector(cross_product(edge1, edge2));

  const float2 coords = get_coordinates_in_triangle(vertex, edge1, edge2, position);

  position = add_vector(vertex, add_vector(scale_vector(edge1, coords.x), scale_vector(edge2, coords.y)));
  position = transform_apply(trans, position);

  const UV vertex_texture  = uv_unpack(__float_as_uint(tri.x));
  const UV vertex1_texture = uv_unpack(__float_as_uint(tri.y));
  const UV vertex2_texture = uv_unpack(__float_as_uint(tri.z));

  const UV tex_coords = lerp_uv(vertex_texture, vertex1_texture, vertex2_texture, coords);

  const uint16_t material_id = __float_as_uint(tri.w) & 0xFFFF;
  const DeviceMaterial mat   = load_material(device.ptrs.materials, material_id);

  const vec3 vertex_normal  = normal_unpack(__float_as_uint(v0.w));
  const vec3 vertex1_normal = normal_unpack(__float_as_uint(v1.w));
  const vec3 vertex2_normal = normal_unpack(__float_as_uint(v2.w));

  const vec3 edge1_normal = sub_vector(vertex1_normal, vertex_normal);
  const vec3 edge2_normal = sub_vector(vertex2_normal, vertex_normal);

  const bool normal_map_is_compressed = (mat.flags & DEVICE_MATERIAL_FLAG_NORMAL_MAP_COMPRESSED) != 0;

  bool is_inside;
  const vec3 normal = geometry_compute_normal(
    vertex_normal, edge1_normal, edge2_normal, ray, uv_sub(vertex_texture, vertex1_texture), uv_sub(vertex_texture, vertex2_texture),
    mat.normal_tex, coords, tex_coords, normal_map_is_compressed, face_normal, is_inside);

  RGBAF albedo = mat.albedo;
  if (mat.albedo_tex != TEXTURE_NONE) {
    const DeviceTextureObject tex = load_texture_object(mat.albedo_tex);

    TextureLoadArgs tex_load_args = texture_get_default_args();
    tex_load_args.default_result  = make_float4(0.9f, 0.9f, 0.9f, 1.0f);

    const float4 albedo_f = texture_load(tex, tex_coords, tex_load_args);
    albedo.r              = albedo_f.x;
    albedo.g              = albedo_f.y;
    albedo.b              = albedo_f.z;
    albedo.a              = albedo_f.w;
  }

  const bool emissive_side    = (is_inside == false) || (mat.flags & DEVICE_MATERIAL_FLAG_BIDIRECTIONAL_EMISSION);
  const bool has_emission     = (mat.flags & DEVICE_MATERIAL_FLAG_EMISSION) && emissive_side;
  const bool include_emission = has_emission && ((task.state & STATE_FLAG_ALLOW_EMISSION) != 0);

  RGBF emission = get_color(0.0f, 0.0f, 0.0f);
  if (include_emission) {
    emission = mat.emission;

    if (include_emission && (mat.luminance_tex != TEXTURE_NONE)) {
      const DeviceTextureObject tex = load_texture_object(mat.luminance_tex);

      const float4 luminance_f = texture_load(tex, tex_coords);

      emission = get_color(luminance_f.x, luminance_f.y, luminance_f.z);
      emission = scale_color(emission, albedo.a * mat.emission_scale);
    }
  }

  float roughness = mat.roughness;
  if (mat.roughness_tex != TEXTURE_NONE) {
    const DeviceTextureObject tex = load_texture_object(mat.roughness_tex);

    TextureLoadArgs tex_load_args = texture_get_default_args();
    tex_load_args.default_result  = make_float4(0.5f, 0.0f, 0.0f, 0.0f);

    const float4 material_f = texture_load(tex, tex_coords, tex_load_args);

    roughness = material_f.x;
  }

  if (mat.flags & DEVICE_MATERIAL_FLAG_ROUGHNESS_AS_SMOOTHNESS) {
    roughness = 1.0f - roughness;
  }

  // We have to clamp due to numerical precision issues in the microfacet models.
  roughness = fmaxf(roughness, BSDF_ROUGHNESS_CLAMP);

  // We clamp the roughness to avoid caustics which would never clean up.
  if ((task.state & STATE_FLAG_DELTA_PATH) == 0) {
    roughness = fmaxf(roughness, mat.roughness_clamp);
  }

  MaterialFlags flags = (mat.flags & DEVICE_MATERIAL_BASE_SUBSTRATE_MASK);

  if (mat.metallic_tex != TEXTURE_NONE) {
    // TODO: Stochastic filtering of metallic texture.
  }
  else if (mat.flags & DEVICE_MATERIAL_FLAG_METALLIC) {
    flags |= MATERIAL_FLAG_METALLIC;
  }

  if (mat.flags & DEVICE_MATERIAL_FLAG_COLORED_TRANSPARENCY) {
    flags |= MATERIAL_FLAG_COLORED_TRANSPARENCY;
  }

  if (is_inside) {
    flags |= MATERIAL_FLAG_REFRACTION_IS_INSIDE;
  }

  const bool refraction_is_inside = (flags & MATERIAL_FLAG_REFRACTION_IS_INSIDE) != 0;

  const float other_ior    = medium_stack_ior_peek(medium, refraction_is_inside);
  const uint16_t volume_id = medium_stack_volume_peek(medium, false);

  const float ior_in  = (refraction_is_inside) ? mat.refraction_index : other_ior;
  const float ior_out = (refraction_is_inside) ? other_ior : mat.refraction_index;

  // If we have a translucent substrate and the IOR change is within some small threshold, treat the material as fully transparent.
  if (MATERIAL_IS_SUBSTRATE_TRANSLUCENT(flags) && (fabsf(1.0f - ior_in / ior_out) < 1e-4f)) {
    // Fudge the albedo to be a blend between the transparent color and the refraction color.
    if ((flags & MATERIAL_FLAG_COLORED_TRANSPARENCY) == 0.0f) {
      albedo.r = lerp(1.0f, albedo.r, albedo.a);
      albedo.g = lerp(1.0f, albedo.g, albedo.a);
      albedo.b = lerp(1.0f, albedo.b, albedo.a);
    }

    albedo.a = 0.0f;
    flags |= MATERIAL_FLAG_COLORED_TRANSPARENCY;
  }

  MaterialContextGeometry ctx;
  ctx.instance_id = trace.handle.instance_id;
  ctx.tri_id      = trace.handle.tri_id;
  ctx.normal      = transform_apply_rotation(trans, normal);
  ctx.face_normal = normal_pack(face_normal);
  ctx.position    = position;
  ctx.V           = scale_vector(task.ray, -1.0f);
  ctx.state       = task.state;
  ctx.volume_type = VolumeType(volume_id);

  ctx.params.flags = flags;

  material_set_color<MATERIAL_GEOMETRY_PARAM_ALBEDO>(ctx.params, opaque_color(albedo));
  material_set_float<MATERIAL_GEOMETRY_PARAM_OPACITY>(ctx.params, albedo.a);
  material_set_float<MATERIAL_GEOMETRY_PARAM_ROUGHNESS>(ctx.params, roughness);
  material_set_color<MATERIAL_GEOMETRY_PARAM_EMISSION>(ctx.params, emission);
  material_set_float<MATERIAL_GEOMETRY_PARAM_IOR>(ctx.params, ior_in / ior_out);

  return ctx;
}

#endif /* CU_GEOMETRY_UTILS_H */
