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
  RGBF color;
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
__device__ float light_tree_child_importance(
  const float transmittance_importance, const vec3 origin, const vec3 ray, const LightTreeNode8Packed node, const vec3 exp,
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

__device__ TriangleHandle
  light_tree_query(const VolumeDescriptor volume, const vec3 origin, const vec3 ray, float random, float& pdf, DeviceTransform& trans) {
  pdf = 1.0f;

  const uint32_t light_tree_handle_key = light_tree_traverse(volume, origin, ray, random, pdf);

  const TriangleHandle handle = device.ptrs.light_tree_tri_handle_map[light_tree_handle_key];

  trans = load_transform(handle.instance_id);

  return handle;
}

#else /* VOLUME_KERNEL */

__device__ float light_tree_child_importance(
  const vec3 position, const LightTreeNode8Packed node, const vec3 exp, const float exp_c, const uint32_t i) {
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

  const uint32_t confidence_light = lower_data ? node.confidence_light[0] : node.confidence_light[1];

  float confidence;
  confidence = (confidence_light >> (shift + 2)) & 0x3F;
  confidence = confidence * exp_c;

  const vec3 diff = sub_vector(point, position);

  return energy / fmaxf(dot_product(diff, diff), confidence * confidence);
}

__device__ uint32_t light_tree_traverse(const vec3 position, float& random, float& pdf) {
  DeviceLightTreeNode node = load_light_tree_node(0);

  uint32_t result_id = 0xFFFFFFFF;

  random = random_saturate(random);

  while (result_id == 0xFFFFFFFFu) {
    const vec3 exp    = get_vector(exp2f(node.exp_x), exp2f(node.exp_y), exp2f(node.exp_z));
    const float exp_c = exp2f(node.exp_confidence);

    float importance[8];

    importance[0] = light_tree_child_importance(position, node, exp, exp_c, 0);
    importance[1] = light_tree_child_importance(position, node, exp, exp_c, 1);
    importance[2] = light_tree_child_importance(position, node, exp, exp_c, 2);
    importance[3] = light_tree_child_importance(position, node, exp, exp_c, 3);
    importance[4] = light_tree_child_importance(position, node, exp, exp_c, 4);
    importance[5] = light_tree_child_importance(position, node, exp, exp_c, 5);
    importance[6] = light_tree_child_importance(position, node, exp, exp_c, 6);
    importance[7] = light_tree_child_importance(position, node, exp, exp_c, 7);

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
      result_id = 0;
      break;
    }

    pdf *= selected_importance / sum_importance;

    // Rescale random number
    random = random_saturate((random - random_shift) / selected_importance);

    if (selected_child_light_count > 0) {
      result_id = node.light_ptr + selected_child_light_ptr;
      break;
    }

    node = load_light_tree_node(node.child_ptr + selected_child);
  }

  return result_id;
}

__device__ float light_tree_traverse_pdf(const vec3 position, const uint32_t primitive_id) {
  float pdf = 1.0f;

  const uint2 light_paths = __ldg(device.ptrs.light_tree_paths + primitive_id);

  uint32_t current_light_path = light_paths.x;
  uint32_t current_depth      = 0;

  LightTreeNode8Packed node = load_light_tree_node(0);

  while (true) {
    const vec3 exp    = get_vector(exp2f(node.exp_x), exp2f(node.exp_y), exp2f(node.exp_z));
    const float exp_c = exp2f(node.exp_confidence);

    float importance[8];

    importance[0] = light_tree_child_importance(position, node, exp, exp_c, 0);
    importance[1] = light_tree_child_importance(position, node, exp, exp_c, 1);
    importance[2] = light_tree_child_importance(position, node, exp, exp_c, 2);
    importance[3] = light_tree_child_importance(position, node, exp, exp_c, 3);
    importance[4] = light_tree_child_importance(position, node, exp, exp_c, 4);
    importance[5] = light_tree_child_importance(position, node, exp, exp_c, 5);
    importance[6] = light_tree_child_importance(position, node, exp, exp_c, 6);
    importance[7] = light_tree_child_importance(position, node, exp, exp_c, 7);

    float sum_importance = 0.0f;
    for (uint32_t i = 0; i < 8; i++) {
      sum_importance += importance[i];
    }

    const float one_over_sum = 1.0f / sum_importance;

    uint32_t selected_child             = 0xFFFFFFFF;
    uint32_t selected_child_light_count = 0;
    float child_pdf                     = 0.0f;

    for (uint32_t i = 0; i < 8; i++) {
      const float child_importance = importance[i];

      if ((current_light_path & 0x7) == i) {
        selected_child             = i;
        selected_child_light_count = ((i < 4) ? (node.confidence_light[0] >> (i * 8)) : (node.confidence_light[1] >> ((i - 4) * 8))) & 0x3;
        child_pdf                  = child_importance * one_over_sum;
      }
    }

    if (selected_child == 0xFFFFFFFF) {
      break;
    }

    pdf *= child_pdf;

    if (selected_child_light_count > 0) {
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

__device__ TriangleHandle light_tree_query(const GBufferData data, float random, float& pdf, DeviceTransform& trans) {
  pdf = 1.0f;

  const uint32_t light_tree_handle_key = light_tree_traverse(data.position, random, pdf);

  const TriangleHandle handle = device.ptrs.light_tree_tri_handle_map[light_tree_handle_key];

  trans = load_transform(handle.instance_id);

  return handle;
}

__device__ float light_tree_query_pdf(const GBufferData data, const uint32_t light_tree_handle_key) {
  return light_tree_traverse_pdf(data.position, light_tree_handle_key);
}

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

LUMINARY_KERNEL void light_compute_power(const TriangleHandle* handles, const uint32_t lights_count, float* power_dst) {
  for (uint32_t light = THREAD_ID; light < lights_count; light += blockDim.x * gridDim.x) {
    const TriangleHandle handle = handles[light];

    const uint32_t mesh_id = mesh_id_load(handle.instance_id);

    const DeviceTriangle* tri_ptr = device.ptrs.triangles[mesh_id];
    const uint32_t triangle_count = device.ptrs.triangle_counts[mesh_id];

    const float4 t2 = __ldg((float4*) triangle_get_entry_address(tri_ptr, 2, 0, handle.tri_id, triangle_count));
    const float4 t3 = __ldg((float4*) triangle_get_entry_address(tri_ptr, 3, 0, handle.tri_id, triangle_count));

    const UV vertex_texture  = uv_unpack(__float_as_uint(t2.y));
    const UV vertex1_texture = uv_unpack(__float_as_uint(t2.z));
    const UV vertex2_texture = uv_unpack(__float_as_uint(t2.w));

    const UV edge1_texture = uv_sub(vertex1_texture, vertex_texture);
    const UV edge2_texture = uv_sub(vertex2_texture, vertex_texture);

    const uint16_t material_id    = __float_as_uint(t3.w) & 0xFFFF;
    const DeviceMaterial material = load_material(device.ptrs.materials, material_id);

    power_dst[light] = lights_integrate_emission(material, vertex_texture, edge1_texture, edge2_texture);
  }
}

#endif /* !SHADING_KERNEL */

#endif /* CU_LIGHT_H */
