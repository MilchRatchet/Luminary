#ifndef CU_LIGHT_H
#define CU_LIGHT_H

#if defined(SHADING_KERNEL)

#include "intrinsics.cuh"
#include "memory.cuh"
#include "sky_utils.cuh"
#include "texture_utils.cuh"
#include "utils.cuh"

__device__ float light_tree_child_importance(
  const GBufferData data, const LightTreeNode8Packed node, const vec3 exp, const uint32_t i, const uint32_t shift) {
  vec3 point;
  point = get_vector((node.rel_point_x[i] >> shift) & 0xFF, (node.rel_point_y[i] >> shift) & 0xFF, (node.rel_point_z[i] >> shift) & 0xFF);
  point = mul_vector(point, exp);
  point = add_vector(point, node.base_point);

  const float energy     = ((node.rel_energy[i] >> shift) & 0xFF) * (node.max_energy / 255.0f);
  const float confidence = ((node.rel_confidence[i] >> shift) & 0xFF) * (node.max_confidence / 255.0f);

  const vec3 diff  = sub_vector(point, data.position);
  const float dist = fmaxf(get_length(diff), confidence);

  return energy / (dist * dist);
}

__device__ uint32_t light_tree_traverse(const GBufferData data, float random, uint32_t& subset_length, float& pdf) {
  if (!device.scene.material.light_tree_active) {
    subset_length = device.scene.triangle_lights_count;
    pdf           = 1.0f;
    return 0;
  }

  pdf = 1.0f;

#if 0
  LightTreeNode node = load_light_tree_node(device.light_tree_nodes, 0);

  while (node.light_count == 0) {
    const vec3 left_diff  = sub_vector(node.left_ref_point, data.position);
    const float left_dist = fmaxf(get_length(left_diff), node.left_confidence);

    const float left_importance = node.left_energy / (left_dist * left_dist);

    const vec3 right_diff  = sub_vector(node.right_ref_point, data.position);
    const float right_dist = fmaxf(get_length(right_diff), node.right_confidence);

    const float right_importance = node.right_energy / (right_dist * right_dist);

    const float sum_importance = left_importance + right_importance;

    if (sum_importance == 0.0f) {
      subset_length = 0;
      pdf           = 1.0f;
      return 0;
    }

    const float left_prob  = left_importance / sum_importance;
    const float right_prob = right_importance / sum_importance;

    uint32_t next_node_address = node.ptr;

    if (random < left_prob) {
      pdf *= left_prob;

      random = random / left_prob;
    }
    else {
      pdf *= right_prob;
      next_node_address++;

      random = (random - left_prob) / right_prob;
    }

    node = load_light_tree_node(device.light_tree_nodes, next_node_address);
  }

  subset_length = node.light_count;

  return node.ptr;
#else
  LightTreeNode8Packed node = load_light_tree_node_8(device.light_tree_nodes_8, 0);

  uint32_t subset_ptr = 0xFFFFFFFF;
  subset_length       = 0;

  random = random_saturate(random);

  while (subset_ptr == 0xFFFFFFFF) {
    const vec3 exp = get_vector(expf(node.exp_x), expf(node.exp_y), expf(node.exp_z));

    float importance[8];

    importance[0] = (node.child_count > 0) ? light_tree_child_importance(data, node, exp, 0, 0) : 0.0f;
    importance[1] = (node.child_count > 1) ? light_tree_child_importance(data, node, exp, 0, 8) : 0.0f;
    importance[2] = (node.child_count > 2) ? light_tree_child_importance(data, node, exp, 0, 16) : 0.0f;
    importance[3] = (node.child_count > 3) ? light_tree_child_importance(data, node, exp, 0, 24) : 0.0f;
    importance[4] = (node.child_count > 4) ? light_tree_child_importance(data, node, exp, 1, 0) : 0.0f;
    importance[5] = (node.child_count > 5) ? light_tree_child_importance(data, node, exp, 1, 8) : 0.0f;
    importance[6] = (node.child_count > 6) ? light_tree_child_importance(data, node, exp, 1, 16) : 0.0f;
    importance[7] = (node.child_count > 7) ? light_tree_child_importance(data, node, exp, 1, 24) : 0.0f;

    float sum_importance = 0.0f;
    for (uint32_t i = 0; i < 8; i++) {
      sum_importance += importance[i];
    }

    float accumulated_importance = 0.0f;
    const float one_over_sum     = 1.0f / sum_importance;

    uint32_t selected_child = 0xFFFFFFFF;
    float child_pdf         = 0.0f;
    float random_shift      = 0.0f;

    for (uint32_t i = 0; i < 8; i++) {
      const float child_importance = importance[i] * one_over_sum;
      accumulated_importance += child_importance;

      if (accumulated_importance > random) {
        selected_child = i;
        child_pdf      = child_importance;

        random_shift = accumulated_importance - child_importance;

        // No control flow, we always loop over all children.
        accumulated_importance = -FLT_MAX;
      }
    }

    if (selected_child == 0xFFFFFFFF) {
      subset_length = 0;
      subset_ptr    = 0;
      break;
    }

    pdf *= child_pdf;

    // Rescale random number
    random = random_saturate((random - random_shift) / child_pdf);

    uint32_t selected_light_index;
    selected_light_index = (selected_child > 3) ? node.light_index[1] : node.light_index[0];

    const uint32_t normalized_child = (selected_child > 3) ? selected_child - 4 : selected_child;

    selected_light_index = (selected_light_index >> (normalized_child * 8)) & 0xFF;

    const uint32_t light_count = selected_light_index & 0x3;
    const uint32_t light_ptr   = selected_light_index >> 2;

    if (light_count > 0) {
      subset_length = light_count;
      subset_ptr    = node.light_ptr + light_ptr;
      break;
    }

    node = load_light_tree_node_8(device.light_tree_nodes_8, node.child_ptr + selected_child);
  }

  return subset_ptr;
#endif
}

__device__ float light_tree_traverse_pdf(const GBufferData data, uint32_t light_id) {
  if (!device.scene.material.light_tree_active) {
    return 1.0f / device.scene.triangle_lights_count;
  }

  uint32_t light_path = __ldg(device.light_tree_paths + light_id);
  LightTreeNode node  = load_light_tree_node(device.light_tree_nodes, 0);

  float pdf = 1.0f;

  while (node.light_count == 0) {
    const vec3 left_diff  = sub_vector(node.left_ref_point, data.position);
    const float left_dist = get_length(left_diff);

    const float left_importance = node.left_energy / (left_dist * left_dist);

    const vec3 right_diff  = sub_vector(node.right_ref_point, data.position);
    const float right_dist = get_length(right_diff);

    const float right_importance = node.right_energy / (right_dist * right_dist);

    const float sum_importance = left_importance + right_importance;

    const float left_prob  = left_importance / sum_importance;
    const float right_prob = right_importance / sum_importance;

    uint32_t next_node_address = node.ptr;

    if (light_path & 0b1) {
      pdf *= left_prob;
    }
    else {
      pdf *= right_prob;
      next_node_address++;
    }

    light_path = light_path >> 1;

    node = load_light_tree_node(device.light_tree_nodes, next_node_address);
  }

  pdf *= 1.0f / node.light_count;

  return pdf;
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

/*
 * Solid angle sample a triangle.
 * @param triangle Triangle.
 * @param data Data about shading point.
 * @param random Random numbers.
 * @param pdf PDF of sampled direction.
 * @param dist Distance to sampled point on triangle.
 * @param color Emission from triangle at sampled point.
 * @result Normalized direction to the point on the triangle.
 *
 * Robust solid angle sampling method from
 * C. Peters, "BRDF Importance Sampling for Linear Lights", Computer Graphics Forum (Proc. HPG) 40, 8, 2021.
 *
 */
__device__ vec3 light_sample_triangle(
  const TriangleLight triangle, const GBufferData data, const float2 random, float& solid_angle, float& dist, RGBF& color) {
  const vec3 v0 = normalize_vector(sub_vector(triangle.vertex, data.position));
  const vec3 v1 = normalize_vector(sub_vector(add_vector(triangle.vertex, triangle.edge1), data.position));
  const vec3 v2 = normalize_vector(sub_vector(add_vector(triangle.vertex, triangle.edge2), data.position));

  const float G0 = fabsf(dot_product(cross_product(v0, v1), v2));
  const float G1 = dot_product(v0, v2) + dot_product(v1, v2);
  const float G2 = 1.0f + dot_product(v0, v1);

  solid_angle = 2.0f * atan2f(G0, G1 + G2);

  if (isnan(solid_angle) || isinf(solid_angle) || solid_angle < 1e-7f) {
    solid_angle = 0.0f;
    return get_vector(0.0f, 0.0f, 0.0f);
  }

  const float sampled_solid_angle = random.x * solid_angle;

  const vec3 r = add_vector(
    scale_vector(v0, G0 * cosf(0.5f * sampled_solid_angle) - G1 * sinf(0.5f * sampled_solid_angle)),
    scale_vector(v2, G2 * sinf(0.5f * sampled_solid_angle)));

  const vec3 v2_t = sub_vector(scale_vector(r, 2.0f * dot_product(v0, r) / dot_product(r, r)), v0);

  const float s2 = dot_product(v1, v2_t);
  const float s  = (1.0f - random.y) + random.y * s2;
  const float t  = sqrtf(fmaxf((1.0f - s * s) / (1.0f - s2 * s2), 0.0f));

  const vec3 dir = normalize_vector(add_vector(scale_vector(v1, s - t * s2), scale_vector(v2_t, t)));

  if (isnan(dir.x) || isnan(dir.y) || isnan(dir.z)) {
    solid_angle = 0.0f;
    return get_vector(0.0f, 0.0f, 0.0f);
  }

  float2 coords;
  dist = light_triangle_intersection_uv(triangle, data.position, dir, coords);

  // Our ray does not actually hit the light, abort.
  if (dist == FLT_MAX) {
    solid_angle = 0.0f;
    return get_vector(0.0f, 0.0f, 0.0f);
  }

  const uint16_t albedo_tex = device.scene.materials[triangle.material_id].albedo_map;
  const uint16_t illum_tex  = device.scene.materials[triangle.material_id].luminance_map;

  // Load texture coordinates if we need them.
  UV tex_coords;
  if (illum_tex != TEXTURE_NONE || albedo_tex != TEXTURE_NONE) {
    tex_coords = load_triangle_tex_coords(triangle.triangle_id, coords);
  }

  if (illum_tex != TEXTURE_NONE) {
    const float4 emission = texture_load(device.ptrs.luminance_atlas[illum_tex], tex_coords);

    color = scale_color(get_color(emission.x, emission.y, emission.z), device.scene.material.default_material.b * emission.w);
  }
  else {
    color.r = random_uint16_t_to_float(device.scene.materials[triangle.material_id].emission_r);
    color.g = random_uint16_t_to_float(device.scene.materials[triangle.material_id].emission_g);
    color.b = random_uint16_t_to_float(device.scene.materials[triangle.material_id].emission_b);

    const float scale = (float) (device.scene.materials[triangle.material_id].emission_scale);

    color = scale_color(color, device.scene.material.default_material.b * scale);
  }

  if (color_importance(color) > 0.0f) {
    float alpha;
    if (albedo_tex != TEXTURE_NONE) {
      alpha = texture_load(device.ptrs.albedo_atlas[albedo_tex], tex_coords).w;
    }
    else {
      alpha = random_uint16_t_to_float(device.scene.materials[triangle.material_id].albedo_a);
    }

    color = scale_color(color, alpha);
  }

  return dir;
}

__device__ void light_sample_triangle_presampled(
  const TriangleLight triangle, const GBufferData data, const vec3 ray, float& solid_angle, float& dist, RGBF& color) {
  float2 coords;
  dist = light_triangle_intersection_uv(triangle, data.position, ray, coords);

  // Our ray does not actually hit the light, abort. This should never happen!
  if (dist == FLT_MAX) {
    solid_angle = 0.0f;
    return;
  }

  solid_angle = sample_triangle_solid_angle(triangle, data.position);

  const uint16_t albedo_tex = device.scene.materials[triangle.material_id].albedo_map;
  const uint16_t illum_tex  = device.scene.materials[triangle.material_id].luminance_map;

  // Load texture coordinates if we need them.
  UV tex_coords;
  if (illum_tex != TEXTURE_NONE || albedo_tex != TEXTURE_NONE) {
    tex_coords = load_triangle_tex_coords(triangle.triangle_id, coords);
  }

  if (illum_tex != TEXTURE_NONE) {
    const float4 emission = texture_load(device.ptrs.luminance_atlas[illum_tex], tex_coords);

    color = scale_color(get_color(emission.x, emission.y, emission.z), device.scene.material.default_material.b * emission.w);
  }
  else {
    color.r = random_uint16_t_to_float(device.scene.materials[triangle.material_id].emission_r);
    color.g = random_uint16_t_to_float(device.scene.materials[triangle.material_id].emission_g);
    color.b = random_uint16_t_to_float(device.scene.materials[triangle.material_id].emission_b);

    const float scale = (float) (device.scene.materials[triangle.material_id].emission_scale);

    color = scale_color(color, device.scene.material.default_material.b * scale);
  }

  if (color_importance(color) > 0.0f) {
    float alpha;
    if (albedo_tex != TEXTURE_NONE) {
      alpha = texture_load(device.ptrs.albedo_atlas[albedo_tex], tex_coords).w;
    }
    else {
      alpha = random_uint16_t_to_float(device.scene.materials[triangle.material_id].albedo_a);
    }

    color = scale_color(color, alpha);
  }
}

#else /* SHADING_KERNEL */

////////////////////////////////////////////////////////////////////
// Light Processing
////////////////////////////////////////////////////////////////////

__device__ float lights_integrate_emission(const TriangleLight light, const UV vertex, const UV edge1, const UV edge2) {
  const Material mat = load_material(device.scene.materials, light.material_id);

  const DeviceTexture tex = device.ptrs.luminance_atlas[mat.luminance_map];

  // Super crude way of determining the number of texel fetches I will need. If performance of this becomes an issue
  // then I will have to rethink this here.
  const float texel_steps_u = fmaxf(fabsf(edge1.u), fabsf(edge2.u)) / tex.inv_width;
  const float texel_steps_v = fmaxf(fabsf(edge1.v), fabsf(edge2.v)) / tex.inv_height;

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

LUMINARY_KERNEL void lights_compute_power(const TriangleLight* tris, const uint32_t lights_count, float* power_dst) {
  for (uint32_t light = THREAD_ID; light < lights_count; light += blockDim.x * gridDim.x) {
    const TriangleLight light_triangle = load_triangle_light(tris, light);

    const float2 t5 = __ldg((float2*) triangle_get_entry_address(4, 2, light_triangle.triangle_id));
    const float4 t6 = __ldg((float4*) triangle_get_entry_address(5, 0, light_triangle.triangle_id));

    const UV vertex_texture = get_uv(t5.x, t5.y);
    const UV edge1_texture  = get_uv(t6.x, t6.y);
    const UV edge2_texture  = get_uv(t6.z, t6.w);

    power_dst[light] = lights_integrate_emission(light_triangle, vertex_texture, edge1_texture, edge2_texture);
  }
}

void lights_compute_power_host(const TriangleLight* device_triangle_lights, uint32_t lights_count, float* device_power_dst) {
  lights_compute_power<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(device_triangle_lights, lights_count, device_power_dst);

  gpuErrchk(cudaDeviceSynchronize());
}

#endif /* !SHADING_KERNEL */

#endif /* CU_LIGHT_H */
