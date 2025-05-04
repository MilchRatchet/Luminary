#ifndef CU_LIGHT_H
#define CU_LIGHT_H

#include "bsdf.cuh"
#include "hashmap.cuh"
#include "intrinsics.cuh"
#include "light_ltc.cuh"
#include "light_microtriangle.cuh"
#include "light_tree.cuh"
#include "light_triangle.cuh"
#include "memory.cuh"
#include "ris.cuh"
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

#if defined(SHADING_KERNEL)

////////////////////////////////////////////////////////////////////
// SG Lighting
////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////
// Light Tree
////////////////////////////////////////////////////////////////////

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
        selected_child_is_leaf = ((i < 4) ? (node.rel_variance[0] >> (i * 8)) : (node.rel_variance[1] >> ((i - 4) * 8))) & 0x1;
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

////////////////////////////////////////////////////////////////////
// Light Sampling
////////////////////////////////////////////////////////////////////

struct LightSampleWorkData {
  // Current Sample
  uint32_t light_id;
  vec3 ray;
  RGBF light_color;
  float dist;
  bool is_refraction;
  // Reused data
  float solid_angle;
  float2 hit_coords;
  float tree_sampling_weight;
} typedef LightSampleWorkData;

__device__ void light_sample_common(
  const GBufferData data, const uint32_t light_id, const vec3 ray, const float dist, TriangleLight& light, RISReservoir& reservoir,
  LightSampleWorkData& work) {
  RGBF light_color = light_get_color(light);

  // TODO: When I improve the BSDF, I need to make sure that I handle correctly refraction BSDF sampled directions, they could be
  // incorrectly flagged as a reflection here or vice versa.
  bool is_refraction;
  const RGBF bsdf_weight = bsdf_evaluate(data, ray, BSDF_SAMPLING_GENERAL, is_refraction);
  light_color            = mul_color(light_color, bsdf_weight);
  const float target     = color_importance(light_color);

  const float one_over_solid_angle_pdf = work.solid_angle;
  const float bsdf_pdf                 = bsdf_sample_for_light_pdf(data, ray);

  const float sampling_weight = work.tree_sampling_weight * one_over_solid_angle_pdf / (1.0f + bsdf_pdf * one_over_solid_angle_pdf);

  if (ris_reservoir_add_sample(reservoir, target, sampling_weight)) {
    work.light_id      = light_id;
    work.ray           = ray;
    work.light_color   = light_color;
    work.dist          = dist;
    work.is_refraction = is_refraction;
  }
}

__device__ void light_sample_solid_angle(
  const GBufferData data, const uint32_t light_id, const ushort2 pixel, TriangleLight& triangle_light, const uint3 light_uv_packed,
  RISReservoir& reservoir, LightSampleWorkData& work) {
  const float2 ray_random = quasirandom_sequence_2D(QUASI_RANDOM_TARGET_RIS_RAY_DIR, pixel);

  vec3 ray;
  float dist;
  if (light_triangle_sample_finalize(triangle_light, light_uv_packed, data.position, ray_random, ray, dist, work.solid_angle) == false)
    return;

  light_sample_common(data, light_id, ray, dist, triangle_light, reservoir, work);
}

__device__ void light_sample_bsdf(
  const GBufferData data, const uint32_t light_id, const ushort2 pixel, TriangleLight& triangle_light, const uint3 light_uv_packed,
  RISReservoir& reservoir, LightSampleWorkData& work) {
  bool bsdf_sample_is_refraction = false;
  bool bsdf_sample_is_valid      = false;
  const vec3 ray = bsdf_sample_for_light(data, pixel, QUASI_RANDOM_TARGET_LIGHT_BSDF, bsdf_sample_is_refraction, bsdf_sample_is_valid);

  if (bsdf_sample_is_valid == false)
    return;

  float dist;
  if (light_triangle_sample_finalize_dist_and_uvs(triangle_light, light_uv_packed, data.position, ray, dist) == false)
    return;

  light_sample_common(data, light_id, ray, dist, triangle_light, reservoir, work);
}

__device__ void light_linked_list_resample_brute_force(
  const GBufferData data, const LightSubsetReference stack[LIGHT_TREE_MAX_SUBSET_REFERENCES], const uint32_t num_references, ushort2 pixel,
  const TriangleHandle blocked_handle, RISReservoir& reservoir, LightSampleWorkData& work) {
  uint32_t reference_ptr = 0;
  uint32_t subset_id     = 0;

  bool reached_end = false;

  DeviceLightSubset subset;
  subset.index = 0;
  subset.count = 0;

  while (!reached_end) {
    if (subset.count == 0) {
      if (reference_ptr < num_references) {
        const LightSubsetReference reference = stack[reference_ptr++];

        subset_id                 = reference.subset_id;
        work.tree_sampling_weight = reference.sampling_weight;
      }
      else {
        reached_end = true;
        break;
      }

      subset = load_light_subset(subset_id);
    }

    const uint32_t light_id = subset.index;

    DeviceTransform trans;
    const TriangleHandle light_handle = light_tree_get_light(light_id, trans);

    if (triangle_handle_equal(light_handle, blocked_handle) == false) {
      uint3 light_uv_packed;
      TriangleLight triangle_light = light_triangle_sample_init(light_handle, trans, light_uv_packed);

      light_sample_solid_angle(data, light_id, pixel, triangle_light, light_uv_packed, reservoir, work);
      light_sample_bsdf(data, light_id, pixel, triangle_light, light_uv_packed, reservoir, work);
    }

    subset.index++;
    subset.count--;
  }
}

__device__ TriangleHandle light_sample(
  const GBufferData data, const ushort2 pixel, vec3& selected_ray, RGBF& selected_light_color, float& selected_dist,
  bool& selected_is_refraction) {
  selected_ray           = get_vector(0.0f, 0.0f, 1.0f);
  selected_light_color   = get_color(0.0f, 0.0f, 0.0f);
  selected_dist          = 1.0f;
  selected_is_refraction = false;

  // Don't allow triangles to sample themselves.
  const TriangleHandle blocked_handle = triangle_handle_get(data.instance_id, data.tri_id);

  ////////////////////////////////////////////////////////////////////
  // Sample light tree
  ////////////////////////////////////////////////////////////////////

  const float light_tree_random = quasirandom_sequence_1D(QUASI_RANDOM_TARGET_RIS_LIGHT_TREE, pixel);

  LightSubsetReference stack[LIGHT_TREE_MAX_SUBSET_REFERENCES];
  const uint32_t num_references = light_tree_query(data, light_tree_random, pixel, stack);

  // This happens if no linked list with non zero importance was found.
  if (num_references == 0)
    return triangle_handle_get(LIGHT_ID_NONE, 0);

  ////////////////////////////////////////////////////////////////////
  // Sample from set of linked lists
  ////////////////////////////////////////////////////////////////////

  RISReservoir reservoir = ris_reservoir_init(quasirandom_sequence_1D(QUASI_RANDOM_TARGET_RIS_RESAMPLING, pixel));

  // We don't need to initialize it. It will end up uninitialized if and only if ris_reservoir.target is 0.
  LightSampleWorkData work;

  light_linked_list_resample_brute_force(data, stack, num_references, pixel, blocked_handle, reservoir, work);

  const float sampling_weight = ris_reservoir_get_sampling_weight(reservoir);

  // This happens if no light with non zero importance was found.
  if (work.light_id == 0xFFFFFFFF)
    return triangle_handle_get(LIGHT_ID_NONE, 0);

  if (reservoir.selected_target == 0.0f)
    return triangle_handle_get(LIGHT_ID_NONE, 0);

  ////////////////////////////////////////////////////////////////////
  // Sample direction
  ////////////////////////////////////////////////////////////////////

  DeviceTransform trans;
  const TriangleHandle light_handle = light_tree_get_light(work.light_id, trans);

  if (triangle_handle_equal(light_handle, blocked_handle))
    return triangle_handle_get(LIGHT_ID_NONE, 0);

  // if (is_center_pixel(pixel) && IS_PRIMARY_RAY) {
  //   printf(
  //     "ID:%u Num:%u Importance:%f Weight:%f\n=================\n", light_id, reservoir.num_samples, reservoir.selected_target,
  //     direction_sampling_weight * linked_list_sampling_weight);
  // }

  selected_ray           = work.ray;
  selected_light_color   = work.light_color;
  selected_dist          = work.dist;
  selected_is_refraction = work.is_refraction;

  ////////////////////////////////////////////////////////////////////
  // Compute the shading weight of the selected light (Probability of selecting the light through WRS)
  ////////////////////////////////////////////////////////////////////

  selected_light_color = scale_color(selected_light_color, sampling_weight);

  UTILS_CHECK_NANS(pixel, selected_light_color);

  return light_handle;
}

#endif /* !VOLUME_KERNEL */

#else /* SHADING_KERNEL */

////////////////////////////////////////////////////////////////////
// Light Processing
////////////////////////////////////////////////////////////////////

__device__ float lights_integrate_emission(
  const DeviceMaterial material, const UV vertex, const UV edge1, const UV edge2, const uint32_t microtriangle_id) {
  const DeviceTextureObject tex = load_texture_object(material.luminance_tex);

  float2 bary0, bary1, bary2;
  light_microtriangle_id_to_bary(microtriangle_id, bary0, bary1, bary2);

  const UV microv0 = get_uv(vertex.u + bary0.x * edge1.u + bary0.y * edge2.u, vertex.v + bary0.x * edge1.v + bary0.y * edge2.v);
  const UV microv1 = get_uv(vertex.u + bary1.x * edge1.u + bary1.y * edge2.u, vertex.v + bary1.x * edge1.v + bary1.y * edge2.v);
  const UV microv2 = get_uv(vertex.u + bary2.x * edge1.u + bary2.y * edge2.u, vertex.v + bary2.x * edge1.v + bary2.y * edge2.v);

  const UV microedge1 = uv_sub(microv1, microv0);
  const UV microedge2 = uv_sub(microv2, microv0);

  // Super crude way of determining the number of texel fetches I will need. If performance of this becomes an issue
  // then I will have to rethink this here.
  const float texel_steps_u = fmaxf(fabsf(microedge1.u), fabsf(microedge2.u)) * tex.width;
  const float texel_steps_v = fmaxf(fabsf(microedge1.v), fabsf(microedge2.v)) * tex.height;

  const float steps = ceilf(fmaxf(texel_steps_u, texel_steps_v));

  const float step_size = 1.0f / steps;

  RGBF accumulator  = get_color(0.0f, 0.0f, 0.0f);
  float texel_count = 0.0f;

  for (float a = 0.0f; a < 1.0f; a += step_size) {
    for (float b = 0.0f; a + b < 1.0f; b += step_size) {
      const float u = microv0.u + a * microedge1.u + b * microedge2.u;
      const float v = microv0.v + a * microedge1.v + b * microedge2.v;

      const float4 texel = texture_load(tex, get_uv(u, v));

      const RGBF color = scale_color(get_color(texel.x, texel.y, texel.z), texel.w);

      accumulator = add_color(accumulator, color);
      texel_count += 1.0f;
    }
  }

  return color_importance(accumulator) / texel_count;
}

LUMINARY_KERNEL void light_compute_intensity(const KernelArgsLightComputeIntensity args) {
  const uint32_t light_id = THREAD_ID >> 5;

  if (light_id >= args.lights_count)
    return;

  const uint32_t microtriangle_id = (THREAD_ID & ((1 << 5) - 1)) << 1;

  const uint32_t mesh_id     = args.mesh_ids[light_id];
  const uint32_t triangle_id = args.triangle_ids[light_id];

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

  const float microtriangle_intensity1 =
    lights_integrate_emission(material, vertex_texture, edge1_texture, edge2_texture, microtriangle_id + 0);
  const float microtriangle_intensity2 =
    lights_integrate_emission(material, vertex_texture, edge1_texture, edge2_texture, microtriangle_id + 1);

  const float max_microtriangle_intensity = fmaxf(microtriangle_intensity1, microtriangle_intensity2);
  const float sum_microtriangle_intensity = microtriangle_intensity1 + microtriangle_intensity2;

  const float max_intensity = warp_reduce_max(max_microtriangle_intensity);
  const float sum_intensity = warp_reduce_sum(sum_microtriangle_intensity);

  const float normalized_intensity1   = (max_intensity > 0.0f) ? microtriangle_intensity1 / max_intensity : 0.0f;
  const uint8_t compressed_intensity1 = max((uint8_t) (normalized_intensity1 * 15.0f + 0.5f), 1);

  const float normalized_intensity2   = (max_intensity > 0.0f) ? microtriangle_intensity2 / max_intensity : 0.0f;
  const uint8_t compressed_intensity2 = max((uint8_t) (normalized_intensity2 * 15.0f + 0.5f), 1);

  const uint8_t compressed_intensity = compressed_intensity1 | (compressed_intensity2 << 4);

  args.dst_microtriangle_importance[light_id * (LIGHT_NUM_MICROTRIANGLES >> 1) + (microtriangle_id >> 1)] = compressed_intensity;

  const float importance = compressed_intensity1 + compressed_intensity2;

  const float sum_importance = warp_reduce_sum(importance);

  if (microtriangle_id == 0) {
    args.dst_importance_normalization[light_id] = sum_importance;
    args.dst_intensities[light_id]              = sum_intensity;
  }
}

#endif /* !SHADING_KERNEL */

#endif /* CU_LIGHT_H */
