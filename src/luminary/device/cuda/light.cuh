#ifndef CU_LIGHT_H
#define CU_LIGHT_H

#if defined(OPTIX_KERNEL)

#include "bsdf.cuh"
#include "hashmap.cuh"
#include "intrinsics.cuh"
#include "light_bridges.cuh"
#include "light_common.cuh"
#include "light_microtriangle.cuh"
#include "light_tree.cuh"
#include "light_triangle.cuh"
#include "material.cuh"
#include "memory.cuh"
#include "mis.cuh"
#include "optix_common.cuh"
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

////////////////////////////////////////////////////////////////////
// Light Sampling
////////////////////////////////////////////////////////////////////

template <MaterialType TYPE>
__device__ TriangleHandle light_get_blocked_handle(const MaterialContext<TYPE> ctx) {
  return triangle_handle_get(INSTANCE_ID_INVALID, 0);
}

template <>
__device__ TriangleHandle light_get_blocked_handle<MATERIAL_GEOMETRY>(const MaterialContextGeometry ctx) {
  return triangle_handle_get(ctx.instance_id, ctx.tri_id);
}

template <MaterialType TYPE>
__device__ void light_evaluate_candidate(
  const MaterialContext<TYPE> ctx, const ushort2 pixel, TriangleLight& light, const TriangleHandle handle, const uint3 light_uv_packed,
  const float tree_sampling_weight, const uint32_t output_id, RISReservoir& reservoir, LightSampleResult<TYPE>& result) {
  const float2 ray_random = random_2D(RANDOM_TARGET_LIGHT_GEO_RAY + output_id, pixel);

  vec3 ray;
  float dist;
  float solid_angle;
  if (light_triangle_sample_finalize(light, light_uv_packed, ctx.position, ray_random, ray, dist, solid_angle) == false)
    return;

  RGBF light_color = light_get_color(light);

  // TODO: When I improve the BSDF, I need to make sure that I handle correctly refraction BSDF sampled directions, they could be
  // incorrectly flagged as a reflection here or vice versa.
  bool is_refraction;
  const RGBF bsdf_weight = bsdf_evaluate(ctx, ray, BSDF_SAMPLING_GENERAL, is_refraction, 1.0f);

  const float mis_weight = mis_compute_weight_dl(ctx, ray, light, light_color, solid_angle, is_refraction);
  light_color            = scale_color(mul_color(light_color, bsdf_weight), mis_weight);
  const float target     = color_importance(light_color);

  const float sampling_weight = tree_sampling_weight * solid_angle;

  if (ris_reservoir_add_sample(reservoir, target, sampling_weight)) {
    result.handle        = handle;
    result.ray           = ray;
    result.light_color   = light_color;
    result.dist          = dist;
    result.is_refraction = is_refraction;
  }
}

template <>
__device__ void light_evaluate_candidate<MATERIAL_VOLUME>(
  const MaterialContextVolume ctx, const ushort2 pixel, TriangleLight& light, const TriangleHandle handle, const uint3 light_uv_packed,
  const float tree_sampling_weight, const uint32_t output_id, RISReservoir& reservoir, LightSampleResult<MATERIAL_VOLUME>& result) {
  float2 target_and_weight;
  LightSampleResult<MATERIAL_VOLUME> sample = bridges_sample(ctx, light, handle, light_uv_packed, pixel, output_id, target_and_weight);

  const float target          = target_and_weight.x;
  const float sampling_weight = target_and_weight.y * tree_sampling_weight;

  if (ris_reservoir_add_sample(reservoir, target, sampling_weight)) {
    result = sample;
  }
}

template <MaterialType TYPE>
__device__ LightSampleResult<TYPE> light_list_resample(
  const MaterialContext<TYPE> ctx, const LightTreeWork& light_tree_work, ushort2 pixel, const TriangleHandle blocked_handle) {
  LightSampleResult<TYPE> result;
  result.handle = triangle_handle_get(INSTANCE_ID_INVALID, 0);

  RISReservoir reservoir = ris_reservoir_init(random_1D(RANDOM_TARGET_LIGHT_GEO_RESAMPLING, pixel));

  for (uint32_t output_id = 0; output_id < LIGHT_TREE_NUM_OUTPUTS; output_id++) {
    const LightTreeResult output = light_tree_traverse_postpass<TYPE>(ctx, pixel, output_id, light_tree_work);

    const uint32_t light_id = output.light_id;

    if (light_id == 0xFFFFFFFF)
      continue;

    DeviceTransform trans;
    const TriangleHandle light_handle = light_tree_get_light(light_id, trans);

    if (triangle_handle_equal(light_handle, blocked_handle))
      continue;

    uint3 light_uv_packed;
    TriangleLight triangle_light = light_triangle_sample_init(light_handle, trans, light_uv_packed);

    light_evaluate_candidate(ctx, pixel, triangle_light, light_handle, light_uv_packed, output.weight, output_id, reservoir, result);
  }

  const float sampling_weight = ris_reservoir_get_sampling_weight(reservoir);

  result.light_color = scale_color(result.light_color, sampling_weight);

  return result;
}

template <MaterialType TYPE>
__device__ LightSampleResult<TYPE> light_sample(const MaterialContext<TYPE> ctx, const ushort2 pixel) {
  ////////////////////////////////////////////////////////////////////
  // Sample light tree
  ////////////////////////////////////////////////////////////////////

  const LightTreeWork light_tree_work = light_tree_traverse_prepass(ctx, pixel);

  ////////////////////////////////////////////////////////////////////
  // Sample from set of list of candidates
  ////////////////////////////////////////////////////////////////////

  // Don't allow triangles to sample themselves.
  const TriangleHandle blocked_handle = light_get_blocked_handle(ctx);

  LightSampleResult<TYPE> result = light_list_resample(ctx, light_tree_work, pixel, blocked_handle);

  UTILS_CHECK_NANS(pixel, result.light_color);

  return result;
}

#else /* OPTIX_KERNEL */

#include "light_microtriangle.cuh"
#include "math.cuh"
#include "memory.cuh"
#include "utils.cuh"

////////////////////////////////////////////////////////////////////
// Light Processing
////////////////////////////////////////////////////////////////////

__device__ float lights_integrate_emission(
  const DeviceMaterial material, const UV vertex, const UV edge1, const UV edge2, const uint32_t microtriangle_id) {
  const DeviceTextureObject tex = load_texture_object(material.luminance_tex);

  float2 bary0, bary1, bary2;
  light_microtriangle_id_to_bary(microtriangle_id, bary0, bary1, bary2);

  const UV microUV0 = get_uv(vertex.u + bary0.x * edge1.u + bary0.y * edge2.u, vertex.v + bary0.x * edge1.v + bary0.y * edge2.v);
  const UV microUV1 = get_uv(vertex.u + bary1.x * edge1.u + bary1.y * edge2.u, vertex.v + bary1.x * edge1.v + bary1.y * edge2.v);
  const UV microUV2 = get_uv(vertex.u + bary2.x * edge1.u + bary2.y * edge2.u, vertex.v + bary2.x * edge1.v + bary2.y * edge2.v);

  const UV microedge1 = uv_sub(microUV1, microUV0);
  const UV microedge2 = uv_sub(microUV2, microUV0);

  // Super crude way of determining the number of texel fetches I will need. If performance of this becomes an issue
  // then I will have to rethink this here.
  const float texel_steps_u = fmaxf(fabsf(microedge1.u), fabsf(microedge2.u)) * tex.width;
  const float texel_steps_v = fmaxf(fabsf(microedge1.v), fabsf(microedge2.v)) * tex.height;

  const float steps = ceilf(fmaxf(texel_steps_u, texel_steps_v));

  const float step_size = 1.0f / steps;

  RGBF accumulator     = get_color(0.0f, 0.0f, 0.0f);
  uint32_t texel_count = 0;

  for (float a = 0.0f; a < 1.0f; a += step_size) {
    for (float b = 0.0f; a + b < 1.0f; b += step_size) {
      const float u = microUV0.u + a * microedge1.u + b * microedge2.u;
      const float v = microUV0.v + a * microedge1.v + b * microedge2.v;

      const float4 texel = texture_load(tex, get_uv(u, v));

      const RGBF color = scale_color(get_color(texel.x, texel.y, texel.z), texel.w);

      accumulator = add_color(accumulator, color);
      texel_count++;
    }
  }

  if (texel_count == 0)
    return 1.0f;

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

  const float sum_microtriangle_intensity = microtriangle_intensity1 + microtriangle_intensity2;

  const float sum_intensity = warp_reduce_sum(sum_microtriangle_intensity);

  if (microtriangle_id == 0) {
    args.dst_intensities[light_id] = sum_intensity / LIGHT_NUM_MICROTRIANGLES;
  }
}

#endif /* !OPTIX_KERNEL */

#endif /* CU_LIGHT_H */
