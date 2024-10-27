#ifndef CU_OPTIX_SHADOW_TRACE_CUH
#define CU_OPTIX_SHADOW_TRACE_CUH

#if defined(OPTIX_KERNEL) && defined(SHADING_KERNEL)

#include "optix_utils.cuh"
#include "toy_utils.cuh"
#include "utils.cuh"

#define SKIP_IOR_CHECK (0xFFFFFFFF)
#define MAX_COMPRESSABLE_COLOR (1.99999988079071044921875f)

__device__ CompressedAlpha optix_compress_color(const RGBF color) {
  const uint32_t bits_r = (__float_as_uint(fminf(color.r + 1.0f, MAX_COMPRESSABLE_COLOR)) >> 2) & 0x1FFFFF;
  const uint32_t bits_g = (__float_as_uint(fminf(color.g + 1.0f, MAX_COMPRESSABLE_COLOR)) >> 2) & 0x1FFFFF;
  const uint32_t bits_b = (__float_as_uint(fminf(color.b + 1.0f, MAX_COMPRESSABLE_COLOR)) >> 2) & 0x1FFFFF;

  CompressedAlpha alpha;
  alpha.data0 = bits_r | (bits_g << 21);
  alpha.data1 = (bits_g >> 11) | (bits_b << 10);

  return alpha;
}

__device__ RGBF optix_decompress_color(const CompressedAlpha alpha) {
  const uint32_t bits_r = alpha.data0 & 0x1FFFFF;
  const uint32_t bits_g = (alpha.data0 >> 21) & 0x7FF | ((alpha.data1 & 0x3FF) << 11);
  const uint32_t bits_b = (alpha.data1 >> 10) & 0x1FFFFF;

  RGBF color;
  color.r = __uint_as_float(0x3F800000u | (bits_r << 2)) - 1.0f;
  color.g = __uint_as_float(0x3F800000u | (bits_g << 2)) - 1.0f;
  color.b = __uint_as_float(0x3F800000u | (bits_b << 2)) - 1.0f;

  return color;
}

__device__ bool optix_evaluate_ior_culling(const uint32_t ior_data, const ushort2 index) {
  const int8_t ior_stack_pop_max = (int8_t) (ior_data >> 24);
  if (ior_stack_pop_max > 0) {
    const uint32_t ior_stack = device.ptrs.ior_stack[get_pixel_id(index)];

    const uint32_t ray_ior = (ior_data & 0xFF);

    if (ior_stack_pop_max >= 3) {
      if (ray_ior != (ior_stack >> 24))
        return true;
    }

    if (ior_stack_pop_max >= 2) {
      if (ray_ior != (ior_stack >> 16))
        return true;
    }

    if (ior_stack_pop_max >= 1) {
      if (ray_ior != (ior_stack >> 8))
        return true;
    }
  }

  return false;
}

__device__ RGBF optix_toy_shadowing(const vec3 position, const vec3 dir, const float dist, unsigned int& compressed_ior) {
  if (device.toy.active) {
    const float toy_dist = get_toy_distance(position, dir);

    if (toy_dist < dist) {
      RGBF toy_transparency = scale_color(opaque_color(device.toy.albedo), 1.0f - device.toy.albedo.a);

      if (color_importance(toy_transparency) == 0.0f)
        return get_color(0.0f, 0.0f, 0.0f);

      // Toy can be hit at most twice, compute the intersection and on hit apply the alpha.
      vec3 toy_hit_origin = add_vector(position, scale_vector(dir, toy_dist));
      toy_hit_origin      = add_vector(toy_hit_origin, scale_vector(dir, get_length(toy_hit_origin) * eps * 16.0f));

      const float toy_dist2 = get_toy_distance(toy_hit_origin, dir);

      bool two_hits = false;
      if (toy_dist2 < dist) {
        toy_transparency = mul_color(toy_transparency, toy_transparency);
        two_hits         = true;
      }

      if (!two_hits) {
        // Set ray ior pop values to 1,1 or 0,-1 (max,balance)
        compressed_ior |= (toy_is_inside(position, dir)) ? 0x01010000 : 0x00FF0000;
      }

      return toy_transparency;
    }
  }

  return get_color(1.0f, 1.0f, 1.0f);
}

__device__ RGBF optix_geometry_shadowing(
  const vec3 position, const vec3 dir, const float dist, TriangleHandle target_light, const ushort2 index, unsigned int& compressed_ior) {
  const float3 origin = make_float3(position.x, position.y, position.z);
  const float3 ray    = make_float3(dir.x, dir.y, dir.z);

  // 21 bits for each color component.
  CompressedAlpha alpha = optix_compress_color(get_color(1.0f, 1.0f, 1.0f));

  // For triangle lights, we cannot rely on fully opaque OMMs because if we first hit the target light and then execute the closest hit for
  // that, then we will never know if there is an occluder. Similarly, skipping anyhit for fully opaque needs to still terminate the ray so
  // I enforce anyhit.
  const unsigned int ray_flag =
    (target_light.instance_id <= LIGHT_ID_TRIANGLE_ID_LIMIT) ? OPTIX_RAY_FLAG_ENFORCE_ANYHIT : OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT;

  OPTIX_PAYLOAD_INDEX_REQUIRE(OPTIX_PAYLOAD_TRIANGLE_HANDLE, 0);
  OPTIX_PAYLOAD_INDEX_REQUIRE(OPTIX_PAYLOAD_COMPRESSED_ALPHA, 2);
  OPTIX_PAYLOAD_INDEX_REQUIRE(OPTIX_PAYLOAD_IOR, 4);
  optixTrace(
    device.optix_bvh_shadow, origin, ray, 0.0f, dist, 0.0f, OptixVisibilityMask(0xFFFF), ray_flag, 0, 0, 0, target_light.instance_id,
    target_light.tri_id, alpha.data0, alpha.data1, compressed_ior);

  RGBF visibility = optix_decompress_color(alpha);

  if (target_light.instance_id == HIT_TYPE_REJECT) {
    visibility = get_color(0.0f, 0.0f, 0.0f);
  }

  if (optix_evaluate_ior_culling(compressed_ior, index)) {
    visibility = get_color(0.0f, 0.0f, 0.0f);
  }

  return visibility;
}

/*
 * Performs alpha test on triangle
 * @result 0 if opaque, 1 if transparent, 2 if alpha cutoff
 */
__device__ RGBAF optix_alpha_test(const TriangleHandle handle, const uint32_t material_id, const unsigned int ray_ior) {
  // Don't check for IOR when querying a light from a BSDF sample
  if (ray_ior != SKIP_IOR_CHECK) {
    const uint32_t compressed_ior = ior_compress(__ldg(&(device.ptrs.materials[material_id].refraction_index)));

    // This assumes that IOR is compressed into 8 bits.
    if (compressed_ior != (ray_ior & 0xFF)) {
      // Terminate ray.
      return get_RGBAF(0.0f, 0.0f, 0.0f, 1.0f);
    }
  }

  const uint16_t tex = __ldg(&(device.ptrs.materials[material_id].albedo_tex));

  if (tex != TEXTURE_NONE) {
    const UV uv = load_triangle_tex_coords(handle, optixGetTriangleBarycentrics());

    const float4 tex_value = tex2D<float4>(device.ptrs.textures[tex].handle, uv.u, 1.0f - uv.v);

    RGBAF albedo;
    albedo.r = tex_value.x;
    albedo.g = tex_value.y;
    albedo.b = tex_value.z;
    albedo.a = tex_value.w;

    return albedo;
  }

  return load_material_albedo(device.ptrs.materials, material_id);
}

extern "C" __global__ void __anyhit__optix() {
  const TriangleHandle handle = optixGetTriangleHandle();

  const TriangleHandle target_light = optixGetPayloadTriangleHandle();

  if (triangle_handle_equal(handle, target_light)) {
    optixIgnoreIntersection();
  }

  const uint32_t material_id = load_instance_material_id(handle.instance_id);

  const bool bsdf_sampling_query        = (target_light.instance_id == HIT_TYPE_LIGHT_BSDF_HINT);
  const bool material_has_ior_shadowing = (device.ptrs.materials[material_id].flags & DEVICE_MATERIAL_FLAG_IOR_SHADOWING) != 0;

  unsigned int ray_ior = (bsdf_sampling_query || !material_has_ior_shadowing) ? SKIP_IOR_CHECK : optixGetPayload_3();

  const RGBAF albedo = optix_alpha_test(handle, material_id, ray_ior);

  // For finding the hit light of BSDF rays we only care about ignoring fully transparent hits.
  // I don't have OMMs for this BVH.
  if (bsdf_sampling_query) {
    if (albedo.a == 0.0f) {
      optixIgnoreIntersection();
    }

    return;
  }

  if (albedo.a == 1.0f) {
    optixSetPayloadGeneric(OPTIX_PAYLOAD_TRIANGLE_HANDLE, HIT_TYPE_REJECT);

    optixTerminateRay();
  }

  int8_t ray_ior_pop_balance = (int8_t) ((ray_ior >> 16) & 0xFF);
  int8_t ray_ior_pop_max     = (int8_t) (ray_ior >> 24);

  ray_ior_pop_balance += (optixIsBackFaceHit()) ? 1 : -1;
  ray_ior_pop_max = max(ray_ior_pop_max, ray_ior_pop_balance);

  ray_ior = (ray_ior & 0xFFFF) | (uint32_t) (ray_ior_pop_balance << 16) | (uint32_t) (ray_ior_pop_max << 24);

  optixSetPayloadGeneric(OPTIX_PAYLOAD_IOR, ray_ior);

  if (albedo.a == 0.0f) {
    optixIgnoreIntersection();
  }

  const RGBF alpha = (device.ptrs.materials[material_id].flags & DEVICE_MATERIAL_FLAG_COLORED_TRANSPARENCY)
                       ? scale_color(opaque_color(albedo), 1.0f - albedo.a)
                       : get_color(1.0f - albedo.a, 1.0f - albedo.a, 1.0f - albedo.a);

  CompressedAlpha compressed_alpha = optixGetPayloadCompressedAlpha();

  RGBF accumulated_alpha = optix_decompress_color(compressed_alpha);
  accumulated_alpha      = mul_color(accumulated_alpha, alpha);
  compressed_alpha       = optix_compress_color(accumulated_alpha);

  optixSetPayloadCompressedAlpha(compressed_alpha);

  optixIgnoreIntersection();
}

extern "C" __global__ void __closesthit__optix() {
  unsigned int target_light = optixGetPayload_0();
  if (target_light == HIT_TYPE_LIGHT_BSDF_HINT) {
    optixSetPayload_0(optixGetPrimitiveIndex());
    return;
  }

  // This is never executed for triangle lights so we don't need to check if the closest hit is the target light.
  optixSetPayload_0(HIT_TYPE_REJECT);
}

#endif /* OPTIX_KERNEL && SHADING_KERNEL */

#endif /* CU_OPTIX_SHADOW_TRACE_CUH */
