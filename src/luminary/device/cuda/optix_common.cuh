#ifndef CU_LUMINARY_OPTIX_COMMON_H
#define CU_LUMINARY_OPTIX_COMMON_H

#include "memory.cuh"
#include "optix_utils.cuh"
#include "toy_utils.cuh"
#include "utils.cuh"

enum OptixAlphaResult {
  OPTIX_ALPHA_RESULT_OPAQUE      = 0,
  OPTIX_ALPHA_RESULT_SEMI        = 1,
  OPTIX_ALPHA_RESULT_TRANSPARENT = 2
} typedef OptixAlphaResult;

// TODO: Port over.
#if 0
/*
 * Performs alpha test on triangle
 * @result 0 if opaque, 1 if transparent, 2 if alpha cutoff
 */
__device__ OptixAlphaResult optix_alpha_test() {
  const unsigned int hit_id = optixGetPrimitiveIndex();

  const uint32_t material_id = load_triangle_material_id(hit_id);
  const uint16_t tex         = __ldg(&(device.ptrs.materials[material_id].albedo_tex));

  if (tex != TEXTURE_NONE) {
    const UV uv = load_triangle_tex_coords(hit_id, optixGetTriangleBarycentrics());

    const float alpha = tex2D<float4>(device.ptrs.albedo_atlas[tex].handle, uv.u, 1.0f - uv.v).w;

    if (alpha == 0.0f) {
      return OPTIX_ALPHA_RESULT_TRANSPARENT;
    }

    if (alpha < 1.0f) {
      return OPTIX_ALPHA_RESULT_SEMI;
    }
  }

  return OPTIX_ALPHA_RESULT_OPAQUE;
}
#endif

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

__device__ RGBAF optix_get_albedo_with_ior_check(const TriangleHandle handle, const uint32_t material_id, const unsigned int ray_ior) {
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

__device__ bool particle_opacity_cutout(const float2 coord) {
  const float dx = fabsf(coord.x - 0.5f);
  const float dy = fabsf(coord.y - 0.5f);

  const float r = dx * dx + dy * dy;

  return (r > 0.25f);
}

#ifdef SHADING_KERNEL
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
    device.optix_bvh_shadow, origin, ray, 0.0f, dist, 0.0f, OptixVisibilityMask(0xFFFF), ray_flag, OPTIX_SBT_OFFSET_SHADOW_TRACE, 0, 0,
    target_light.instance_id, target_light.tri_id, alpha.data0, alpha.data1, compressed_ior);

  RGBF visibility = optix_decompress_color(alpha);

  if (target_light.instance_id == HIT_TYPE_REJECT) {
    visibility = get_color(0.0f, 0.0f, 0.0f);
  }

  if (optix_evaluate_ior_culling(compressed_ior, index)) {
    visibility = get_color(0.0f, 0.0f, 0.0f);
  }

  return visibility;
}
#endif

#endif /* CU_LUMINARY_OPTIX_COMMON_H */
