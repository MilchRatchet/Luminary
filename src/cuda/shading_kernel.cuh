#ifndef CU_SHADING_KERNEL_H
#define CU_SHADING_KERNEL_H

#if defined(SHADING_KERNEL) && defined(OPTIX_KERNEL)

#include "bsdf.cuh"
#include "light.cuh"
#include "math.cuh"
#include "memory.cuh"
#include "ris.cuh"
#include "utils.cuh"

#define MAX_COMPRESSABLE_COLOR (1.99999988079071044921875f)

__device__ void optix_compress_color(RGBF color, unsigned int& data0, unsigned int& data1) {
  uint32_t bits_r = (__float_as_uint(fminf(color.r + 1.0f, MAX_COMPRESSABLE_COLOR)) >> 2) & 0x1FFFFF;
  uint32_t bits_g = (__float_as_uint(fminf(color.g + 1.0f, MAX_COMPRESSABLE_COLOR)) >> 2) & 0x1FFFFF;
  uint32_t bits_b = (__float_as_uint(fminf(color.b + 1.0f, MAX_COMPRESSABLE_COLOR)) >> 2) & 0x1FFFFF;

  data0 = bits_r | (bits_g << 21);
  data1 = (bits_g >> 11) | (bits_b << 10);
}

__device__ RGBF optix_decompress_color(unsigned int data0, unsigned int data1) {
  uint32_t bits_r = data0 & 0x1FFFFF;
  uint32_t bits_g = (data0 >> 21) & 0x7FF | ((data1 & 0x3FF) << 11);
  uint32_t bits_b = (data1 >> 10) & 0x1FFFFF;

  RGBF color;
  color.r = __uint_as_float(0x3F800000u | (bits_r << 2)) - 1.0f;
  color.g = __uint_as_float(0x3F800000u | (bits_g << 2)) - 1.0f;
  color.b = __uint_as_float(0x3F800000u | (bits_b << 2)) - 1.0f;

  return color;
}

__device__ RGBF
  optix_compute_light_ray(const GBufferData data, const ushort2 index, const LightRayTarget light_target, const uint32_t light_ray_index) {
  float light_sampling_weight;
  uint32_t light_id;
  switch (light_target) {
    case LIGHT_RAY_TARGET_SUN: {
      const vec3 sky_pos     = world_to_sky_transform(data.position);
      const bool sun_visible = !sph_ray_hit_p0(normalize_vector(sub_vector(device.sun_pos, sky_pos)), sky_pos, SKY_EARTH_RADIUS);
      light_id               = (sun_visible) ? LIGHT_ID_SUN : LIGHT_ID_NONE;
      light_sampling_weight  = (sun_visible) ? 1.0f : 0.0f;
    } break;
    case LIGHT_RAY_TARGET_TOY: {
      const bool toy_visible = (device.scene.toy.active && device.scene.toy.emissive);
      light_id               = (toy_visible) ? LIGHT_ID_TOY : LIGHT_ID_NONE;
      light_sampling_weight  = (toy_visible) ? 1.0f : 0.0f;
    } break;
    case LIGHT_RAY_TARGET_GEOMETRY:
    default:
      light_id = ris_sample_light(data, index, light_ray_index, light_sampling_weight);
      break;
  }

  if (light_sampling_weight == 0.0f)
    return get_color(0.0f, 0.0f, 0.0f);

  float solid_angle, dist;
  RGBF light_color;
  const vec3 dir = light_sample(light_target, light_id, data, index, light_ray_index, solid_angle, dist, light_color);

  if (solid_angle == 0.0f || luminance(light_color) == 0.0f)
    return get_color(0.0f, 0.0f, 0.0f);

  RGBF bsdf_value = bsdf_evaluate(data, dir, BSDF_SAMPLING_GENERAL, solid_angle * light_sampling_weight);

  light_color = mul_color(light_color, bsdf_value);

  const float3 origin = make_float3(data.position.x, data.position.y, data.position.z);
  const float3 ray    = make_float3(dir.x, dir.y, dir.z);

  unsigned int hit_id = light_id;

  // 21 bits for each color component.
  unsigned int alpha_data0, alpha_data1;
  optix_compress_color(get_color(1.0f, 1.0f, 1.0f), alpha_data0, alpha_data1);

  if (light_target != LIGHT_RAY_TARGET_TOY && device.scene.toy.active) {
    const float toy_dist = get_toy_distance(data.position, dir);

    if (toy_dist < dist) {
      if (ior_compress(device.scene.toy.refractive_index) != ior_compress(data.ior_in))
        return get_color(0.0f, 0.0f, 0.0f);

      RGBF toy_transparency = scale_color(opaque_color(device.scene.toy.albedo), 1.0f - device.scene.toy.albedo.a);

      if (luminance(toy_transparency) == 0.0f)
        return get_color(0.0f, 0.0f, 0.0f);

      // Toy can be hit at most twice, compute the intersection and on hit apply the alpha.
      vec3 toy_hit_origin = add_vector(data.position, scale_vector(dir, toy_dist));
      toy_hit_origin      = add_vector(toy_hit_origin, scale_vector(dir, get_length(toy_hit_origin) * eps * 16.0f));

      const float toy_dist2 = get_toy_distance(toy_hit_origin, dir);

      if (toy_dist2 + toy_dist < dist) {
        toy_transparency = mul_color(toy_transparency, toy_transparency);
      }

      light_color = mul_color(light_color, toy_transparency);
    }
  }

  unsigned int compressed_ior = ior_compress(data.ior_in);

  // Disable OMM opaque hits because we want to know if we hit something that is fully opaque so we can reject.
  optixTrace(
    device.optix_bvh_light, origin, ray, 0.0f, dist, 0.0f, OptixVisibilityMask(0xFFFF), OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT, 0, 0, 0,
    hit_id, alpha_data0, alpha_data1, compressed_ior);

  if (hit_id == HIT_TYPE_REJECT)
    return get_color(0.0f, 0.0f, 0.0f);

  RGBF visibility = optix_decompress_color(alpha_data0, alpha_data1);

  visibility = mul_color(visibility, volume_integrate_transmittance(data.position, dir, dist));

  return mul_color(light_color, visibility);
}

/*
 * Performs alpha test on triangle
 * @result 0 if opaque, 1 if transparent, 2 if alpha cutoff
 */
__device__ RGBAF optix_alpha_test() {
  const unsigned int hit_id = optixGetPrimitiveIndex();

  const uint32_t material_id    = load_triangle_material_id(hit_id);
  const uint32_t compressed_ior = ior_compress(__ldg(&(device.scene.materials[material_id].refraction_index)));
  const uint16_t tex            = __ldg(&(device.scene.materials[material_id].albedo_map));

  RGBAF albedo = get_RGBAF(0.0f, 0.0f, 0.0f, 1.0f);

  if (tex != TEXTURE_NONE && compressed_ior == __uint_as_float(optixGetPayload_3())) {
    const UV uv = load_triangle_tex_coords(hit_id, optixGetTriangleBarycentrics());

    const float4 tex_value = tex2D<float4>(device.ptrs.albedo_atlas[tex].tex, uv.u, 1.0f - uv.v);

    albedo.r = tex_value.x;
    albedo.g = tex_value.y;
    albedo.b = tex_value.z;
    albedo.a = tex_value.w;
  }

  return albedo;
}

extern "C" __global__ void __anyhit__optix() {
  if (load_triangle_light_id(optixGetPrimitiveIndex()) == optixGetPayload_0()) {
    optixIgnoreIntersection();
  }

  RGBAF albedo = optix_alpha_test();

  if (albedo.a == 0.0f) {
    optixIgnoreIntersection();
  }

  if (albedo.a == 1.0f) {
    optixSetPayload_0(HIT_TYPE_REJECT);

    optixTerminateRay();
  }

  RGBF alpha = (device.scene.material.colored_transparency) ? scale_color(opaque_color(albedo), 1.0f - albedo.a)
                                                            : get_color(1.0f - albedo.a, 1.0f - albedo.a, 1.0f - albedo.a);

  unsigned int alpha_data0 = optixGetPayload_1();
  unsigned int alpha_data1 = optixGetPayload_2();

  RGBF accumulated_alpha = optix_decompress_color(alpha_data0, alpha_data1);
  accumulated_alpha      = mul_color(accumulated_alpha, alpha);
  optix_compress_color(accumulated_alpha, alpha_data0, alpha_data1);

  optixSetPayload_1(alpha_data0);
  optixSetPayload_2(alpha_data1);

  optixIgnoreIntersection();
}

extern "C" __global__ void __closesthit__optix() {
  if (load_triangle_light_id(optixGetPrimitiveIndex()) != optixGetPayload_0()) {
    optixSetPayload_0(HIT_TYPE_REJECT);
  }
}

#endif /* SHADING_KERNEL && OPTIX_KERNEL */

#endif /* CU_SHADING_KERNEL */
