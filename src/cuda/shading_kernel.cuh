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

__device__ RGBF optix_compute_light_ray_sun(const GBufferData data, const ushort2 index, const uint32_t light_ray_index) {
  const vec3 sky_pos     = world_to_sky_transform(data.position);
  const bool sun_visible = !sph_ray_hit_p0(normalize_vector(sub_vector(device.sun_pos, sky_pos)), sky_pos, SKY_EARTH_RADIUS);

  if (!sun_visible)
    return get_color(0.0f, 0.0f, 0.0f);

  const float2 random = quasirandom_sequence_2D(QUASI_RANDOM_TARGET_TBD_3 + light_ray_index, index);

  float solid_angle;
  const vec3 dir   = sample_sphere(device.sun_pos, SKY_SUN_RADIUS, sky_pos, random, solid_angle);
  RGBF light_color = sky_get_sun_color(sky_pos, dir);

  const RGBF bsdf_value = bsdf_evaluate(data, dir, BSDF_SAMPLING_GENERAL, solid_angle);
  light_color           = mul_color(light_color, bsdf_value);

  const float3 origin = make_float3(data.position.x, data.position.y, data.position.z);
  const float3 ray    = make_float3(dir.x, dir.y, dir.z);

  // TODO: Add specialized anyhit shaders for non geometry lights
  unsigned int hit_id = LIGHT_ID_SUN;

  // 21 bits for each color component.
  unsigned int alpha_data0, alpha_data1;
  optix_compress_color(get_color(1.0f, 1.0f, 1.0f), alpha_data0, alpha_data1);

  unsigned int compressed_ior = ior_compress((data.flags & G_BUFFER_IS_TRANSPARENT_PASS) ? data.ior_out : data.ior_in);

  if (device.scene.toy.active) {
    const float toy_dist = get_toy_distance(data.position, dir);

    if (toy_dist < FLT_MAX) {
      // TODO: This only works when we enter a surface, what about the exit???
      if (device.scene.material.enable_ior_shadowing && ior_compress(device.scene.toy.refractive_index) != compressed_ior)
        return get_color(0.0f, 0.0f, 0.0f);

      RGBF toy_transparency = scale_color(opaque_color(device.scene.toy.albedo), 1.0f - device.scene.toy.albedo.a);

      if (luminance(toy_transparency) == 0.0f)
        return get_color(0.0f, 0.0f, 0.0f);

      // Toy can be hit at most twice, compute the intersection and on hit apply the alpha.
      vec3 toy_hit_origin = add_vector(data.position, scale_vector(dir, toy_dist));
      toy_hit_origin      = add_vector(toy_hit_origin, scale_vector(dir, get_length(toy_hit_origin) * eps * 16.0f));

      const float toy_dist2 = get_toy_distance(toy_hit_origin, dir);

      if (toy_dist2 < FLT_MAX)
        toy_transparency = mul_color(toy_transparency, toy_transparency);

      light_color = mul_color(light_color, toy_transparency);
    }
  }

  optixTrace(
    device.optix_bvh_light, origin, ray, 0.0f, FLT_MAX, 0.0f, OptixVisibilityMask(0xFFFF), OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT, 0, 0, 0,
    hit_id, alpha_data0, alpha_data1, compressed_ior);

  if (hit_id == HIT_TYPE_REJECT)
    return get_color(0.0f, 0.0f, 0.0f);

  RGBF visibility = optix_decompress_color(alpha_data0, alpha_data1);

  visibility = mul_color(visibility, volume_integrate_transmittance(data.position, dir, FLT_MAX));

  return mul_color(light_color, visibility);
}

__device__ RGBF optix_compute_light_ray_toy(const GBufferData data, const ushort2 index, const uint32_t light_ray_index) {
  const bool toy_visible = (device.scene.toy.active && device.scene.toy.emissive);

  if (!toy_visible)
    return get_color(0.0f, 0.0f, 0.0f);

  const float2 random = quasirandom_sequence_2D(QUASI_RANDOM_TARGET_TBD_3 + light_ray_index, index);

  const vec3 dir   = toy_sample_ray(data.position, random);
  const float dist = get_toy_distance(data.position, dir);

  const RGBF bsdf_value = bsdf_evaluate(data, dir, BSDF_SAMPLING_GENERAL, toy_get_solid_angle(data.position));

  RGBF light_color = scale_color(device.scene.toy.emission, device.scene.toy.material.b);
  light_color      = mul_color(light_color, bsdf_value);

  const float3 origin = make_float3(data.position.x, data.position.y, data.position.z);
  const float3 ray    = make_float3(dir.x, dir.y, dir.z);

  unsigned int hit_id = LIGHT_ID_TOY;

  // 21 bits for each color component.
  unsigned int alpha_data0, alpha_data1;
  optix_compress_color(get_color(1.0f, 1.0f, 1.0f), alpha_data0, alpha_data1);

  unsigned int compressed_ior = ior_compress((data.flags & G_BUFFER_IS_TRANSPARENT_PASS) ? data.ior_out : data.ior_in);

  optixTrace(
    device.optix_bvh_light, origin, ray, 0.0f, dist, 0.0f, OptixVisibilityMask(0xFFFF), OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT, 0, 0, 0,
    hit_id, alpha_data0, alpha_data1, compressed_ior);

  if (hit_id == HIT_TYPE_REJECT)
    return get_color(0.0f, 0.0f, 0.0f);

  RGBF visibility = optix_decompress_color(alpha_data0, alpha_data1);

  visibility = mul_color(visibility, volume_integrate_transmittance(data.position, dir, dist));

  return mul_color(light_color, visibility);
}

__device__ RGBF optix_compute_light_ray_geometry(const GBufferData data, const ushort2 index, const uint32_t light_ray_index) {
  if (!device.scene.material.lights_active)
    return get_color(0.0f, 0.0f, 0.0f);

  vec3 dir;
  RGBF light_color;
  float dist;
  const uint32_t light_id = ris_sample_light(data, index, light_ray_index, dir, light_color, dist);

  if (luminance(light_color) == 0.0f || light_id == LIGHT_ID_NONE)
    return get_color(0.0f, 0.0f, 0.0f);

  const float3 origin = make_float3(data.position.x, data.position.y, data.position.z);
  const float3 ray    = make_float3(dir.x, dir.y, dir.z);

  unsigned int hit_id = light_id;

  // 21 bits for each color component.
  unsigned int alpha_data0, alpha_data1;
  optix_compress_color(get_color(1.0f, 1.0f, 1.0f), alpha_data0, alpha_data1);

  unsigned int compressed_ior = ior_compress((data.flags & G_BUFFER_IS_TRANSPARENT_PASS) ? data.ior_out : data.ior_in);

  if (device.scene.toy.active) {
    const float toy_dist = get_toy_distance(data.position, dir);

    if (toy_dist < dist) {
      // TODO: This only works when we enter a surface, what about the exit???
      if (device.scene.material.enable_ior_shadowing && ior_compress(device.scene.toy.refractive_index) != compressed_ior)
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

  if (device.scene.material.enable_ior_shadowing && compressed_ior != __uint_as_float(optixGetPayload_3())) {
    // Terminate ray.
    return get_RGBAF(0.0f, 0.0f, 0.0f, 1.0f);
  }

  const uint16_t tex = __ldg(&(device.scene.materials[material_id].albedo_map));

  if (tex != TEXTURE_NONE) {
    const UV uv = load_triangle_tex_coords(hit_id, optixGetTriangleBarycentrics());

    const float4 tex_value = tex2D<float4>(device.ptrs.albedo_atlas[tex].tex, uv.u, 1.0f - uv.v);

    RGBAF albedo;
    albedo.r = tex_value.x;
    albedo.g = tex_value.y;
    albedo.b = tex_value.z;
    albedo.a = tex_value.w;

    return albedo;
  }

  RGBAF albedo;
  albedo.r = random_uint16_t_to_float(__ldg(&(device.scene.materials[material_id].albedo_r)));
  albedo.g = random_uint16_t_to_float(__ldg(&(device.scene.materials[material_id].albedo_g)));
  albedo.b = random_uint16_t_to_float(__ldg(&(device.scene.materials[material_id].albedo_b)));
  albedo.a = random_uint16_t_to_float(__ldg(&(device.scene.materials[material_id].albedo_a)));

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
