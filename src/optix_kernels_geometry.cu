// More ideas:
// - Add a ray flag for if MIS is disabled (would be set for bounce rays coming from ocean surface): STATE_FLAG_MIS_DISABLED

#define UTILS_NO_DEVICE_TABLE

// Functions work differently when executed from this kernel
// This emulates the old device.iteration_type == TYPE_LIGHT checks.
#define SHADING_KERNEL
#define OPTIX_KERNEL

#include "utils.h"

extern "C" static __constant__ DeviceConstantMemory device;

#include "bsdf.cuh"
#include "directives.cuh"
#include "geometry.cuh"
#include "ior_stack.cuh"
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

  if (solid_angle == 0.0f)
    return get_color(0.0f, 0.0f, 0.0f);

  bool is_transparent_pass;
  RGBF bsdf_value = bsdf_evaluate(data, dir, BSDF_SAMPLING_GENERAL, is_transparent_pass, solid_angle * light_sampling_weight);

  // TODO: Fix transparent ray directions. (They are currently causing a massive amount of fireflies.)
  if (is_transparent_pass)
    return get_color(0.0f, 0.0f, 0.0f);

  light_color = mul_color(light_color, bsdf_value);

  const float shift           = (is_transparent_pass) ? -eps : eps;
  const vec3 shifted_position = add_vector(data.position, scale_vector(data.V, shift * get_length(data.position)));

  const float3 origin = make_float3(shifted_position.x, shifted_position.y, shifted_position.z);
  const float3 ray    = make_float3(dir.x, dir.y, dir.z);

  unsigned int hit_id = light_id;

  // 21 bits for each color component.
  unsigned int alpha_data0, alpha_data1;
  optix_compress_color(get_color(1.0f, 1.0f, 1.0f), alpha_data0, alpha_data1);

  // Disable OMM opaque hits because we want to know if we hit something that is fully opaque so we can reject.
  optixTrace(
    device.optix_bvh, origin, ray, 0.0f, dist, 0.0f, OptixVisibilityMask(0xFFFF), OPTIX_RAY_FLAG_ENFORCE_ANYHIT, 0, 0, 0, hit_id,
    alpha_data0, alpha_data1);

  RGBF visibility = (hit_id != HIT_TYPE_REJECT) ? optix_decompress_color(alpha_data0, alpha_data1) : get_color(0.0f, 0.0f, 0.0f);

  return mul_color(light_color, visibility);
}

extern "C" __global__ void __raygen__optix() {
  const int task_count  = device.ptrs.task_counts[THREAD_ID * TASK_ADDRESS_COUNT_STRIDE + TASK_ADDRESS_OFFSET_GEOMETRY];
  const int task_offset = device.ptrs.task_offsets[THREAD_ID * TASK_ADDRESS_OFFSET_STRIDE + TASK_ADDRESS_OFFSET_GEOMETRY];
  int trace_count       = device.ptrs.trace_counts[THREAD_ID];

  for (int i = 0; i < task_count; i++) {
    GeometryTask task = load_geometry_task(device.ptrs.trace_tasks + get_task_address(task_offset + i));
    const int pixel   = task.index.y * device.width + task.index.x;

    GBufferData data = geometry_generate_g_buffer(task, pixel);

    write_normal_buffer(data.normal, pixel);

    if (!material_is_mirror(data.roughness, data.metallic))
      write_albedo_buffer(opaque_color(data.albedo), pixel);

    RGBF accumulated_light   = get_color(0.0f, 0.0f, 0.0f);
    uint32_t light_ray_index = 0;

    if (device.restir.num_light_rays) {
      for (int j = 0; j < device.restir.num_light_rays; j++) {
        accumulated_light =
          add_color(accumulated_light, optix_compute_light_ray(data, task.index, LIGHT_RAY_TARGET_GEOMETRY, light_ray_index++));
      }

      accumulated_light = scale_color(accumulated_light, 1.0f / device.restir.num_light_rays);
    }

    accumulated_light = add_color(accumulated_light, optix_compute_light_ray(data, task.index, LIGHT_RAY_TARGET_SUN, light_ray_index++));
    accumulated_light = add_color(accumulated_light, optix_compute_light_ray(data, task.index, LIGHT_RAY_TARGET_TOY, light_ray_index++));

    const RGBF record = load_RGBF(device.ptrs.records + pixel);

    if (state_peek(pixel, STATE_FLAG_BOUNCE_LIGHTING))
      accumulated_light = add_color(accumulated_light, data.emission);

    accumulated_light = mul_color(accumulated_light, record);

    write_beauty_buffer(accumulated_light, pixel);

    BSDFSampleInfo bounce_info;
    float bsdf_marginal;
    vec3 bounce_ray = bsdf_sample(data, task.index, bounce_info, bsdf_marginal);

    RGBF bounce_record = mul_color(record, bounce_info.weight);

    const float shift = (bounce_info.is_transparent_pass) ? -eps : eps;
    data.position     = add_vector(data.position, scale_vector(data.V, shift * get_length(data.position)));

    if (bounce_info.is_transparent_pass) {
      const IORStackMethod ior_stack_method = (data.flags & G_BUFFER_REFRACTION_IS_INSIDE) ? IOR_STACK_METHOD_PULL : IOR_STACK_METHOD_PUSH;
      ior_stack_interact(data.ior_in, pixel, ior_stack_method);
    }

    TraceTask bounce_task;
    bounce_task.origin = data.position;
    bounce_task.ray    = bounce_ray;
    bounce_task.index  = task.index;

    if (validate_trace_task(bounce_task, bounce_record)) {
      store_RGBF(device.ptrs.records + pixel, bounce_record);
      store_trace_task(device.ptrs.trace_tasks + get_task_address(trace_count++), bounce_task);

      state_release(pixel, STATE_FLAG_BOUNCE_LIGHTING);
    }
  }

  device.ptrs.trace_counts[THREAD_ID] = trace_count;
}

/*
 * Performs alpha test on triangle
 * @result 0 if opaque, 1 if transparent, 2 if alpha cutoff
 */
__device__ RGBAF optix_alpha_test() {
  const unsigned int hit_id = optixGetPrimitiveIndex();

  const uint32_t material_id = load_triangle_material_id(hit_id);
  const uint16_t tex         = __ldg(&(device.scene.materials[material_id].albedo_map));

  RGBAF albedo = get_RGBAF(0.0f, 0.0f, 0.0f, 1.0f);

  if (tex != TEXTURE_NONE) {
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
  // Dummy closest hit, this will never get executed anyway due to the anyhit.
  // I could maybe get rid of this by adding more logic during the optix kernel compilation.
}
