#ifndef CU_MEMORY_H
#define CU_MEMORY_H

#include "state.cuh"
#include "utils.cuh"

//===========================================================================================
// Memory Prefetch Functions
//===========================================================================================

__device__ void __prefetch_global_l1(const void* const ptr) {
  asm("prefetch.global.L1 [%0];" : : "l"(ptr));
}

__device__ void __prefetch_global_l2(const void* const ptr) {
  asm("prefetch.global.L2 [%0];" : : "l"(ptr));
}

//===========================================================================================
// Minimal Cache Pollution Loads/Stores
//===========================================================================================

__device__ void stream_float2(const float2* source, float2* target) {
  __stcs(target, __ldcs(source));
}

__device__ void stream_float4(const float4* source, float4* target) {
  __stcs(target, __ldcs(source));
}

__device__ void swap_trace_data(const int index0, const int index1) {
  const int offset0  = get_task_address(index0);
  const float2 temp  = __ldca((float2*) (device.ptrs.trace_results + offset0));
  const float4 data0 = __ldcs((float4*) (device.trace_tasks + offset0));
  const float4 data1 = __ldcs((float4*) (device.trace_tasks + offset0) + 1);

  const int offset1 = get_task_address(index1);
  stream_float2((float2*) (device.ptrs.trace_results + offset1), (float2*) (device.ptrs.trace_results + offset0));
  stream_float4((float4*) (device.trace_tasks + offset1), (float4*) (device.trace_tasks + offset0));
  stream_float4((float4*) (device.trace_tasks + offset1) + 1, (float4*) (device.trace_tasks + offset0) + 1);
  __stcs((float2*) (device.ptrs.trace_results + offset1), temp);
  __stcs((float4*) (device.trace_tasks + offset1), data0);
  __stcs((float4*) (device.trace_tasks + offset1) + 1, data1);
}

__device__ TraceTask load_trace_task(const void* ptr) {
  const float4 data0 = __ldcs((float4*) ptr);
  const float4 data1 = __ldcs(((float4*) ptr) + 1);

  TraceTask task;
  task.origin.x = data0.x;
  task.origin.y = data0.y;
  task.origin.z = data0.z;
  task.ray.x    = data0.w;

  task.ray.y   = data1.x;
  task.ray.z   = data1.y;
  task.index.x = __float_as_uint(data1.z) & 0xffff;
  task.index.y = (__float_as_uint(data1.z) >> 16);

  return task;
}

__device__ void store_trace_task(const void* ptr, const TraceTask task) {
  float4 data0;
  data0.x = task.origin.x;
  data0.y = task.origin.y;
  data0.z = task.origin.z;
  data0.w = task.ray.x;
  __stcs((float4*) ptr, data0);

  float4 data1;
  data1.x = task.ray.y;
  data1.y = task.ray.z;
  data1.z = __uint_as_float(((uint32_t) task.index.x & 0xffff) | ((uint32_t) task.index.y << 16));
  __stcs(((float4*) ptr) + 1, data1);
}

__device__ TraceTask load_trace_task_essentials(const void* ptr) {
  const float4 data0 = __ldcs((float4*) ptr);
  const float2 data1 = __ldcs(((float2*) ptr) + 2);

  TraceTask task;
  task.origin.x = data0.x;
  task.origin.y = data0.y;
  task.origin.z = data0.z;
  task.ray.x    = data0.w;

  task.ray.y = data1.x;
  task.ray.z = data1.y;

  return task;
}

__device__ GeometryTask load_geometry_task(const void* ptr) {
  const float4 data0 = __ldcs((float4*) ptr);
  const float4 data1 = __ldcs(((float4*) ptr) + 1);

  GeometryTask task;
  task.index.x    = __float_as_uint(data0.x) & 0xffff;
  task.index.y    = (__float_as_uint(data0.x) >> 16);
  task.position.x = data0.y;
  task.position.y = data0.z;
  task.position.z = data0.w;

  task.ray.x  = data1.x;
  task.ray.y  = data1.y;
  task.ray.z  = data1.z;
  task.hit_id = __float_as_uint(data1.w);

  return task;
}

__device__ ParticleTask load_particle_task(const void* ptr) {
  const float4 data0 = __ldcs((float4*) ptr);
  const float4 data1 = __ldcs(((float4*) ptr) + 1);

  ParticleTask task;
  task.index.x    = __float_as_uint(data0.x) & 0xffff;
  task.index.y    = (__float_as_uint(data0.x) >> 16);
  task.position.x = data0.y;
  task.position.y = data0.z;
  task.position.z = data0.w;

  task.ray.x  = data1.x;
  task.ray.y  = data1.y;
  task.ray.z  = data1.z;
  task.hit_id = __float_as_uint(data1.w);

  return task;
}

__device__ OceanTask load_ocean_task(const void* ptr) {
  const float4 data0 = __ldcs((float4*) ptr);
  const float4 data1 = __ldcs(((float4*) ptr) + 1);

  OceanTask task;
  task.index.x    = __float_as_uint(data0.x) & 0xffff;
  task.index.y    = (__float_as_uint(data0.x) >> 16);
  task.position.x = data0.y;
  task.position.y = data0.z;
  task.position.z = data0.w;
  task.ray.x      = data1.x;
  task.ray.y      = data1.y;
  task.ray.z      = data1.z;

  return task;
}

__device__ SkyTask load_sky_task(const void* ptr) {
  const float4 data0 = __ldcs((float4*) ptr);
  const float4 data1 = __ldcs(((float4*) ptr) + 1);

  SkyTask task;
  task.index.x  = __float_as_uint(data0.x) & 0xffff;
  task.index.y  = (__float_as_uint(data0.x) >> 16);
  task.origin.x = data0.y;
  task.origin.y = data0.z;
  task.origin.z = data0.w;
  task.ray.x    = data1.x;
  task.ray.y    = data1.y;
  task.ray.z    = data1.z;

  return task;
}

__device__ ToyTask load_toy_task(const void* ptr) {
  const float4 data0 = __ldcs((float4*) ptr);
  const float4 data1 = __ldcs(((float4*) ptr) + 1);

  ToyTask task;
  task.index.x    = __float_as_uint(data0.x) & 0xffff;
  task.index.y    = (__float_as_uint(data0.x) >> 16);
  task.position.x = data0.y;
  task.position.y = data0.z;
  task.position.z = data0.w;
  task.ray.x      = data1.x;
  task.ray.y      = data1.y;
  task.ray.z      = data1.z;

  return task;
}

__device__ VolumeTask load_volume_task(const void* ptr) {
  const float4 data0 = __ldcs((float4*) ptr);
  const float4 data1 = __ldcs(((float4*) ptr) + 1);

  VolumeTask task;
  task.index.x    = __float_as_uint(data0.x) & 0xffff;
  task.index.y    = (__float_as_uint(data0.x) >> 16);
  task.position.x = data0.y;
  task.position.y = data0.z;
  task.position.z = data0.w;
  task.ray.x      = data1.x;
  task.ray.y      = data1.y;
  task.ray.z      = data1.z;
  task.hit_id     = __float_as_uint(data1.w);

  return task;
}

__device__ RGBAhalf load_RGBAhalf(const void* ptr) {
  const ushort4 data0 = __ldcs((ushort4*) ptr);

  RGBAhalf result;
  result.rg.x = __ushort_as_half(data0.x);
  result.rg.y = __ushort_as_half(data0.y);
  result.ba.x = __ushort_as_half(data0.z);
  result.ba.y = __ushort_as_half(data0.w);

  return result;
}

__device__ void store_RGBAhalf(void* ptr, const RGBAhalf a) {
  ushort4 data0 = make_ushort4(__half_as_ushort(a.rg.x), __half_as_ushort(a.rg.y), __half_as_ushort(a.ba.x), __half_as_ushort(a.ba.y));

  __stcs((ushort4*) ptr, data0);
}

__device__ RGBF load_RGBF(const void* ptr) {
  return *((RGBF*) ptr);
}

__device__ void store_RGBF(void* ptr, const RGBF a) {
  *((RGBF*) ptr) = a;
}

/*
 * Updates the albedo buffer if criteria are met.
 * @param albedo Albedo color to be added to the albedo buffer.
 * @param pixel Index of pixel.
 */
__device__ void write_albedo_buffer(RGBF albedo, const int pixel) {
  if ((!device.denoiser && !device.aov_mode) || device.iteration_type == TYPE_LIGHT)
    return;

  if (state_consume(pixel, STATE_FLAG_ALBEDO)) {
    if (device.temporal_frames && device.accum_mode == TEMPORAL_ACCUMULATION) {
      RGBF out_albedo = device.ptrs.albedo_buffer[pixel];
      out_albedo      = scale_color(out_albedo, device.temporal_frames);
      albedo          = add_color(albedo, out_albedo);
      albedo          = scale_color(albedo, 1.0f / (device.temporal_frames + 1));
    }

    device.ptrs.albedo_buffer[pixel] = albedo;
  }
}

__device__ void write_normal_buffer(vec3 normal, const int pixel) {
  if ((!device.denoiser && !device.aov_mode) || device.iteration_type != TYPE_CAMERA)
    return;

  if (device.temporal_frames && device.accum_mode == TEMPORAL_ACCUMULATION)
    return;

  const float normal_norm = sqrtf(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);

  if (normal_norm > eps) {
    normal = scale_vector(normal, 1.0f / normal_norm);
  }

  device.ptrs.normal_buffer[pixel] = get_color(normal.x, normal.y, normal.z);
}

__device__ void write_beauty_buffer(RGBF beauty, const int pixel, bool mode_set = false) {
  RGBF output = beauty;
  if (!mode_set) {
    output = add_color(beauty, load_RGBF(device.ptrs.frame_buffer + pixel));
  }
  store_RGBF(device.ptrs.frame_buffer + pixel, output);

  if (device.aov_mode) {
    if (device.depth <= 1) {
      RGBF output = beauty;
      if (!mode_set) {
        output = add_color(beauty, load_RGBF(device.ptrs.frame_direct_buffer + pixel));
      }
      store_RGBF(device.ptrs.frame_direct_buffer + pixel, output);
    }
    else {
      RGBF output = beauty;
      if (!mode_set) {
        output = add_color(beauty, load_RGBF(device.ptrs.frame_indirect_buffer + pixel));
      }
      store_RGBF(device.ptrs.frame_indirect_buffer + pixel, output);
    }
  }
}

__device__ LightSample load_light_sample(const LightSample* ptr, const int offset) {
  const float4 packet = __ldcs((float4*) (ptr + offset));

  LightSample sample;
  sample.seed          = __float_as_uint(packet.x);
  sample.presampled_id = __float_as_uint(packet.y);
  sample.id            = __float_as_uint(packet.z);
  sample.weight        = packet.w;

  return sample;
}

__device__ void store_light_sample(LightSample* ptr, const LightSample sample, const int offset) {
  float4 packet;
  packet.x = __uint_as_float(sample.seed);
  packet.y = __uint_as_float(sample.presampled_id);
  packet.z = __uint_as_float(sample.id);
  packet.w = sample.weight;

  __stcs((float4*) (ptr + offset), packet);
}

__device__ TraversalTriangle load_traversal_triangle(const int offset) {
  const float4* ptr = (float4*) (device.bvh_triangles + offset);
  const float4 v1   = __ldg(ptr);
  const float4 v2   = __ldg(ptr + 1);
  const float4 v3   = __ldg(ptr + 2);

  TraversalTriangle triangle;
  triangle.vertex     = get_vector(v1.x, v1.y, v1.z);
  triangle.edge1      = get_vector(v1.w, v2.x, v2.y);
  triangle.edge2      = get_vector(v2.z, v2.w, v3.x);
  triangle.albedo_tex = __float_as_uint(v3.y);
  triangle.id         = __float_as_uint(v3.z);

  return triangle;
}

__device__ const void* interleaved_buffer_get_entry_address(
  const void* ptr, const uint32_t count, const uint32_t chunk, const uint32_t offset, const uint32_t id) {
  return (const void*) (((const float*) ptr) + (count * chunk + id) * 4 + offset);
}

__device__ const void* pixel_buffer_get_entry_address(const void* ptr, const uint32_t chunk, const uint32_t offset, const uint32_t id) {
  return interleaved_buffer_get_entry_address(ptr, device.width * device.height, chunk, offset, id);
}

__device__ const void* triangle_get_entry_address(const uint32_t chunk, const uint32_t offset, const uint32_t id) {
  return interleaved_buffer_get_entry_address(device.scene.triangles, device.scene.triangle_data.triangle_count, chunk, offset, id);
}

__device__ UV load_triangle_tex_coords(const int offset, const float2 coords) {
  const float2 bytes0x48 = __ldg((float2*) triangle_get_entry_address(4, 2, offset));
  const float4 bytes0x50 = __ldg((float4*) triangle_get_entry_address(5, 0, offset));

  const UV vertex_texture = get_UV(bytes0x48.x, bytes0x48.y);
  const UV edge1_texture  = get_UV(bytes0x50.x, bytes0x50.y);
  const UV edge2_texture  = get_UV(bytes0x50.z, bytes0x50.w);

  return lerp_uv(vertex_texture, edge1_texture, edge2_texture, coords);
}

__device__ uint32_t load_triangle_material_id(const uint32_t id) {
  const uint32_t* triangles_material_ids = ((uint32_t*) device.scene.triangles) + device.scene.triangle_data.triangle_count * 6 * 4;
  return __ldg(triangles_material_ids + id);
}

__device__ uint32_t load_triangle_light_id(const uint32_t id) {
  const uint32_t* triangles_light_ids = ((uint32_t*) device.scene.triangles) + device.scene.triangle_data.triangle_count * (6 * 4 + 1);
  return __ldg(triangles_light_ids + id);
}

__device__ TriangleLight load_triangle_light(const TriangleLight* data, const int offset) {
  const float4* ptr = (float4*) (data + offset);
  const float4 v1   = __ldg(ptr);
  const float4 v2   = __ldg(ptr + 1);
  const float4 v3   = __ldg(ptr + 2);

  TriangleLight triangle;
  triangle.vertex      = get_vector(v1.x, v1.y, v1.z);
  triangle.edge1       = get_vector(v1.w, v2.x, v2.y);
  triangle.edge2       = get_vector(v2.z, v2.w, v3.x);
  triangle.triangle_id = __float_as_uint(v3.y);
  triangle.material_id = __float_as_uint(v3.z);

  return triangle;
}

__device__ Quad load_quad(const Quad* data, const int offset) {
  const float4* ptr = (float4*) (data + offset);
  const float4 v1   = __ldg(ptr);
  const float4 v2   = __ldg(ptr + 1);
  const float4 v3   = __ldg(ptr + 2);

  Quad quad;
  quad.vertex = get_vector(v1.x, v1.y, v1.z);
  quad.edge1  = get_vector(v1.w, v2.x, v2.y);
  quad.edge2  = get_vector(v2.z, v2.w, v3.x);
  quad.normal = get_vector(v3.y, v3.z, v3.w);

  return quad;
}

__device__ Material load_material(const PackedMaterial* data, const int offset) {
  const float4* ptr = (float4*) (data + offset);
  const float4 v0   = __ldg(ptr + 0);
  const float4 v1   = __ldg(ptr + 1);

  Material mat;
  mat.refraction_index = v0.x;
  mat.albedo.r         = random_uint16_t_to_float(__float_as_uint(v0.y) & 0xFFFF);
  mat.albedo.g         = random_uint16_t_to_float(__float_as_uint(v0.y) >> 16);
  mat.albedo.b         = random_uint16_t_to_float(__float_as_uint(v0.z) & 0xFFFF);
  mat.albedo.a         = random_uint16_t_to_float(__float_as_uint(v0.z) >> 16);
  mat.emission.r       = random_uint16_t_to_float(__float_as_uint(v0.w) & 0xFFFF);
  mat.emission.g       = random_uint16_t_to_float(__float_as_uint(v0.w) >> 16);
  mat.emission.b       = random_uint16_t_to_float(__float_as_uint(v1.x) & 0xFFFF);
  float emission_scale = (float) (__float_as_uint(v1.x) >> 16);
  mat.metallic         = random_uint16_t_to_float(__float_as_uint(v1.y) & 0xFFFF);
  mat.roughness        = random_uint16_t_to_float(__float_as_uint(v1.y) >> 16);
  mat.albedo_map       = __float_as_uint(v1.z) & 0xFFFF;
  mat.luminance_map    = __float_as_uint(v1.z) >> 16;
  mat.material_map     = __float_as_uint(v1.w) & 0xFFFF;
  mat.normal_map       = __float_as_uint(v1.w) >> 16;

  mat.emission = scale_color(mat.emission, emission_scale);

  return mat;
}

__device__ void store_gbuffer_data(const GBufferData data, const int pixel) {
  PackedGBufferData* ptr = device.ptrs.packed_gbuffer_history;

  float4 bytes0x00;
  bytes0x00.x = __uint_as_float(data.hit_id);
  bytes0x00.y = ((uint32_t) (data.albedo.r * 0xFFFF + 0.5f)) | (((uint32_t) (data.albedo.g * 0xFFFF + 0.5f)) << 16);
  bytes0x00.z = ((uint32_t) (data.albedo.b * 0xFFFF + 0.5f)) | (((uint32_t) (data.albedo.a * 0xFFFF + 0.5f)) << 16);
  bytes0x00.w = ((uint32_t) (data.roughness * 0xFFFF + 0.5f)) | (((uint32_t) (data.metallic * 0xFFFF + 0.5f)) << 16);

  __stcs((float4*) pixel_buffer_get_entry_address(ptr, 0, 0, pixel), bytes0x00);

  float4 bytes0x10;
  bytes0x10.x = data.position.x;
  bytes0x10.y = data.position.y;
  bytes0x10.z = data.position.z;
  bytes0x10.w = data.V.x;

  __stcs((float4*) pixel_buffer_get_entry_address(ptr, 1, 0, pixel), bytes0x10);

  float4 bytes0x20;
  bytes0x20.x = data.V.y;
  bytes0x20.y = data.V.z;
  bytes0x20.z = data.normal.x;
  bytes0x20.w = data.normal.y;

  __stcs((float4*) pixel_buffer_get_entry_address(ptr, 2, 0, pixel), bytes0x20);

  float2 bytes0x30;
  bytes0x30.x = data.normal.z;
  bytes0x30.y = __uint_as_float(data.flags);

  __stcs((float2*) pixel_buffer_get_entry_address(ptr, 3, 0, pixel), bytes0x30);

  float bytes0x38;
  bytes0x38 = ((uint32_t) (data.ior_in * 0xFFFF + 0.5f)) | (((uint32_t) (data.ior_out * 0xFFFF + 0.5f)) << 16);

  __stcs((float*) pixel_buffer_get_entry_address(ptr, 3, 8, pixel), bytes0x38);
}

__device__ GBufferData load_gbuffer_data(const int pixel) {
  const PackedGBufferData* ptr = device.ptrs.packed_gbuffer_history;

  const float4 bytes0x00 = __ldcs((float4*) pixel_buffer_get_entry_address(ptr, 0, 0, pixel));
  const float4 bytes0x10 = __ldcs((float4*) pixel_buffer_get_entry_address(ptr, 1, 0, pixel));
  const float4 bytes0x20 = __ldcs((float4*) pixel_buffer_get_entry_address(ptr, 2, 0, pixel));
  const float2 bytes0x30 = __ldcs((float2*) pixel_buffer_get_entry_address(ptr, 3, 0, pixel));
  const float bytes0x38  = __ldcs((float*) pixel_buffer_get_entry_address(ptr, 3, 8, pixel));

  GBufferData data;
  data.hit_id    = __float_as_uint(bytes0x00.x);
  data.albedo.r  = (__float_as_uint(bytes0x00.y) & 0xFFFF) * (1.0f / 0xFFFF);
  data.albedo.g  = (__float_as_uint(bytes0x00.y) >> 16) * (1.0f / 0xFFFF);
  data.albedo.b  = (__float_as_uint(bytes0x00.z) & 0xFFFF) * (1.0f / 0xFFFF);
  data.albedo.a  = (__float_as_uint(bytes0x00.z) >> 16) * (1.0f / 0xFFFF);
  data.roughness = (__float_as_uint(bytes0x00.w) & 0xFFFF) * (1.0f / 0xFFFF);
  data.metallic  = (__float_as_uint(bytes0x00.w) >> 16) * (1.0f / 0xFFFF);
  data.position  = get_vector(bytes0x10.x, bytes0x10.y, bytes0x10.z);
  data.V         = get_vector(bytes0x10.w, bytes0x20.x, bytes0x20.y);
  data.normal    = get_vector(bytes0x20.z, bytes0x20.w, bytes0x30.x);
  data.flags     = __float_as_uint(bytes0x30.y);
  data.ior_in    = (__float_as_uint(bytes0x38) & 0xFFFF) * (1.0f / 0xFFFF);
  data.ior_out   = (__float_as_uint(bytes0x38) >> 16) * (1.0f / 0xFFFF);

  data.colored_dielectric = (data.hit_id <= LIGHT_ID_TRIANGLE_ID_LIMIT) ? device.scene.material.colored_transparency : 1;

  data.emission = get_color(0.0f, 0.0f, 0.0f);

  return data;
}

__device__ void store_mis_data(const MISData data, const int pixel) {
  float2 bytes;
  bytes.x = data.light_target_pdf_normalization;
  bytes.y = data.bsdf_marginal;

  __stcs((float2*) (device.ptrs.mis_buffer + pixel), bytes);
}

__device__ MISData load_mis_data(const int pixel) {
  float2 bytes = __ldcs((float2*) (device.ptrs.mis_buffer + pixel));

  MISData data;
  data.light_target_pdf_normalization = bytes.x;
  data.bsdf_marginal                  = bytes.y;

  return data;
}

#endif /* CU_MEMORY_H */
