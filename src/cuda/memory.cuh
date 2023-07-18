#ifndef CU_MEMORY_H
#define CU_MEMORY_H

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
  task.ray_y      = data1.x;

  task.ray_xz = data1.y;
  task.hit_id = __float_as_uint(data1.z);

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
  task.ray_y      = data1.x;
  task.ray_xz     = data1.y;
  task.distance   = data1.z;

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
  task.index.x     = __float_as_uint(data0.x) & 0xffff;
  task.index.y     = (__float_as_uint(data0.x) >> 16);
  task.position.x  = data0.y;
  task.position.y  = data0.z;
  task.position.z  = data0.w;
  task.ray_y       = data1.x;
  task.ray_xz      = data1.y;
  task.distance    = data1.z;
  task.volume_type = __float_as_uint(data1.w);

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
  if (!device.denoiser || device.ptrs.state_buffer[pixel] & STATE_ALBEDO || device.iteration_type == TYPE_LIGHT)
    return;

  if (device.temporal_frames && device.accum_mode == TEMPORAL_ACCUMULATION) {
    RGBF out_albedo = RGBAhalf_to_RGBF(device.ptrs.albedo_buffer[pixel]);
    out_albedo      = scale_color(out_albedo, device.temporal_frames);
    albedo          = add_color(albedo, out_albedo);
    albedo          = scale_color(albedo, 1.0f / (device.temporal_frames + 1));
  }

  device.ptrs.albedo_buffer[pixel] = RGBF_to_RGBAhalf(albedo);
  device.ptrs.state_buffer[pixel] |= STATE_ALBEDO;
}

__device__ void write_normal_buffer(vec3 normal, const int pixel) {
  if (!device.denoiser || device.iteration_type != TYPE_CAMERA || (device.temporal_frames && device.accum_mode == TEMPORAL_ACCUMULATION))
    return;

  normal = transform_vec4_3(device.emitter.view_space, normal);

  const float normal_norm = sqrtf(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);

  if (normal_norm > eps) {
    normal = scale_vector(normal, 1.0f / normal_norm);
  }

  device.ptrs.normal_buffer[pixel] = get_RGBAhalf(normal.x, normal.y, normal.z, 0.0f);
}

__device__ LightEvalData load_light_eval_data(const int offset) {
  const float4* ptr  = (float4*) (device.ptrs.light_eval_data + offset);
  const float4 data0 = __ldcs(ptr + 0);
  const float4 data1 = __ldcs(ptr + 1);
  const float4 data2 = __ldcs(ptr + 2);

  LightEvalData result;
  result.position  = get_vector(data0.x, data0.y, data0.z);
  result.V         = get_vector(data0.w, data1.x, data1.y);
  result.normal    = get_vector(data1.z, data1.w, data2.x);
  result.roughness = data2.y;
  result.metallic  = data2.z;
  result.flags     = __float_as_uint(data2.w);

  return result;
}

__device__ void store_light_eval_data(const LightEvalData data, const int offset) {
  float4 data0, data1, data2;

  data0.x = data.position.x;
  data0.y = data.position.y;
  data0.z = data.position.z;
  data0.w = data.V.x;
  data1.x = data.V.y;
  data1.y = data.V.z;
  data1.z = data.normal.x;
  data1.w = data.normal.y;
  data2.x = data.normal.z;
  data2.y = data.roughness;
  data2.z = data.metallic;
  data2.w = __uint_as_float(data.flags);

  float4* ptr = (float4*) (device.ptrs.light_eval_data + offset);
  __stcs(ptr + 0, data0);
  __stcs(ptr + 1, data1);
  __stcs(ptr + 2, data2);
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

__device__ UV load_triangle_tex_coords(const int offset, const float2 coords) {
  const float4* ptr      = (float4*) (device.scene.triangles + offset);
  const float2 bytes0x48 = __ldg(((float2*) (ptr + 4)) + 1);
  const float4 bytes0x50 = __ldg(ptr + 5);

  const UV vertex_texture = get_UV(bytes0x48.x, bytes0x48.y);
  const UV edge1_texture  = get_UV(bytes0x50.x, bytes0x50.y);
  const UV edge2_texture  = get_UV(bytes0x50.z, bytes0x50.w);

  return lerp_uv(vertex_texture, edge1_texture, edge2_texture, coords);
}

__device__ TriangleLight load_triangle_light(const TriangleLight* data, const int offset) {
  const float4* ptr = (float4*) (data + offset);
  const float4 v1   = __ldg(ptr);
  const float4 v2   = __ldg(ptr + 1);
  const float2 v3   = __ldg((float2*) (ptr + 2));

  TriangleLight triangle;
  triangle.vertex      = get_vector(v1.x, v1.y, v1.z);
  triangle.edge1       = get_vector(v1.w, v2.x, v2.y);
  triangle.edge2       = get_vector(v2.z, v2.w, v3.x);
  triangle.triangle_id = __float_as_uint(v3.y);

  return triangle;
}

#endif /* CU_MEMORY_H */
