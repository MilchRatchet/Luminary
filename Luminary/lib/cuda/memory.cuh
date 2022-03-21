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
  const float2 temp  = __ldca((float2*) (device.trace_results + offset0));
  const float4 data0 = __ldcs((float4*) (device_trace_tasks + offset0));
  const float4 data1 = __ldcs((float4*) (device_trace_tasks + offset0) + 1);

  const int offset1 = get_task_address(index1);
  stream_float2((float2*) (device.trace_results + offset1), (float2*) (device.trace_results + offset0));
  stream_float4((float4*) (device_trace_tasks + offset1), (float4*) (device_trace_tasks + offset0));
  stream_float4((float4*) (device_trace_tasks + offset1) + 1, (float4*) (device_trace_tasks + offset0) + 1);
  __stcs((float2*) (device.trace_results + offset1), temp);
  __stcs((float4*) (device_trace_tasks + offset1), data0);
  __stcs((float4*) (device_trace_tasks + offset1) + 1, data1);
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
  task.state   = __float_as_uint(data1.w);

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
  data1.z = __uint_as_float((task.index.x & 0xffff) | (task.index.y << 16));
  data1.w = __uint_as_float(task.state);
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
  task.position.x = data0.x;
  task.position.y = data0.y;
  task.position.z = data0.z;
  task.ray_y      = data0.w;

  task.ray_xz  = data1.x;
  task.hit_id  = __float_as_uint(data1.y);
  task.index.x = __float_as_uint(data1.z) & 0xffff;
  task.index.y = (__float_as_uint(data1.z) >> 16);
  task.state   = __float_as_uint(data1.w);

  return task;
}

__device__ OceanTask load_ocean_task(const void* ptr) {
  const float4 data0 = __ldcs((float4*) ptr);
  const float4 data1 = __ldcs(((float4*) ptr) + 1);

  OceanTask task;
  task.position.x = data0.x;
  task.position.y = data0.y;
  task.position.z = data0.z;
  task.ray_y      = data0.w;

  task.ray_xz   = data1.x;
  task.distance = data1.y;
  task.index.x  = __float_as_uint(data1.z) & 0xffff;
  task.index.y  = (__float_as_uint(data1.z) >> 16);
  task.state    = __float_as_uint(data1.w);

  return task;
}

__device__ SkyTask load_sky_task(const void* ptr) {
  const float4 data0 = __ldcs((float4*) ptr);
  const float4 data1 = __ldcs(((float4*) ptr) + 1);

  SkyTask task;
  task.origin.x = data0.x;
  task.origin.y = data0.y;
  task.origin.z = data0.z;
  task.ray.x    = data0.w;
  task.ray.y    = data1.x;
  task.ray.z    = data1.y;
  task.index.x  = __float_as_uint(data1.z) & 0xffff;
  task.index.y  = (__float_as_uint(data1.z) >> 16);
  task.state    = __float_as_uint(data1.w);

  return task;
}

__device__ ToyTask load_toy_task(const void* ptr) {
  const float4 data0 = __ldcs((float4*) ptr);
  const float4 data1 = __ldcs(((float4*) ptr) + 1);

  ToyTask task;
  task.position.x = data0.x;
  task.position.y = data0.y;
  task.position.z = data0.z;
  task.ray.x      = data0.w;

  task.ray.y   = data1.x;
  task.ray.z   = data1.y;
  task.index.x = __float_as_uint(data1.z) & 0xffff;
  task.index.y = (__float_as_uint(data1.z) >> 16);
  task.state   = __float_as_uint(data1.w);

  return task;
}

__device__ FogTask load_fog_task(const void* ptr) {
  const float4 data0 = __ldcs((float4*) ptr);
  const float4 data1 = __ldcs(((float4*) ptr) + 1);

  FogTask task;
  task.position.x = data0.x;
  task.position.y = data0.y;
  task.position.z = data0.z;
  task.ray_y      = data0.w;

  task.ray_xz   = data1.x;
  task.distance = data1.y;
  task.index.x  = __float_as_uint(data1.z) & 0xffff;
  task.index.y  = (__float_as_uint(data1.z) >> 16);
  task.state    = __float_as_uint(data1.w);

  return task;
}

/*
 * Updates the albedo buffer if criteria are met.
 * @param albedo Albedo color to be added to the albedo buffer.
 * @param pixel Index of pixel.
 */
__device__ void write_albedo_buffer(RGBF albedo, const int pixel) {
  if (!device_denoiser || device.state_buffer[pixel] & STATE_ALBEDO)
    return;

  if (device_temporal_frames && device_accum_mode == TEMPORAL_ACCUMULATION) {
    RGBF out_albedo = device.albedo_buffer[pixel];
    out_albedo      = scale_color(out_albedo, device_temporal_frames);
    albedo          = add_color(albedo, out_albedo);
    albedo          = scale_color(albedo, 1.0f / (device_temporal_frames + 1));
  }

  device.albedo_buffer[pixel] = albedo;
  device.state_buffer[pixel] |= STATE_ALBEDO;
}

__device__ RestirEvalData load_restir_eval_data(const int offset) {
  const float4 packet = __ldcs((float4*) (device.restir_eval_data + offset));

  RestirEvalData result;
  result.position = get_vector(packet.x, packet.y, packet.z);
  result.flags    = __float_as_uint(packet.w);

  return result;
}

__device__ void store_restir_eval_data(const RestirEvalData data, const int offset) {
  float4 packet;
  packet.x = data.position.x;
  packet.y = data.position.y;
  packet.z = data.position.z;
  packet.w = __uint_as_float(data.flags);

  __stcs((float4*) (device.restir_eval_data + offset), packet);
}

__device__ RestirSample load_restir_sample(const RestirSample* ptr, const int offset) {
  const float2 packet = __ldcs((float2*) (ptr + offset));

  RestirSample sample;
  sample.id     = __float_as_uint(packet.x);
  sample.weight = packet.y;

  return sample;
}

__device__ void store_restir_sample(const RestirSample* ptr, const RestirSample sample, const int offset) {
  float2 packet;
  packet.x = __uint_as_float(sample.id);
  packet.y = sample.weight;

  __stcs((float2*) (ptr + offset), packet);
}

__device__ TraversalTriangle load_traversal_triangle(const int offset) {
  const float4* ptr = (float4*) (device_scene.traversal_triangles + offset);
  const float4 v1   = __ldg(ptr);
  const float4 v2   = __ldg(ptr + 1);
  const float v3    = __ldg((float*) (ptr + 2));

  TraversalTriangle triangle;
  triangle.vertex = get_vector(v1.x, v1.y, v1.z);
  triangle.edge1  = get_vector(v1.w, v2.x, v2.y);
  triangle.edge2  = get_vector(v2.z, v2.w, v3);

  return triangle;
}

#endif /* CU_MEMORY_H */
