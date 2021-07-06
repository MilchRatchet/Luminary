#ifndef CU_MEMORY_H
#define CU_MEMORY_H

#include "utils.cuh"

//===========================================================================================
// Memory Prefetch Functions
//===========================================================================================

__device__
void __prefetch_global_l1(const void* const ptr)
{
  asm("prefetch.global.L1 [%0];" : : "l"(ptr));
}

__device__
void __prefetch_global_l2(const void* const ptr)
{
  asm("prefetch.global.L2 [%0];" : : "l"(ptr));
}

//===========================================================================================
// Minimal Cache Pollution Loads/Stores
//===========================================================================================

__device__
TraceTask load_trace_task(const void* ptr) {
  const float4 data0 = __ldcs((float4*)ptr);
  const float4 data1 = __ldcs(((float4*)ptr) + 1);

  TraceTask task;
  task.origin.x = data0.x;
  task.origin.y = data0.y;
  task.origin.z = data0.z;
  task.ray.x = data0.w;

  task.ray.y = data1.x;
  task.ray.z = data1.y;
  task.index.x = float_as_uint(data1.z) & 0xffff;
  task.index.y = (float_as_uint(data1.z) >> 16);
  task.state = float_as_uint(data1.w);

  return task;
}

__device__
TraceTask load_trace_task_essentials(const void* ptr) {
  const float4 data0 = __ldcs((float4*)ptr);
  const float2 data1 = __ldcs(((float2*)ptr) + 2);

  TraceTask task;
  task.origin.x = data0.x;
  task.origin.y = data0.y;
  task.origin.z = data0.z;
  task.ray.x = data0.w;

  task.ray.y = data1.x;
  task.ray.z = data1.y;

  return task;
}

__device__
GeometryTask load_geometry_task(const void* ptr) {
  const float4 data0 = __ldcs((float4*)ptr);
  const float4 data1 = __ldcs(((float4*)ptr) + 1);

  GeometryTask task;
  task.position.x = data0.x;
  task.position.y = data0.y;
  task.position.z = data0.z;
  task.ray_y = data0.w;

  task.ray_xz = data1.x;
  task.hit_id = float_as_uint(data1.y);
  task.index.x = float_as_uint(data1.z) & 0xffff;
  task.index.y = (float_as_uint(data1.z) >> 16);
  task.state = float_as_uint(data1.w);

  return task;
}

__device__
OceanTask load_ocean_task(const void* ptr) {
  const float4 data0 = __ldcs((float4*)ptr);
  const float4 data1 = __ldcs(((float4*)ptr) + 1);

  OceanTask task;
  task.position.x = data0.x;
  task.position.y = data0.y;
  task.position.z = data0.z;
  task.ray_y = data0.w;

  task.ray_xz = data1.x;
  task.distance = data1.y;
  task.index.x = float_as_uint(data1.z) & 0xffff;
  task.index.y = (float_as_uint(data1.z) >> 16);
  task.state = float_as_uint(data1.w);

  return task;
}

__device__
SkyTask load_sky_task(const void* ptr) {
  const float4 data = __ldcs((float4*)ptr);

  SkyTask task;
  task.ray.x = data.x;
  task.ray.y = data.y;
  task.ray.z = data.z;
  task.index.x = float_as_uint(data.w) & 0xffff;
  task.index.y = (float_as_uint(data.w) >> 16);

  return task;
}



#endif /* CU_MEMORY_H */
