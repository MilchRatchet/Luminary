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
Sample load_active_sample_no_temporal_hint(void* _ptr) {
  float* ptr = (float*)_ptr;
  Sample sample;

  const float4 data4 = __ldcs((float4*)ptr);
  sample.origin.x = data4.x;
  sample.origin.y = data4.y;
  sample.origin.z = data4.z;
  sample.ray.x = data4.w;

  const float2 data2 = __ldcs((float2*)(ptr + 4));
  sample.ray.y = data2.x;
  sample.ray.z = data2.y;

  sample.state = __ldcs((ushort2*)(ptr + 6));
  sample.random_index = __ldcs((int*)(ptr + 7));

  const float4 data4_2 =  __ldcs((float4*)(ptr + 8));
  sample.record.r = data4_2.x;
  sample.record.g = data4_2.y;
  sample.record.b = data4_2.z;
  sample.result.r = data4_2.w;

  const float4 data4_3 = __ldcs((float4*)(ptr + 12));
  sample.result.g = data4_3.x;
  sample.result.b = data4_3.y;
  sample.albedo_buffer.r = data4_3.z;
  sample.albedo_buffer.g = data4_3.w;

  sample.albedo_buffer.b = __ldcs((float*)(ptr + 16));
  sample.index = __ldcs((ushort2*)(ptr + 17));

  const float2 data2_2 = __ldcs((float2*)(ptr + 18));
  sample.depth = data2_2.x;
  sample.hit_id = float_as_uint(data2_2.y);

  return sample;
}

__device__
void store_active_sample_no_temporal_hint(Sample sample, void* _ptr) {
  float* ptr = (float*)_ptr;
  float4 data4;
  data4.x = sample.origin.x;
  data4.y = sample.origin.y;
  data4.z = sample.origin.z;
  data4.w = sample.ray.x;
  __stcs((float4*)ptr, data4);
  float2 data2;
  data2.x = sample.ray.y;
  data2.y = sample.ray.z;
  __stcs((float2*)(ptr + 4), data2);
  __stcs((ushort2*)(ptr + 6), sample.state);
  __stcs((int*)(ptr + 7), sample.random_index);
  data4.x = sample.record.r;
  data4.y = sample.record.g;
  data4.z = sample.record.b;
  data4.w = sample.result.r;
  __stcs((float4*)(ptr + 8), data4);
  data4.x = sample.result.g;
  data4.y = sample.result.b;
  data4.z = sample.albedo_buffer.r;
  data4.w = sample.albedo_buffer.g;
  __stcs((float4*)(ptr + 12), data4);
  __stcs((float*)(ptr + 16), sample.albedo_buffer.b);
  __stcs((ushort2*)(ptr + 17), sample.index);
  data2.x = sample.depth;
  data2.y = uint_as_float(sample.hit_id);
  __stcs((float2*)(ptr + 18), data2);
}

__device__
Sample load_finished_sample_no_temporal_hint(void* _ptr) {
  float* ptr = (float*)_ptr;
  Sample sample;

  sample.result.r = __ldlu((float*)(ptr + 11));
  const float4 data4 = __ldlu((float4*)(ptr + 12));
  sample.result.g = data4.x;
  sample.result.b = data4.y;
  sample.albedo_buffer.r = data4.z;
  sample.albedo_buffer.g = data4.w;
  sample.albedo_buffer.b = __ldlu((float*)(ptr + 16));

  return sample;
}

__device__
void store_finished_sample_no_temporal_hint(Sample sample, void* _ptr) {
  float* ptr = (float*)_ptr;
  __stcs((float*)(ptr + 11), sample.result.r);
  float4 data4;
  data4.x = sample.result.g;
  data4.y = sample.result.b;
  data4.z = sample.albedo_buffer.r;
  data4.w = sample.albedo_buffer.g;
  __stcs((float4*)(ptr + 12), data4);
  __stcs((float*)(ptr + 16), sample.albedo_buffer.b);
}

#endif /* CU_MEMORY_H */
