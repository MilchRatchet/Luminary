#ifndef CU_MEMORY_H
#define CU_MEMORY_H

#include "math.cuh"
#include "utils.cuh"

////////////////////////////////////////////////////////////////////
// Interleaved storage access
////////////////////////////////////////////////////////////////////

__device__ void* interleaved_buffer_get_entry_address_chunk_16(
  void* ptr, const uint32_t count, const uint32_t chunk, const uint32_t offset, const uint32_t id) {
  return (void*) (((float*) ptr) + (count * chunk + id) * 4 + offset);
}

__device__ void* interleaved_buffer_get_entry_address_chunk_8(
  void* ptr, const uint32_t count, const uint32_t chunk, const uint32_t offset, const uint32_t id) {
  return (void*) (((float*) ptr) + (count * chunk + id) * 2 + offset);
}

__device__ void* triangle_get_entry_address(const uint32_t chunk, const uint32_t offset, const uint32_t tri_id) {
  return interleaved_buffer_get_entry_address_chunk_16(
    (void*) device.ptrs.triangles, device.non_instanced_triangle_count, chunk, offset, tri_id);
}

__device__ void* trace_result_get_entry_address(const uint32_t chunk, const uint32_t offset, const uint32_t task_id) {
  return interleaved_buffer_get_entry_address_chunk_8((void*) device.ptrs.trace_results, device.max_task_count, chunk, offset, task_id);
}

////////////////////////////////////////////////////////////////////
// Load/Store functions
////////////////////////////////////////////////////////////////////

__device__ void stream_uint16(const uint16_t* source, uint16_t* target) {
  __stcs(target, __ldcs(source));
}

__device__ void stream_float2(const float2* source, float2* target) {
  __stcs(target, __ldcs(source));
}

__device__ void stream_float4(const float4* source, float4* target) {
  __stcs(target, __ldcs(source));
}

__device__ void swap_trace_data(const uint32_t index0, const uint32_t index1) {
  const uint32_t offset0 = get_task_address(index0);

  const float2 trace0   = __ldcs((float2*) trace_result_get_entry_address(0, 0, offset0));
  const uint16_t trace1 = __ldcs((uint16_t*) trace_result_get_entry_address(1, 0, offset0));
  const float4 data0    = __ldcs((float4*) (device.ptrs.trace_tasks + offset0));
  const float4 data1    = __ldcs((float4*) (device.ptrs.trace_tasks + offset0) + 1);

  const uint32_t offset1 = get_task_address(index1);
  stream_float2((float2*) trace_result_get_entry_address(0, 0, offset1), (float2*) trace_result_get_entry_address(0, 0, offset0));
  stream_uint16((uint16_t*) trace_result_get_entry_address(1, 0, offset1), (uint16_t*) trace_result_get_entry_address(1, 0, offset0));
  stream_float4((float4*) (device.ptrs.trace_tasks + offset1), (float4*) (device.ptrs.trace_tasks + offset0));
  stream_float4((float4*) (device.ptrs.trace_tasks + offset1) + 1, (float4*) (device.ptrs.trace_tasks + offset0) + 1);

  __stcs((float2*) trace_result_get_entry_address(0, 0, offset1), trace0);
  __stcs((uint16_t*) trace_result_get_entry_address(1, 0, offset1), trace1);
  __stcs((float4*) (device.ptrs.trace_tasks + offset1), data0);
  __stcs((float4*) (device.ptrs.trace_tasks + offset1) + 1, data1);
}

__device__ TraceTask load_trace_task(const uint32_t offset) {
  const float4* ptr  = (float4*) (device.ptrs.trace_tasks + offset);
  const float4 data0 = __ldcs(ptr + 0);
  const float4 data1 = __ldcs(ptr + 1);

  TraceTask task;
  task.state    = __float_as_uint(data0.x) & 0xFF;
  task.index.x  = __float_as_uint(data0.y) & 0xFFFF;
  task.index.y  = (__float_as_uint(data0.y) >> 16);
  task.origin.x = data0.z;
  task.origin.y = data0.w;

  task.origin.z = data1.x;
  task.ray.x    = data1.y;
  task.ray.y    = data1.z;
  task.ray.z    = data1.w;

  return task;
}

__device__ void store_trace_task(const TraceTask task, const uint32_t offset) {
  float4 data0;
  data0.x = __uint_as_float(task.state);
  data0.y = __uint_as_float(((uint32_t) task.index.x & 0xFFFF) | ((uint32_t) task.index.y << 16));
  data0.z = task.origin.x;
  data0.w = task.origin.y;

  float4 data1;
  data1.x = task.origin.z;
  data1.y = task.ray.x;
  data1.z = task.ray.y;
  data1.w = task.ray.z;

  float4* ptr = (float4*) (device.ptrs.trace_tasks + offset);
  __stcs(ptr + 0, data0);
  __stcs(ptr + 1, data1);
}

__device__ ShadingTask load_shading_task(const uint32_t offset) {
  float4* data_ptr = (float4*) (device.ptrs.trace_tasks + offset);

  const float4 data0 = __ldcs(data_ptr + 0);
  const float4 data1 = __ldcs(data_ptr + 1);

  ShadingTask task;
  task.instance_id = __float_as_uint(data0.x);
  task.index.x     = __float_as_uint(data0.y) & 0xFFFF;
  task.index.y     = (__float_as_uint(data0.y) >> 16);
  task.position.x  = data0.z;
  task.position.y  = data0.w;

  task.position.z = data1.x;
  task.ray.x      = data1.y;
  task.ray.y      = data1.z;
  task.ray.z      = data1.w;

  return task;
}

__device__ void store_shading_task(const ShadingTask task, const uint32_t offset) {
  float4* ptr = (float4*) (device.ptrs.trace_tasks + offset);
  float4 data0;
  float4 data1;

  data0.x = __uint_as_float(task.instance_id);
  data0.y = __uint_as_float(((uint32_t) task.index.x & 0xffff) | ((uint32_t) task.index.y << 16));
  data0.z = task.position.x;
  data0.w = task.position.y;

  __stcs(ptr, data0);

  data1.x = task.position.z;
  data1.y = task.ray.x;
  data1.z = task.ray.y;
  data1.w = task.ray.z;

  __stcs(ptr + 1, data1);
}

__device__ ShadingTaskAuxData load_shading_task_aux_data(const uint32_t offset) {
  uint32_t* data_ptr = (uint32_t*) (device.ptrs.aux_data + offset);

  const uint32_t data = __ldcs(data_ptr);

  ShadingTaskAuxData aux_data;
  aux_data.tri_id = data & 0xFFFF;
  aux_data.state  = (data >> 16) & 0xFF;

  return aux_data;
}

__device__ void store_shading_task_aux_data(const ShadingTaskAuxData aux_data, const uint32_t offset) {
  uint32_t* ptr = (uint32_t*) (device.ptrs.aux_data + offset);

  const uint32_t data = aux_data.tri_id | (((uint32_t) aux_data.state) << 16);

  __stcs(ptr, data);
}

__device__ RGBF load_RGBF(RGBF* ptr) {
  return *ptr;
}

__device__ RGBF load_RGBF(const RGBF* ptr) {
  return *ptr;
}

__device__ void store_RGBF(RGBF* ptr, const RGBF a) {
  *ptr = a;
}

__device__ TraceResult load_trace_result(const uint32_t task_offset) {
  const float2 data0   = __ldcs((float2*) trace_result_get_entry_address(0, 0, task_offset));
  const uint16_t data1 = __ldcs((uint16_t*) trace_result_get_entry_address(1, 0, task_offset));

  TraceResult result;
  result.depth       = data0.x;
  result.instance_id = __float_as_uint(data0.y);
  result.tri_id      = data1;

  return result;
}

__device__ void store_trace_result(const TraceResult result, const uint32_t task_offset) {
  float2 data0;
  data0.x = result.depth;
  data0.y = __uint_as_float(result.instance_id);

  __stcs((float2*) trace_result_get_entry_address(0, 0, task_offset), data0);
  __stcs((uint16_t*) trace_result_get_entry_address(1, 0, task_offset), result.tri_id);
}

////////////////////////////////////////////////////////////////////
// Beauty Buffer IO
////////////////////////////////////////////////////////////////////

__device__ void write_beauty_buffer_impl(const RGBF beauty, const int pixel, const bool mode_set, RGBF* buffer) {
  RGBF output = beauty;
  if (!mode_set) {
    output = add_color(beauty, load_RGBF(buffer + pixel));
  }
  store_RGBF(buffer + pixel, output);
}

__device__ void write_beauty_buffer_direct(const RGBF beauty, const int pixel, const bool mode_set = false) {
  write_beauty_buffer_impl(beauty, pixel, mode_set, device.ptrs.frame_direct_buffer);
}

__device__ void write_beauty_buffer_indirect(const RGBF beauty, const int pixel, const bool mode_set = false) {
  write_beauty_buffer_impl(beauty, pixel, mode_set, device.ptrs.frame_indirect_buffer);
}

__device__ void write_beauty_buffer(const RGBF beauty, const int pixel, const uint8_t state, const bool mode_set = false) {
  const bool is_direct = state & STATE_FLAG_DELTA_PATH;

  RGBF* buffer = (is_direct) ? device.ptrs.frame_direct_buffer : device.ptrs.frame_indirect_buffer;

  write_beauty_buffer_impl(beauty, pixel, mode_set, buffer);
}

__device__ void write_beauty_buffer_forced(const RGBF beauty, const int pixel) {
  write_beauty_buffer(beauty, pixel, STATE_FLAG_DELTA_PATH, true);
}

#ifndef NO_LUMINARY_BVH

__device__ TraversalTriangle load_traversal_triangle(const int offset) {
  float4* ptr     = (float4*) (device.bvh_triangles + offset);
  const float4 v1 = __ldg(ptr);
  const float4 v2 = __ldg(ptr + 1);
  const float4 v3 = __ldg(ptr + 2);

  TraversalTriangle triangle;
  triangle.vertex     = get_vector(v1.x, v1.y, v1.z);
  triangle.edge1      = get_vector(v1.w, v2.x, v2.y);
  triangle.edge2      = get_vector(v2.z, v2.w, v3.x);
  triangle.albedo_tex = __float_as_uint(v3.y);
  triangle.id         = __float_as_uint(v3.z);

  return triangle;
}

#endif

__device__ DeviceInstancelet load_instance(const DeviceInstancelet* data, const uint32_t offset) {
  const float2* ptr = (float2*) (data + offset);
  const float2 v    = __ldg(ptr);

  DeviceInstancelet instance;
  instance.triangles_offset = __float_as_uint(v.x);
  instance.material_id      = __float_as_uint(v.y) & 0xFFFF;

  return instance;
}

__device__ UV load_triangle_tex_coords(const TriangleHandle handle, const float2 coords) {
  const DeviceInstancelet instance = load_instance(device.ptrs.instances, handle.instance_id);

  const float4 data = __ldg((float4*) triangle_get_entry_address(2, 0, instance.triangles_offset + handle.tri_id));

  const UV vertex_texture = uv_unpack(__float_as_uint(data.y));
  const UV edge1_texture  = uv_unpack(__float_as_uint(data.z));
  const UV edge2_texture  = uv_unpack(__float_as_uint(data.w));

  return lerp_uv(vertex_texture, edge1_texture, edge2_texture, coords);
}

__device__ uint32_t load_instance_material_id(const uint32_t instance_id) {
  return __ldg(&device.ptrs.instances[instance_id].material_id);
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

__device__ DeviceMaterial load_material(const DeviceMaterialCompressed* data, const uint32_t offset) {
  const float4* ptr = (float4*) (data + offset);
  const float4 v0   = __ldg(ptr + 0);
  const float4 v1   = __ldg(ptr + 1);

  DeviceMaterial mat;
  mat.flags            = __float_as_uint(v0.x) & 0xFF;
  mat.roughness_clamp  = random_uint16_t_to_float(__float_as_uint(v0.x) & 0xFF00);
  mat.metallic         = random_uint16_t_to_float(__float_as_uint(v0.x) >> 16);
  mat.roughness        = random_uint16_t_to_float(__float_as_uint(v0.y) & 0xFFFF);
  mat.refraction_index = random_uint16_t_to_float(__float_as_uint(v0.y) >> 16) * 2.0f + 1.0f;
  mat.albedo.r         = random_uint16_t_to_float(__float_as_uint(v0.z) & 0xFFFF);
  mat.albedo.g         = random_uint16_t_to_float(__float_as_uint(v0.z) >> 16);
  mat.albedo.b         = random_uint16_t_to_float(__float_as_uint(v0.w) & 0xFFFF);
  mat.albedo.a         = random_uint16_t_to_float(__float_as_uint(v0.w) >> 16);
  mat.emission.r       = random_uint16_t_to_float(__float_as_uint(v1.x) & 0xFFFF);
  mat.emission.g       = random_uint16_t_to_float(__float_as_uint(v1.x) >> 16);
  mat.emission.b       = random_uint16_t_to_float(__float_as_uint(v1.y) & 0xFFFF);
  mat.emission_scale   = (float) (__float_as_uint(v1.y) >> 16);
  mat.albedo_tex       = __float_as_uint(v1.z) & 0xFFFF;
  mat.luminance_tex    = __float_as_uint(v1.z) >> 16;
  mat.material_tex     = __float_as_uint(v1.w) & 0xFFFF;
  mat.normal_tex       = __float_as_uint(v1.w) >> 16;

  mat.emission = scale_color(mat.emission, mat.emission_scale);

  return mat;
}

__device__ RGBAF load_material_albedo(const DeviceMaterialCompressed* data, const uint32_t offset) {
  const float2* ptr = (float2*) (data + offset);
  const float2 v    = __ldg(ptr + 1);

  RGBAF albedo;
  albedo.r = random_uint16_t_to_float(__float_as_uint(v.x) & 0xFFFF);
  albedo.g = random_uint16_t_to_float(__float_as_uint(v.x) >> 16);
  albedo.b = random_uint16_t_to_float(__float_as_uint(v.y) & 0xFFFF);
  albedo.a = random_uint16_t_to_float(__float_as_uint(v.y) >> 16);

  return albedo;
}

__device__ DeviceTransform load_transform(const uint32_t offset) {
  const float4* ptr = (float4*) (device.ptrs.instance_transforms + offset);
  const float4 v0   = __ldg(ptr + 0);
  const float4 v1   = __ldg(ptr + 1);

  DeviceTransform trans;
  trans.offset.x = v0.x;
  trans.offset.y = v0.y;
  trans.offset.z = v0.z;
  trans.scale.x  = v0.w;

  trans.scale.y    = v1.x;
  trans.scale.z    = v1.y;
  trans.rotation.x = __float_as_uint(v1.z) & 0xFFFF;
  trans.rotation.y = __float_as_uint(v1.z) >> 16;
  trans.rotation.z = __float_as_uint(v1.w) & 0xFFFF;
  trans.rotation.w = __float_as_uint(v1.w) >> 16;

  return trans;
}

__device__ LightTreeNode8Packed load_light_tree_node(const LightTreeNode8Packed* data, const int offset) {
  const float4* ptr = (float4*) (data + offset);
  const float4 v0   = __ldg(ptr + 0);
  const float4 v1   = __ldg(ptr + 1);
  const float4 v2   = __ldg(ptr + 2);
  const float4 v3   = __ldg(ptr + 3);

  LightTreeNode8Packed node;

  node.base_point.x   = v0.x;
  node.base_point.y   = v0.y;
  node.base_point.z   = v0.z;
  node.exp_x          = *((int8_t*) &v0.w + 0);
  node.exp_y          = *((int8_t*) &v0.w + 1);
  node.exp_z          = *((int8_t*) &v0.w + 2);
  node.exp_confidence = *((uint8_t*) &v0.w + 3);

  node.child_ptr      = __float_as_uint(v1.x);
  node.light_ptr      = __float_as_uint(v1.y);
  node.rel_point_x[0] = __float_as_uint(v1.z);
  node.rel_point_x[1] = __float_as_uint(v1.w);

  node.rel_point_y[0] = __float_as_uint(v2.x);
  node.rel_point_y[1] = __float_as_uint(v2.y);
  node.rel_point_z[0] = __float_as_uint(v2.z);
  node.rel_point_z[1] = __float_as_uint(v2.w);

  node.rel_energy[0]       = __float_as_uint(v3.x);
  node.rel_energy[1]       = __float_as_uint(v3.y);
  node.confidence_light[0] = __float_as_uint(v3.z);
  node.confidence_light[1] = __float_as_uint(v3.w);

  return node;
}

#endif /* CU_MEMORY_H */
