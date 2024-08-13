#ifndef CU_MEMORY_H
#define CU_MEMORY_H

#include "math.cuh"
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
  const float2 temp  = __ldcs((float2*) (device.ptrs.trace_results + offset0));
  const float4 data0 = __ldcs((float4*) (device.ptrs.trace_tasks + offset0));
  const float4 data1 = __ldcs((float4*) (device.ptrs.trace_tasks + offset0) + 1);

  const int offset1 = get_task_address(index1);
  stream_float2((float2*) (device.ptrs.trace_results + offset1), (float2*) (device.ptrs.trace_results + offset0));
  stream_float4((float4*) (device.ptrs.trace_tasks + offset1), (float4*) (device.ptrs.trace_tasks + offset0));
  stream_float4((float4*) (device.ptrs.trace_tasks + offset1) + 1, (float4*) (device.ptrs.trace_tasks + offset0) + 1);
  __stcs((float2*) (device.ptrs.trace_results + offset1), temp);
  __stcs((float4*) (device.ptrs.trace_tasks + offset1), data0);
  __stcs((float4*) (device.ptrs.trace_tasks + offset1) + 1, data1);
}

__device__ TraceTask load_trace_task(const TraceTask* ptr) {
  const float4 data0 = __ldcs((float4*) ptr);
  const float4 data1 = __ldcs(((float4*) ptr) + 1);

  TraceTask task;
  task.index.x  = __float_as_uint(data0.y) & 0xffff;
  task.index.y  = (__float_as_uint(data0.y) >> 16);
  task.origin.x = data0.z;
  task.origin.y = data0.w;

  task.origin.z = data1.x;
  task.ray.x    = data1.y;
  task.ray.y    = data1.z;
  task.ray.z    = data1.w;

  return task;
}

__device__ void store_trace_task(TraceTask* ptr, const TraceTask task) {
  float4 data0;
  data0.x = __uint_as_float(0xffffffff);
  data0.y = __uint_as_float(((uint32_t) task.index.x & 0xffff) | ((uint32_t) task.index.y << 16));
  data0.z = task.origin.x;
  data0.w = task.origin.y;

  float4 data1;
  data1.x = task.origin.z;
  data1.y = task.ray.x;
  data1.z = task.ray.y;
  data1.w = task.ray.z;

  float4* data_ptr = (float4*) ptr;

  __stcs(data_ptr + 0, data0);
  __stcs(data_ptr + 1, data1);
}

__device__ ShadingTask load_shading_task(TraceTask* ptr) {
  float4* data_ptr = (float4*) ptr;

  const float4 data0 = __ldcs(data_ptr + 0);
  const float4 data1 = __ldcs(data_ptr + 1);

  ShadingTask task;
  task.hit_id     = __float_as_uint(data0.x);
  task.index.x    = __float_as_uint(data0.y) & 0xffff;
  task.index.y    = (__float_as_uint(data0.y) >> 16);
  task.position.x = data0.z;
  task.position.y = data0.w;

  task.position.z = data1.x;
  task.ray.x      = data1.y;
  task.ray.y      = data1.z;
  task.ray.z      = data1.w;

  return task;
}

__device__ RGBAhalf load_RGBAhalf(void* ptr) {
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

__device__ RGBF load_RGBF(RGBF* ptr) {
  return *ptr;
}

__device__ RGBF load_RGBF(const RGBF* ptr) {
  return *ptr;
}

__device__ void store_RGBF(RGBF* ptr, const RGBF a) {
  *ptr = a;
}

/*
 * Updates the albedo buffer if criteria are met.
 * @param albedo Albedo color to be added to the albedo buffer.
 * @param pixel Index of pixel.
 */
__device__ void write_albedo_buffer(RGBF albedo, const int pixel) {
  if ((!device.denoiser && !device.aov_mode))
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

__device__ void write_normal_buffer(const vec3 normal, const int pixel) {
  if ((!device.denoiser && !device.aov_mode) || !IS_PRIMARY_RAY)
    return;

  if (device.temporal_frames && device.accum_mode == TEMPORAL_ACCUMULATION)
    return;

  device.ptrs.normal_buffer[pixel] = get_color(normal.x, normal.y, normal.z);
}

__device__ void write_beauty_buffer(const RGBF beauty, const int pixel, const bool mode_set = false) {
  RGBF output = beauty;
  if (!mode_set) {
    output = add_color(beauty, load_RGBF(device.ptrs.frame_buffer + pixel));
  }
  store_RGBF(device.ptrs.frame_buffer + pixel, output);

  const bool is_direct = state_peek(pixel, STATE_FLAG_DELTA_PATH);

  if (is_direct) {
    output = beauty;
    if (!mode_set) {
      output = add_color(beauty, load_RGBF(device.ptrs.frame_direct_buffer + pixel));
    }
    store_RGBF(device.ptrs.frame_direct_buffer + pixel, output);
  }
  else {
    output = beauty;
    if (!mode_set) {
      output = add_color(beauty, load_RGBF(device.ptrs.frame_indirect_buffer + pixel));
    }
    store_RGBF(device.ptrs.frame_indirect_buffer + pixel, output);
  }
}

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

__device__ void* interleaved_buffer_get_entry_address(
  void* ptr, const uint32_t count, const uint32_t chunk, const uint32_t offset, const uint32_t id) {
  return (void*) (((float*) ptr) + (count * chunk + id) * 4 + offset);
}

__device__ void* pixel_buffer_get_entry_address(void* ptr, const uint32_t chunk, const uint32_t offset, const uint32_t pixel) {
  return interleaved_buffer_get_entry_address(ptr, device.width * device.height, chunk, offset, pixel);
}

__device__ void* triangle_get_entry_address(const uint32_t chunk, const uint32_t offset, const uint32_t tri_id) {
  return interleaved_buffer_get_entry_address(device.scene.triangles, device.scene.triangle_data.triangle_count, chunk, offset, tri_id);
}

__device__ UV load_triangle_tex_coords(const int offset, const float2 coords) {
  const float2 bytes0x48 = __ldg((float2*) triangle_get_entry_address(4, 2, offset));
  const float4 bytes0x50 = __ldg((float4*) triangle_get_entry_address(5, 0, offset));

  const UV vertex_texture = get_uv(bytes0x48.x, bytes0x48.y);
  const UV edge1_texture  = get_uv(bytes0x50.x, bytes0x50.y);
  const UV edge2_texture  = get_uv(bytes0x50.z, bytes0x50.w);

  return lerp_uv(vertex_texture, edge1_texture, edge2_texture, coords);
}

__device__ uint32_t load_triangle_material_id(const uint32_t id) {
  return __ldg((uint32_t*) triangle_get_entry_address(6, 0, id));
}

__device__ uint32_t load_triangle_light_id(const uint32_t id) {
  return __ldg((uint32_t*) triangle_get_entry_address(6, 1, id));
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
  triangle.power       = v3.w;

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

__device__ LightTreeNode load_light_tree_node(const LightTreeNode* data, const int offset) {
  const float4* ptr = (float4*) (data + offset);
  const float4 v0   = __ldg(ptr + 0);
  const float4 v1   = __ldg(ptr + 1);
  const float4 v2   = __ldg(ptr + 2);

  LightTreeNode node;

  node.left_ref_point.x  = v0.x;
  node.left_ref_point.y  = v0.y;
  node.left_ref_point.z  = v0.z;
  node.right_ref_point.x = v0.w;
  node.right_ref_point.y = v1.x;
  node.right_ref_point.z = v1.y;
  node.left_confidence   = v1.z;
  node.right_confidence  = v1.w;
  node.left_energy       = v2.x;
  node.right_energy      = v2.y;
  node.ptr               = __float_as_uint(v2.z);
  node.light_count       = __float_as_uint(v2.w);

  return node;
}

__device__ LightTreeNode8Packed load_light_tree_node_8(const LightTreeNode8Packed* data, const int offset) {
  const float4* ptr = (float4*) (data + offset);
  const float4 v0   = __ldg(ptr + 0);
  const float4 v1   = __ldg(ptr + 1);
  const float4 v2   = __ldg(ptr + 2);
  const float4 v3   = __ldg(ptr + 3);
  const float4 v4   = __ldg(ptr + 4);

  LightTreeNode8Packed node;

  node.base_point.x = v0.x;
  node.base_point.y = v0.y;
  node.base_point.z = v0.z;
  node.exp_x        = __float_as_uint(v0.w) >> 0 & 0xFF;
  node.exp_y        = __float_as_uint(v0.w) >> 8 & 0xFF;
  node.exp_z        = __float_as_uint(v0.w) >> 16 & 0xFF;
  node.child_count  = __float_as_uint(v0.w) >> 24 & 0xFF;

  node.child_ptr      = __float_as_uint(v1.x);
  node.light_ptr      = __float_as_uint(v1.y);
  node.max_energy     = v1.z;
  node.max_confidence = v1.w;

  node.rel_point_x[0] = __float_as_uint(v2.x);
  node.rel_point_x[1] = __float_as_uint(v2.y);
  node.rel_point_y[0] = __float_as_uint(v2.z);
  node.rel_point_y[1] = __float_as_uint(v2.w);

  node.rel_point_z[0] = __float_as_uint(v3.x);
  node.rel_point_z[1] = __float_as_uint(v3.y);
  node.rel_energy[0]  = __float_as_uint(v3.z);
  node.rel_energy[1]  = __float_as_uint(v3.w);

  node.rel_confidence[0] = __float_as_uint(v4.x);
  node.rel_confidence[1] = __float_as_uint(v4.y);
  node.light_index[0]    = __float_as_uint(v4.z);
  node.light_index[1]    = __float_as_uint(v4.w);

  return node;
}

#endif /* CU_MEMORY_H */
