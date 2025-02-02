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

__device__ void* triangle_get_entry_address(
  const DeviceTriangle* tri_ptr, const uint32_t chunk, const uint32_t offset, const uint32_t tri_id, const uint32_t triangle_count) {
  return interleaved_buffer_get_entry_address_chunk_16((void*) tri_ptr, triangle_count, chunk, offset, tri_id);
}

////////////////////////////////////////////////////////////////////
// Task IO
////////////////////////////////////////////////////////////////////

__device__ void stream_uint2(const uint2* source, uint2* target) {
  __stcs(target, __ldcs(source));
}

__device__ void stream_float(const float* source, float* target) {
  __stcs(target, __ldcs(source));
}

__device__ void stream_float4(const float4* source, float4* target) {
  __stcs(target, __ldcs(source));
}

__device__ void swap_trace_data(const uint32_t index0, const uint32_t index1) {
  const uint32_t offset0 = get_task_address(index0);

  const float4 data0 = __ldcs((float4*) (device.ptrs.tasks + offset0));
  const float4 data1 = __ldcs((float4*) (device.ptrs.tasks + offset0) + 1);
  const float depth  = __ldcs((float*) (device.ptrs.trace_depths + offset0));
  const uint2 handle = __ldcs((uint2*) (device.ptrs.triangle_handles + offset0));

  const uint32_t offset1 = get_task_address(index1);
  stream_float4((float4*) (device.ptrs.tasks + offset1), (float4*) (device.ptrs.tasks + offset0));
  stream_float4((float4*) (device.ptrs.tasks + offset1) + 1, (float4*) (device.ptrs.tasks + offset0) + 1);
  stream_float((float*) (device.ptrs.trace_depths + offset1), (float*) (device.ptrs.trace_depths + offset0));
  stream_uint2((uint2*) (device.ptrs.triangle_handles + offset1), (uint2*) (device.ptrs.triangle_handles + offset0));

  __stcs((float4*) (device.ptrs.tasks + offset1), data0);
  __stcs((float4*) (device.ptrs.tasks + offset1) + 1, data1);
  __stcs((float*) (device.ptrs.trace_depths + offset1), depth);
  __stcs((uint2*) (device.ptrs.triangle_handles + offset1), handle);
}

__device__ DeviceTask task_load(const uint32_t offset) {
  const float4* data_ptr = (const float4*) (device.ptrs.tasks + offset);

  const float4 data0 = __ldcs(data_ptr + 0);
  const float4 data1 = __ldcs(data_ptr + 1);

  DeviceTask task;
  task.state    = __float_as_uint(data0.x) & 0xFFFF;
  task.padding  = __float_as_uint(data0.x) >> 16;
  task.index.x  = __float_as_uint(data0.y) & 0xFFFF;
  task.index.y  = __float_as_uint(data0.y) >> 16;
  task.origin.x = data0.z;
  task.origin.y = data0.w;

  task.origin.z = data1.x;
  task.ray.x    = data1.y;
  task.ray.y    = data1.z;
  task.ray.z    = data1.w;

  return task;
}

__device__ void task_store(const DeviceTask task, const uint32_t offset) {
  float4* ptr = (float4*) (device.ptrs.tasks + offset);
  float4 data0;
  float4 data1;

  data0.x = __uint_as_float(((uint32_t) task.state & 0xffff) | ((uint32_t) 0 << 16));
  data0.y = __uint_as_float(((uint32_t) task.index.x & 0xffff) | ((uint32_t) task.index.y << 16));
  data0.z = task.origin.x;
  data0.w = task.origin.y;

  __stcs(ptr, data0);

  data1.x = task.origin.z;
  data1.y = task.ray.x;
  data1.z = task.ray.y;
  data1.w = task.ray.z;

  __stcs(ptr + 1, data1);
}

__device__ TriangleHandle triangle_handle_load(const uint32_t offset) {
  const uint2* data_ptr = (const uint2*) (device.ptrs.triangle_handles + offset);

  const uint2 data = __ldcs(data_ptr + 0);

  TriangleHandle handle;
  handle.instance_id = data.x;
  handle.tri_id      = data.y;

  return handle;
}

__device__ void triangle_handle_store(const TriangleHandle handle, const uint32_t offset) {
  uint2* data_ptr = (uint2*) (device.ptrs.triangle_handles + offset);

  uint2 data;
  data.x = handle.instance_id;
  data.y = handle.tri_id;

  __stcs(data_ptr, data);
}

__device__ float trace_depth_load(const uint32_t offset) {
  return __ldcs((float*) (device.ptrs.trace_depths + offset));
}

__device__ void trace_depth_store(const float depth, const uint32_t offset) {
  __stcs((float*) (device.ptrs.trace_depths + offset), depth);
}

////////////////////////////////////////////////////////////////////
// RGBF IO
////////////////////////////////////////////////////////////////////

__device__ RGBF load_RGBF(const CompressedRGBF* ptr) {
  const CompressedRGBF color = *ptr;

  return get_color(color_decompress(color.r), color_decompress(color.g), color_decompress(color.b));
}

#ifdef UTILS_DEBUG_MODE

__device__ void store_RGBF_impl(CompressedRGBF* buffer, const uint32_t offset, const RGBF color, const char* func, uint32_t line) {
  RGBF sanitized_color = color;
  if (is_non_finite(luminance(color))) {
    // Debug code to identify paths that cause NaNs and INFs
    ushort2 pixel;
    pixel.y = (uint16_t) (offset / device.settings.width);
    pixel.x = (uint16_t) (offset - pixel.y * device.settings.width);
    printf(
      "[%s:%u] Path at (%u, %u) at depth %u on frame %u ran into a NaN or INF: (%f %f %f)\n", func, line, pixel.x, pixel.y,
      (uint32_t) device.state.depth, (uint32_t) device.state.sample_id, color.r, color.g, color.b);

    sanitized_color = UTILS_DEBUG_NAN_COLOR;
  }

  CompressedRGBF compressed_color;

  compressed_color.r = color_compress(sanitized_color.r);
  compressed_color.g = color_compress(sanitized_color.g);
  compressed_color.b = color_compress(sanitized_color.b);

  buffer[offset] = compressed_color;
}

#define store_RGBF(__macro_buffer, __macro_offset, __macro_color) \
  store_RGBF_impl(__macro_buffer, __macro_offset, __macro_color, __func__, __LINE__)

#else /* UTILS_DEBUG_MODE */

__device__ void store_RGBF_impl(CompressedRGBF* buffer, const uint32_t offset, const RGBF color) {
  const RGBF sanitized_color = is_non_finite(luminance(color)) ? UTILS_DEBUG_NAN_COLOR : color;

  CompressedRGBF compressed_color;

  compressed_color.r = color_compress(sanitized_color.r);
  compressed_color.g = color_compress(sanitized_color.g);
  compressed_color.b = color_compress(sanitized_color.b);

  buffer[offset] = compressed_color;
}

#define store_RGBF(__macro_buffer, __macro_offset, __macro_color) store_RGBF_impl(__macro_buffer, __macro_offset, __macro_color)

#endif /* !UTILS_DEBUG_MODE */

__device__ void RGBF_load_pair(
  const CompressedRGBF* src, const uint32_t x, const uint32_t y, const uint32_t ld, RGBF& pixel0, RGBF& pixel1) {
  const ushort2* src_ptr = (const ushort2*) (src + x + y * ld);

  const ushort2 data0 = __ldg(src_ptr + 0);
  const ushort2 data1 = __ldg(src_ptr + 1);
  const ushort2 data2 = __ldg(src_ptr + 2);

  pixel0 = get_color(color_decompress(data0.x), color_decompress(data0.y), color_decompress(data1.x));
  pixel1 = get_color(color_decompress(data1.y), color_decompress(data2.x), color_decompress(data2.y));
}

__device__ void RGBF_store_pair(
  CompressedRGBF* dst, const uint32_t x, const uint32_t y, const uint32_t ld, const RGBF pixel0, const RGBF pixel1) {
  ushort2* dst_ptr = (ushort2*) (dst + x + y * ld);

  const ushort2 data0 = make_ushort2(color_compress(pixel0.r), color_compress(pixel0.g));
  const ushort2 data1 = make_ushort2(color_compress(pixel0.b), color_compress(pixel1.r));
  const ushort2 data2 = make_ushort2(color_compress(pixel1.g), color_compress(pixel1.b));

  __stwt(dst_ptr + 0, data0);
  __stwt(dst_ptr + 1, data1);
  __stwt(dst_ptr + 2, data2);
}

////////////////////////////////////////////////////////////////////
// Beauty Buffer IO
////////////////////////////////////////////////////////////////////

__device__ void write_beauty_buffer_impl(const RGBF beauty, const uint32_t pixel, const bool mode_set, CompressedRGBF* buffer) {
  RGBF output = beauty;
  if (!mode_set) {
    output = add_color(beauty, load_RGBF(buffer + pixel));
  }
  store_RGBF(buffer, pixel, output);
}

__device__ void write_beauty_buffer_direct(const RGBF beauty, const uint32_t pixel, const bool mode_set = false) {
  write_beauty_buffer_impl(beauty, pixel, mode_set, device.ptrs.frame_direct_buffer);
}

__device__ void write_beauty_buffer_indirect(const RGBF beauty, const uint32_t pixel, const bool mode_set = false) {
  write_beauty_buffer_impl(beauty, pixel, mode_set, device.ptrs.frame_indirect_buffer);
}

__device__ void write_beauty_buffer(const RGBF beauty, const uint32_t pixel, const uint8_t state, const bool mode_set = false) {
  const bool is_direct = state & STATE_FLAG_DELTA_PATH;

  CompressedRGBF* buffer = (is_direct) ? device.ptrs.frame_direct_buffer : device.ptrs.frame_indirect_buffer;

  write_beauty_buffer_impl(beauty, pixel, mode_set, buffer);
}

__device__ void write_beauty_buffer_forced(const RGBF beauty, const uint32_t pixel) {
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

__device__ uint32_t mesh_id_load(const uint32_t instance_id) {
  return __ldg(device.ptrs.instance_mesh_id + instance_id);
}

__device__ uint16_t material_id_load(const uint32_t mesh_id, const uint32_t triangle_id) {
  const DeviceTriangle* ptr     = device.ptrs.triangles[mesh_id];
  const uint32_t triangle_count = __ldg(device.ptrs.triangle_counts + mesh_id);
  const uint32_t data           = __ldg((uint32_t*) triangle_get_entry_address(ptr, 3, 3, triangle_id, triangle_count));
  const uint16_t material_id    = data & 0xFFFF;

  return material_id;
}

__device__ UV load_triangle_tex_coords(const TriangleHandle handle, const float2 coords) {
  const uint32_t mesh_id = mesh_id_load(handle.instance_id);

  const DeviceTriangle* ptr     = device.ptrs.triangles[mesh_id];
  const uint32_t triangle_count = __ldg(device.ptrs.triangle_counts + mesh_id);

  const float4 data = __ldg((float4*) triangle_get_entry_address(ptr, 2, 0, handle.tri_id, triangle_count));

  const UV vertex_texture  = uv_unpack(__float_as_uint(data.y));
  const UV vertex1_texture = uv_unpack(__float_as_uint(data.z));
  const UV vertex2_texture = uv_unpack(__float_as_uint(data.w));

  return lerp_uv(vertex_texture, vertex1_texture, vertex2_texture, coords);
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

__device__ float unpack_float_from_uint16(const uint32_t data) {
  return __uint_as_float(data << 15);
}

__device__ DeviceMaterial load_material(const DeviceMaterialCompressed* data, const uint32_t offset) {
  const float4* ptr = (float4*) (data + offset);
  const float4 v0   = __ldg(ptr + 0);
  const float4 v1   = __ldg(ptr + 1);

  DeviceMaterial mat;
  mat.flags            = __float_as_uint(v0.x) & 0x00FF;
  mat.roughness_clamp  = random_uint16_t_to_float(__float_as_uint(v0.x) & 0xFF00);
  mat.metallic_tex     = __float_as_uint(v0.x) >> 16;
  mat.roughness        = random_uint16_t_to_float(__float_as_uint(v0.y) & 0xFFFF);
  mat.refraction_index = random_uint16_t_to_float(__float_as_uint(v0.y) >> 16) * 2.0f + 1.0f;
  mat.albedo.r         = random_uint16_t_to_float(__float_as_uint(v0.z) & 0xFFFF);
  mat.albedo.g         = random_uint16_t_to_float(__float_as_uint(v0.z) >> 16);
  mat.albedo.b         = random_uint16_t_to_float(__float_as_uint(v0.w) & 0xFFFF);
  mat.albedo.a         = random_uint16_t_to_float(__float_as_uint(v0.w) >> 16);
  mat.emission.r       = random_uint16_t_to_float(__float_as_uint(v1.x) & 0xFFFF);
  mat.emission.g       = random_uint16_t_to_float(__float_as_uint(v1.x) >> 16);
  mat.emission.b       = random_uint16_t_to_float(__float_as_uint(v1.y) & 0xFFFF);
  mat.emission_scale   = unpack_float_from_uint16(__float_as_uint(v1.y) >> 16);
  mat.albedo_tex       = __float_as_uint(v1.z) & 0xFFFF;
  mat.luminance_tex    = __float_as_uint(v1.z) >> 16;
  mat.roughness_tex    = __float_as_uint(v1.w) & 0xFFFF;
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
  trans.translation.x = v0.x;
  trans.translation.y = v0.y;
  trans.translation.z = v0.z;
  trans.scale.x       = v0.w;

  trans.scale.y    = v1.x;
  trans.scale.z    = v1.y;
  trans.rotation.x = __float_as_uint(v1.z) & 0xFFFF;
  trans.rotation.y = __float_as_uint(v1.z) >> 16;
  trans.rotation.z = __float_as_uint(v1.w) & 0xFFFF;
  trans.rotation.w = __float_as_uint(v1.w) >> 16;

  return trans;
}

__device__ DeviceLightTreeNode load_light_tree_node(const uint32_t offset) {
  const float4* ptr = (float4*) (device.ptrs.light_tree_nodes + offset);
  const float4 v0   = __ldg(ptr + 0);
  const float4 v1   = __ldg(ptr + 1);
  const float4 v2   = __ldg(ptr + 2);
  const float4 v3   = __ldg(ptr + 3);

  DeviceLightTreeNode node;

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

__device__ DeviceTextureObject load_texture_object(const uint16_t offset) {
  const float4* ptr = (float4*) (device.ptrs.textures + offset);
  const float4 v0   = __ldg(ptr + 0);

  union {
    float4 data;
    DeviceTextureObject tex;
  } float4_to_tex_converter;

  float4_to_tex_converter.data = v0;

  return float4_to_tex_converter.tex;
}

//===========================================================================================
// Defaults
//===========================================================================================

__device__ GBufferData gbuffer_data_default() {
  GBufferData data;

  data.instance_id = HIT_TYPE_INVALID;
  data.tri_id      = 0;
  data.albedo      = get_RGBAF(1.0f, 1.0f, 1.0f, 1.0f);
  data.emission    = get_color(0.0f, 0.0f, 0.0f);
  data.normal      = get_vector(0.0f, 0.0f, 1.0f);
  data.position    = get_vector(0.0f, 0.0f, 0.0f);
  data.V           = get_vector(0.0f, 0.0f, 1.0f);
  data.roughness   = 0.5f;
  data.state       = 0;
  data.flags       = 0;
  data.ior_in      = 1.0f;
  data.ior_out     = 1.0f;

  return data;
}

#endif /* CU_MEMORY_H */
