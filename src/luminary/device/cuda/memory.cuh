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
// Interthread IO
////////////////////////////////////////////////////////////////////

template <typename T>
__device__ T warp_reduce_sum(T sum) {
  sum += __shfl_xor_sync(0xFFFFFFFF, sum, 16);
  sum += __shfl_xor_sync(0xFFFFFFFF, sum, 8);
  sum += __shfl_xor_sync(0xFFFFFFFF, sum, 4);
  sum += __shfl_xor_sync(0xFFFFFFFF, sum, 2);
  sum += __shfl_xor_sync(0xFFFFFFFF, sum, 1);
  return sum;
}

template <typename T>
__device__ T warp_reduce_max(T max_value) {
  max_value = fmaxf(max_value, __shfl_xor_sync(0xFFFFFFFF, max_value, 16));
  max_value = fmaxf(max_value, __shfl_xor_sync(0xFFFFFFFF, max_value, 8));
  max_value = fmaxf(max_value, __shfl_xor_sync(0xFFFFFFFF, max_value, 4));
  max_value = fmaxf(max_value, __shfl_xor_sync(0xFFFFFFFF, max_value, 2));
  max_value = fmaxf(max_value, __shfl_xor_sync(0xFFFFFFFF, max_value, 1));
  return max_value;
}

template <typename T>
__device__ T warp_reduce_prefixsum(T value) {
  const uint32_t thread_id_in_warp = THREAD_ID & WARP_SIZE_MASK;

  for (uint32_t stride = 1; stride <= WARP_SIZE; stride = stride << 1) {
    const T shuffledValue = __shfl_up_sync(0xFFFFFFFF, value, stride);

    if (thread_id_in_warp >= stride)
      value += shuffledValue;
  }

  return value;
}

////////////////////////////////////////////////////////////////////
// Generic Scene Data IO
////////////////////////////////////////////////////////////////////

template <typename DATA_TYPE, typename LOAD_TYPE, typename STRIDE_TYPE = DATA_TYPE>
__device__ DATA_TYPE load_generic(const void* src, uint32_t offset) {
  static_assert(
    (sizeof(DATA_TYPE) / sizeof(LOAD_TYPE)) * sizeof(LOAD_TYPE) == sizeof(DATA_TYPE), "DATA_TYPE must be a multiple of LOAD_TYPE in size.");
  const LOAD_TYPE* ptr = (const LOAD_TYPE*) (((const STRIDE_TYPE*) src) + offset);

  union {
    DATA_TYPE dst_type;
    struct {
      LOAD_TYPE data[sizeof(DATA_TYPE) / sizeof(LOAD_TYPE)];
    };
  } converter;

  for (uint32_t i = 0; i < (sizeof(DATA_TYPE) / sizeof(LOAD_TYPE)); i++) {
    converter.data[i] = __ldg(ptr + i);
  }

  return converter.dst_type;
}

#define load_light_tree_node(offset) load_generic<DeviceLightTreeNode, float4>(device.ptrs.light_tree_nodes, offset)
#define load_light_tree_root() load_generic<DeviceLightTreeRootHeader, float4>(device.ptrs.light_tree_root, 0)
#define load_light_tree_root_section(offset)                                   \
  load_generic<DeviceLightTreeRootSection, float4, DeviceLightTreeRootHeader>( \
    device.ptrs.light_tree_root, 1 + offset * LIGHT_TREE_NODE_SECTION_REL_SIZE)

////////////////////////////////////////////////////////////////////
// Task State IO
////////////////////////////////////////////////////////////////////

__device__ uint32_t
  task_address_impl(const uint32_t thread_id_in_warp, const uint32_t warp_id, const uint32_t task_id, const TaskStateBufferIndex index) {
  constexpr uint32_t num_chunks = sizeof(DeviceTaskState) / sizeof(float4);

  uint32_t base_address = 0;
  base_address += thread_id_in_warp;
  base_address += warp_id * num_chunks * WARP_SIZE;
  base_address += task_id * NUM_WARPS * num_chunks * WARP_SIZE;

  base_address += index * device.config.num_tasks_per_thread * NUM_WARPS * num_chunks * WARP_SIZE;

  return base_address;
}

__device__ uint32_t task_arbitrary_warp_address(const uint32_t warp_offset, const TaskStateBufferIndex index) {
  const uint32_t thread_id_in_warp = warp_offset & WARP_SIZE_MASK;
  const uint32_t warp_id           = THREAD_ID >> WARP_SIZE_LOG;
  const uint32_t task_id           = warp_offset >> WARP_SIZE_LOG;

  return task_address_impl(thread_id_in_warp, warp_id, task_id, index);
}

__device__ uint32_t task_get_base_address(const uint32_t task_id, const TaskStateBufferIndex index) {
  const uint32_t thread_id_in_warp = THREAD_ID & WARP_SIZE_MASK;
  const uint32_t warp_id           = THREAD_ID >> WARP_SIZE_LOG;

  return task_address_impl(thread_id_in_warp, warp_id, task_id, index);
}

template <typename DATA_TYPE, typename LOAD_TYPE>
__device__ DATA_TYPE load_task_state(const uint32_t base_address, const uint32_t member_offset) {
  const uint32_t chunk_id          = member_offset >> 4;
  const uint32_t sub_member_offset = member_offset & 0xF;

  const float4* ptr = ((const float4*) device.ptrs.task_states);

  ptr += base_address;
  ptr += chunk_id * WARP_SIZE;

  ptr = (const float4*) (((const uint8_t*) ptr) + sub_member_offset);

  union {
    DATA_TYPE dst;
    struct {
      LOAD_TYPE data[sizeof(DATA_TYPE) / sizeof(LOAD_TYPE)];
    };
  } converter;

  for (uint32_t i = 0; i < (sizeof(DATA_TYPE) / sizeof(LOAD_TYPE)); i++) {
    converter.data[i] = __ldcs((const LOAD_TYPE*) (ptr + i * WARP_SIZE));
  }

  return converter.dst;
}

template <typename DATA_TYPE, typename STORE_TYPE>
__device__ void store_task_state(const uint32_t base_address, const uint32_t member_offset, const DATA_TYPE src) {
  const uint32_t chunk_id          = member_offset >> 4;
  const uint32_t sub_member_offset = member_offset & 0xF;

  float4* ptr = ((float4*) device.ptrs.task_states);

  ptr += base_address;
  ptr += chunk_id * WARP_SIZE;

  ptr = (float4*) (((uint8_t*) ptr) + sub_member_offset);

  union {
    DATA_TYPE src;
    struct {
      STORE_TYPE data[sizeof(DATA_TYPE) / sizeof(STORE_TYPE)];
    };
  } converter;

  converter.src = src;

  for (uint32_t i = 0; i < (sizeof(DATA_TYPE) / sizeof(STORE_TYPE)); i++) {
    __stcs((STORE_TYPE*) (ptr + i * WARP_SIZE), converter.data[i]);
  }
}

__device__ DeviceTask task_load(const uint32_t base_address) {
  return load_task_state<DeviceTask, float4>(base_address, offsetof(DeviceTaskState, task));
}

__device__ DeviceTaskTrace task_trace_load(const uint32_t base_address) {
  return load_task_state<DeviceTaskTrace, float4>(base_address, offsetof(DeviceTaskState, trace_result));
}

__device__ DeviceTaskThroughput task_throughput_load(const uint32_t base_address) {
  return load_task_state<DeviceTaskThroughput, float4>(base_address, offsetof(DeviceTaskState, throughput));
}

// DeviceTask

__device__ void task_store(const uint32_t base_address, const DeviceTask data) {
  store_task_state<DeviceTask, float4>(base_address, offsetof(DeviceTaskState, task), data);
}

// DeviceTaskTrace

__device__ void task_trace_store(const uint32_t base_address, const DeviceTaskTrace data) {
  store_task_state<DeviceTaskTrace, float4>(base_address, offsetof(DeviceTaskState, trace_result), data);
}

__device__ void task_trace_handle_store(const uint32_t base_address, const TriangleHandle data) {
  store_task_state<TriangleHandle, float2>(base_address, offsetof(DeviceTaskState, trace_result.handle), data);
}

__device__ void task_trace_depth_store(const uint32_t base_address, const float data) {
  store_task_state<float, float>(base_address, offsetof(DeviceTaskState, trace_result.depth), data);
}

__device__ void task_trace_ior_stack_store(const uint32_t base_address, const DeviceIORStack data) {
  store_task_state<DeviceIORStack, float>(base_address, offsetof(DeviceTaskState, trace_result.ior_stack), data);
}

// DeviceTaskThroughput

__device__ void task_throughput_store(const uint32_t base_address, const DeviceTaskThroughput data) {
  store_task_state<DeviceTaskThroughput, float4>(base_address, offsetof(DeviceTaskState, throughput), data);
}

__device__ void task_throughput_record_store(const uint32_t base_address, const PackedRecord data) {
  store_task_state<PackedRecord, float2>(base_address, offsetof(DeviceTaskState, throughput.record), data);
}

__device__ void task_throughput_mis_payload_store(const uint32_t base_address, const PackedMISPayload data) {
  store_task_state<PackedMISPayload, float2>(base_address, offsetof(DeviceTaskState, throughput.payload), data);
}

////////////////////////////////////////////////////////////////////
// RGBF IO
////////////////////////////////////////////////////////////////////

__device__ RGBF load_RGBF(const RGBF* ptr) {
  return *ptr;
}

#ifdef UTILS_DEBUG_MODE

__device__ void store_RGBF_impl(RGBF* buffer, const uint32_t offset, const RGBF color, const char* func, uint32_t line) {
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

  buffer[offset] = sanitized_color;
}

#define store_RGBF(__macro_buffer, __macro_offset, __macro_color) \
  store_RGBF_impl(__macro_buffer, __macro_offset, __macro_color, __func__, __LINE__)

#else /* UTILS_DEBUG_MODE */

__device__ void store_RGBF_impl(RGBF* buffer, const uint32_t offset, const RGBF color) {
  const RGBF sanitized_color = is_non_finite(luminance(color)) ? UTILS_DEBUG_NAN_COLOR : color;

  buffer[offset] = sanitized_color;
}

#define store_RGBF(__macro_buffer, __macro_offset, __macro_color) store_RGBF_impl(__macro_buffer, __macro_offset, __macro_color)

#endif /* !UTILS_DEBUG_MODE */

////////////////////////////////////////////////////////////////////
// Beauty Buffer IO
////////////////////////////////////////////////////////////////////

__device__ void write_beauty_buffer_impl(const RGBF beauty, const uint32_t pixel, const bool mode_set, RGBF* buffer) {
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

  RGBF* buffer = (is_direct) ? device.ptrs.frame_direct_buffer : device.ptrs.frame_indirect_buffer;

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

__device__ Star star_load(const uint32_t offset) {
  union {
    float4 data;
    Star star;
  } converter;

  converter.data = __ldg((float4*) (device.ptrs.stars + offset));

  return converter.star;
}

#endif /* CU_MEMORY_H */
