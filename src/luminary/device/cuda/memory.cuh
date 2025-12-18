#ifndef CU_MEMORY_H
#define CU_MEMORY_H

#include "math.cuh"
#include "utils.cuh"

////////////////////////////////////////////////////////////////////
// Interthread IO
////////////////////////////////////////////////////////////////////

template <uint32_t WIDTH = 32, typename T>
LUMINARY_FUNCTION T warp_reduce_sum(T sum) {
  const uint32_t mask = __ballot_sync(0xFFFFFFFF, 1);

  if constexpr (WIDTH >= 32)
    sum += __shfl_xor_sync(mask, sum, 16);

  if constexpr (WIDTH >= 16)
    sum += __shfl_xor_sync(mask, sum, 8);

  if constexpr (WIDTH >= 8)
    sum += __shfl_xor_sync(mask, sum, 4);

  if constexpr (WIDTH >= 4)
    sum += __shfl_xor_sync(mask, sum, 2);

  if constexpr (WIDTH >= 2)
    sum += __shfl_xor_sync(mask, sum, 1);

  return sum;
}

template <uint32_t WIDTH = 32, typename T>
LUMINARY_FUNCTION T warp_reduce_max(T max_value) {
  const uint32_t mask = __ballot_sync(0xFFFFFFFF, 1);

  if constexpr (WIDTH >= 32)
    max_value = fmaxf(max_value, __shfl_xor_sync(mask, max_value, 16));

  if constexpr (WIDTH >= 16)
    max_value = fmaxf(max_value, __shfl_xor_sync(mask, max_value, 8));

  if constexpr (WIDTH >= 8)
    max_value = fmaxf(max_value, __shfl_xor_sync(mask, max_value, 4));

  if constexpr (WIDTH >= 4)
    max_value = fmaxf(max_value, __shfl_xor_sync(mask, max_value, 2));

  if constexpr (WIDTH >= 2)
    max_value = fmaxf(max_value, __shfl_xor_sync(mask, max_value, 1));

  return max_value;
}

template <typename T>
LUMINARY_FUNCTION T warp_reduce_prefixsum(T value) {
  // It is important that all threads are participating. In theory, this should also work as long as all threads enter this function but
  // then some are predicated off. However, I had issues with that. The intention now is to pass in a 0 if a thread does not want to
  // participate.
  const uint32_t thread_id_in_warp = THREAD_ID & WARP_SIZE_MASK;
  // Example code to enable predicating
  // const uint32_t mask              = __ballot_sync(0xFFFFFFFF, thread_predicate);
  // const uint32_t rank              = __popc(mask & ((1u << thread_id_in_warp) - 1));

  for (uint32_t stride = 1; stride < WARP_SIZE; stride = stride << 1) {
    const T shuffledValue = __shfl_up_sync(0xFFFFFFFF, value, stride);

    if (thread_id_in_warp >= stride)
      value += shuffledValue;
  }

  return value;
}

template <typename T>
LUMINARY_FUNCTION T warp_reduce_broadcast(const T value, const uint32_t src_thread_id_in_warp) {
  return __shfl_sync(0xFFFFFFFF, value, src_thread_id_in_warp);
}

////////////////////////////////////////////////////////////////////
// Generic Scene Data IO
////////////////////////////////////////////////////////////////////

template <typename DATA_TYPE, typename LOAD_TYPE, typename STRIDE_TYPE = DATA_TYPE>
LUMINARY_FUNCTION DATA_TYPE load_generic(const void* src, uint32_t offset) {
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

template <typename TYPE>
LUMINARY_FUNCTION uint32_t
  task_address_impl(const uint32_t thread_id_in_warp, const uint32_t warp_id, const uint32_t task_id, const TaskStateBufferIndex index) {
  constexpr uint32_t num_chunks = sizeof(TYPE) / sizeof(float4);

  uint32_t base_address = 0;
  base_address += thread_id_in_warp;
  base_address += warp_id * num_chunks * WARP_SIZE;
  base_address += task_id * NUM_WARPS * num_chunks * WARP_SIZE;

  base_address += index * device.config.num_tasks_per_thread * NUM_WARPS * num_chunks * WARP_SIZE;

  return base_address;
}

template <typename TYPE = DeviceTaskState>
LUMINARY_FUNCTION uint32_t task_arbitrary_warp_address(const uint32_t warp_offset, const TaskStateBufferIndex index) {
  const uint32_t thread_id_in_warp = warp_offset & WARP_SIZE_MASK;
  const uint32_t warp_id           = THREAD_ID >> WARP_SIZE_LOG;
  const uint32_t task_id           = warp_offset >> WARP_SIZE_LOG;

  return task_address_impl<TYPE>(thread_id_in_warp, warp_id, task_id, index);
}

template <typename TYPE = DeviceTaskState>
LUMINARY_FUNCTION uint32_t task_get_base_address(const uint32_t task_id, const TaskStateBufferIndex index) {
  const uint32_t thread_id_in_warp = THREAD_ID & WARP_SIZE_MASK;
  const uint32_t warp_id           = THREAD_ID >> WARP_SIZE_LOG;

  return task_address_impl<TYPE>(thread_id_in_warp, warp_id, task_id, index);
}

template <typename DATA_TYPE, typename LOAD_TYPE>
LUMINARY_FUNCTION DATA_TYPE load_task_state(const void* LUM_RESTRICT src_ptr, const uint32_t base_address, const uint32_t member_offset) {
  const uint32_t chunk_id          = member_offset >> 4;
  const uint32_t sub_member_offset = member_offset & 0xF;

  const float4* ptr = ((const float4* LUM_RESTRICT) src_ptr);

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
LUMINARY_FUNCTION void store_task_state(
  void* LUM_RESTRICT dst_ptr, const uint32_t base_address, const uint32_t member_offset, const DATA_TYPE src) {
  const uint32_t chunk_id          = member_offset >> 4;
  const uint32_t sub_member_offset = member_offset & 0xF;

  float4* ptr = ((float4*) dst_ptr);

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

LUMINARY_FUNCTION DeviceTask task_load(const uint32_t base_address) {
  return load_task_state<DeviceTask, float4>(device.ptrs.task_states, base_address, offsetof(DeviceTaskState, task));
}

LUMINARY_FUNCTION DeviceTaskTrace task_trace_load(const uint32_t base_address) {
  return load_task_state<DeviceTaskTrace, float4>(device.ptrs.task_states, base_address, offsetof(DeviceTaskState, trace_result));
}

LUMINARY_FUNCTION DeviceTaskThroughput task_throughput_load(const uint32_t base_address) {
  return load_task_state<DeviceTaskThroughput, float4>(device.ptrs.task_states, base_address, offsetof(DeviceTaskState, throughput));
}

LUMINARY_FUNCTION DeviceTaskMediumStack task_medium_load(const uint32_t base_address) {
  return load_task_state<DeviceTaskMediumStack, float4>(device.ptrs.task_states, base_address, offsetof(DeviceTaskState, medium));
}

LUMINARY_FUNCTION DeviceTaskResult task_result_load(const uint32_t base_address) {
  return load_task_state<DeviceTaskResult, float4>(device.ptrs.task_results, base_address, 0);
}

// DeviceTask

LUMINARY_FUNCTION void task_store(const uint32_t base_address, const DeviceTask data) {
  store_task_state<DeviceTask, float4>(device.ptrs.task_states, base_address, offsetof(DeviceTaskState, task), data);
}

// DeviceTaskTrace

LUMINARY_FUNCTION void task_trace_store(const uint32_t base_address, const DeviceTaskTrace data) {
  store_task_state<DeviceTaskTrace, float4>(device.ptrs.task_states, base_address, offsetof(DeviceTaskState, trace_result), data);
}

LUMINARY_FUNCTION void task_trace_handle_store(const uint32_t base_address, const TriangleHandle data) {
  store_task_state<TriangleHandle, float2>(device.ptrs.task_states, base_address, offsetof(DeviceTaskState, trace_result.handle), data);
}

LUMINARY_FUNCTION void task_trace_depth_store(const uint32_t base_address, const float data) {
  store_task_state<float, float>(device.ptrs.task_states, base_address, offsetof(DeviceTaskState, trace_result.depth), data);
}

// DeviceTaskThroughput

LUMINARY_FUNCTION void task_throughput_store(const uint32_t base_address, const DeviceTaskThroughput data) {
  store_task_state<DeviceTaskThroughput, float4>(device.ptrs.task_states, base_address, offsetof(DeviceTaskState, throughput), data);
}

LUMINARY_FUNCTION void task_throughput_record_store(const uint32_t base_address, const PackedRecord data) {
  store_task_state<PackedRecord, float2>(device.ptrs.task_states, base_address, offsetof(DeviceTaskState, throughput.record), data);
}

// DeviceTaskMediumStack

LUMINARY_FUNCTION void task_medium_store(const uint32_t base_address, const DeviceTaskMediumStack data) {
  store_task_state<DeviceTaskMediumStack, float4>(device.ptrs.task_states, base_address, offsetof(DeviceTaskState, medium), data);
}

// DeviceTaskResult

LUMINARY_FUNCTION void task_result_store(const uint32_t base_address, const DeviceTaskResult data) {
  store_task_state<DeviceTaskResult, float4>(device.ptrs.task_results, base_address, 0, data);
}

// DeviceTaskDirectLight

LUMINARY_FUNCTION DeviceTaskDirectLightGeo task_direct_light_geo_load(const uint32_t base_address) {
  return load_task_state<DeviceTaskDirectLightGeo, float4>(
    device.ptrs.task_direct_light, base_address, offsetof(DeviceTaskDirectLight, geo));
}

LUMINARY_FUNCTION DeviceTaskDirectLightSun task_direct_light_sun_load(const uint32_t base_address) {
  return load_task_state<DeviceTaskDirectLightSun, float4>(
    device.ptrs.task_direct_light, base_address, offsetof(DeviceTaskDirectLight, sun));
}

LUMINARY_FUNCTION DeviceTaskDirectLightAmbient task_direct_light_ambient_load(const uint32_t base_address) {
  return load_task_state<DeviceTaskDirectLightAmbient, float4>(
    device.ptrs.task_direct_light, base_address, offsetof(DeviceTaskDirectLight, ambient));
}

LUMINARY_FUNCTION DeviceTaskDirectLightBridges task_direct_light_bridges_load(const uint32_t base_address) {
  return load_task_state<DeviceTaskDirectLightBridges, float4>(
    device.ptrs.task_direct_light, base_address, offsetof(DeviceTaskDirectLight, bridges));
}

LUMINARY_FUNCTION DeviceTaskDirectLightBSDF task_direct_light_bsdf_load(const uint32_t base_address) {
  return load_task_state<DeviceTaskDirectLightBSDF, float4>(
    device.ptrs.task_direct_light, base_address, offsetof(DeviceTaskDirectLight, bsdf));
}

LUMINARY_FUNCTION void task_direct_light_geo_store(const uint32_t base_address, const DeviceTaskDirectLightGeo data) {
  store_task_state<DeviceTaskDirectLightGeo, float4>(
    device.ptrs.task_direct_light, base_address, offsetof(DeviceTaskDirectLight, geo), data);
}

LUMINARY_FUNCTION void task_direct_light_sun_store(const uint32_t base_address, const DeviceTaskDirectLightSun data) {
  store_task_state<DeviceTaskDirectLightSun, float4>(
    device.ptrs.task_direct_light, base_address, offsetof(DeviceTaskDirectLight, sun), data);
}

LUMINARY_FUNCTION void task_direct_light_ambient_store(const uint32_t base_address, const DeviceTaskDirectLightAmbient data) {
  store_task_state<DeviceTaskDirectLightAmbient, float4>(
    device.ptrs.task_direct_light, base_address, offsetof(DeviceTaskDirectLight, ambient), data);
}

LUMINARY_FUNCTION void task_direct_light_bridges_store(const uint32_t base_address, const DeviceTaskDirectLightBridges data) {
  store_task_state<DeviceTaskDirectLightBridges, float4>(
    device.ptrs.task_direct_light, base_address, offsetof(DeviceTaskDirectLight, bridges), data);
}

LUMINARY_FUNCTION void task_direct_light_bsdf_store(const uint32_t base_address, const DeviceTaskDirectLightBSDF data) {
  store_task_state<DeviceTaskDirectLightBSDF, float4>(
    device.ptrs.task_direct_light, base_address, offsetof(DeviceTaskDirectLight, bsdf), data);
}

////////////////////////////////////////////////////////////////////
// RGBF IO
////////////////////////////////////////////////////////////////////

LUMINARY_FUNCTION RGBF load_RGBF(const RGBF* ptr) {
  return *ptr;
}

#ifdef UTILS_DEBUG_MODE

LUMINARY_FUNCTION void store_RGBF_impl(RGBF* buffer, const uint32_t offset, const RGBF color, const char* func, uint32_t line) {
  RGBF sanitized_color = color;
  if (is_non_finite(color_luminance(color))) {
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

LUMINARY_FUNCTION void store_RGBF_impl(RGBF* buffer, const uint32_t offset, const RGBF color) {
  const RGBF sanitized_color = is_non_finite(color_luminance(color)) ? UTILS_DEBUG_NAN_COLOR : color;

  buffer[offset] = sanitized_color;
}

#define store_RGBF(__macro_buffer, __macro_offset, __macro_color) store_RGBF_impl(__macro_buffer, __macro_offset, __macro_color)

#endif /* !UTILS_DEBUG_MODE */

////////////////////////////////////////////////////////////////////
// Beauty Buffer IO
////////////////////////////////////////////////////////////////////

LUMINARY_FUNCTION void write_beauty_buffer(const RGBF beauty, const uint32_t results_index) {
  if (color_any(beauty) == false)
    return;

  DeviceTaskResult result = task_result_load(results_index);

  result.color = add_color(result.color, beauty);

  task_result_store(results_index, result);
}

////////////////////////////////////////////////////////////////////
// Geometry data IO
////////////////////////////////////////////////////////////////////

LUMINARY_FUNCTION uint32_t mesh_id_load(const uint32_t instance_id) {
  return __ldg(device.ptrs.instance_mesh_ids + instance_id);
}

LUMINARY_FUNCTION uint16_t material_id_load(const uint32_t mesh_id, const uint32_t triangle_id) {
  const DeviceTriangleTexture* ptr = device.ptrs.texture_triangles[mesh_id];

  return __ldg(&ptr[triangle_id].material_id);
}

LUMINARY_FUNCTION const DeviceTriangleVertex* triangle_vertex_ptr_load(const uint32_t mesh_id) {
  return (const DeviceTriangleVertex*) __ldg((uint64_t*) (device.ptrs.vertices + mesh_id));
}

LUMINARY_FUNCTION const DeviceTriangleTexture* triangle_texture_ptr_load(const uint32_t mesh_id) {
  return (const DeviceTriangleTexture*) __ldg((uint64_t*) (device.ptrs.texture_triangles + mesh_id));
}

LUMINARY_FUNCTION DeviceTriangleVertex triangle_vertex_load(const DeviceTriangleVertex* ptr, const uint32_t vertex_id) {
  const float4 data = __ldg((const float4*) (ptr + vertex_id));

  DeviceTriangleVertex vertex;
  vertex.pos    = get_vector(data.x, data.y, data.z);
  vertex.normal = __float_as_uint(data.w);

  return vertex;
}

LUMINARY_FUNCTION DeviceTriangleTexture triangle_texture_load(const DeviceTriangleTexture* ptr, const uint32_t triangle_id) {
  const float4 data = __ldg((const float4*) (ptr + triangle_id));

  DeviceTriangleTexture triangle;
  triangle.vertex_texture  = __float_as_uint(data.x);
  triangle.vertex1_texture = __float_as_uint(data.y);
  triangle.vertex2_texture = __float_as_uint(data.z);
  triangle.material_id     = __float_as_uint(data.w) & 0xFFFF;

  return triangle;
}

LUMINARY_FUNCTION UV load_triangle_tex_coords(const TriangleHandle handle, const float2 coords) {
  const uint32_t mesh_id = mesh_id_load(handle.instance_id);

  const DeviceTriangleTexture* ptr     = triangle_texture_ptr_load(mesh_id);
  const DeviceTriangleTexture triangle = triangle_texture_load(ptr, handle.tri_id);

  const UV vertex_texture  = uv_unpack(triangle.vertex_texture);
  const UV vertex1_texture = uv_unpack(triangle.vertex1_texture);
  const UV vertex2_texture = uv_unpack(triangle.vertex2_texture);

  return lerp_uv(vertex_texture, vertex1_texture, vertex2_texture, coords);
}

LUMINARY_FUNCTION Quad load_quad(const Quad* data, const int offset) {
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

LUMINARY_FUNCTION DeviceMaterial load_material(const DeviceMaterialCompressed* data, const uint32_t offset) {
  const float4* ptr = (float4*) (data + offset);
  const float4 v0   = __ldg(ptr + 0);
  const float4 v1   = __ldg(ptr + 1);

  DeviceMaterial mat;
  mat.flags            = __float_as_uint(v0.x) & 0x00FF;
  mat.roughness_clamp  = normed_float_unpack(__float_as_uint(v0.x) & 0xFF00);
  mat.metallic_tex     = __float_as_uint(v0.x) >> 16;
  mat.roughness        = normed_float_unpack(__float_as_uint(v0.y) & 0xFFFF);
  mat.refraction_index = normed_float_unpack(__float_as_uint(v0.y) >> 16) * 2.0f + 1.0f;
  mat.albedo.r         = normed_float_unpack(__float_as_uint(v0.z) & 0xFFFF);
  mat.albedo.g         = normed_float_unpack(__float_as_uint(v0.z) >> 16);
  mat.albedo.b         = normed_float_unpack(__float_as_uint(v0.w) & 0xFFFF);
  mat.albedo.a         = normed_float_unpack(__float_as_uint(v0.w) >> 16);
  mat.emission.r       = normed_float_unpack(__float_as_uint(v1.x) & 0xFFFF);
  mat.emission.g       = normed_float_unpack(__float_as_uint(v1.x) >> 16);
  mat.emission.b       = normed_float_unpack(__float_as_uint(v1.y) & 0xFFFF);
  mat.emission_scale   = unsigned_float_unpack(__float_as_uint(v1.y) >> 16);
  mat.albedo_tex       = __float_as_uint(v1.z) & 0xFFFF;
  mat.luminance_tex    = __float_as_uint(v1.z) >> 16;
  mat.roughness_tex    = __float_as_uint(v1.w) & 0xFFFF;
  mat.normal_tex       = __float_as_uint(v1.w) >> 16;

  mat.emission = scale_color(mat.emission, mat.emission_scale);

  return mat;
}

LUMINARY_FUNCTION RGBAF load_material_albedo(const DeviceMaterialCompressed* data, const uint32_t offset) {
  const float2* ptr = (float2*) (data + offset);
  const float2 v    = __ldg(ptr + 1);

  RGBAF albedo;
  albedo.r = normed_float_unpack(__float_as_uint(v.x) & 0xFFFF);
  albedo.g = normed_float_unpack(__float_as_uint(v.x) >> 16);
  albedo.b = normed_float_unpack(__float_as_uint(v.y) & 0xFFFF);
  albedo.a = normed_float_unpack(__float_as_uint(v.y) >> 16);

  return albedo;
}

LUMINARY_FUNCTION DeviceTransform load_transform(const uint32_t offset) {
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

LUMINARY_FUNCTION DeviceTextureObject load_texture_object(const uint16_t offset) {
  const float4* ptr = (float4*) (device.ptrs.textures + offset);
  const float4 v0   = __ldg(ptr + 0);

  union {
    float4 data;
    DeviceTextureObject tex;
  } float4_to_tex_converter;

  float4_to_tex_converter.data = v0;

  return float4_to_tex_converter.tex;
}

LUMINARY_FUNCTION Star star_load(const uint32_t offset) {
  union {
    float4 data;
    Star star;
  } converter;

  converter.data = __ldg((float4*) (device.ptrs.stars + offset));

  return converter.star;
}

#endif /* CU_MEMORY_H */
