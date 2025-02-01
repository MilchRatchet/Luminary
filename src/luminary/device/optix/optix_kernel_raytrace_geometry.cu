#define OPTIX_KERNEL

#include "math.cuh"
#include "memory.cuh"
#include "optix_include.cuh"
#include "trace.cuh"
#include "utils.cuh"

// Kernels must be named __[SEMANTIC]__..., for example, __raygen__...
// This can be found under function name prefix in the programming guide

__device__ uint16_t optix_float_to_bfloat16(const float val) {
  return __float_as_uint(val) >> 16;
}

__device__ void optix_write_out_gbuffer_meta(const DeviceTask task, const float depth, const TriangleHandle handle) {
  if (device.state.sample_id != 0 || device.state.depth != 0)
    return;

  const uint32_t shift = device.settings.supersampling + 1;
  const uint32_t mask  = (1 << shift) - 1;

  if ((task.index.x & mask) || (task.index.y & mask))
    return;

  const uint16_t x  = task.index.x >> shift;
  const uint16_t y  = task.index.y >> shift;
  const uint32_t ld = device.settings.width >> shift;

  uint16_t material_id = MATERIAL_ID_INVALID;
  uint32_t instance_id = HIT_TYPE_INVALID;

  if (handle.instance_id < HIT_TYPE_TRIANGLE_ID_LIMIT) {
    const uint32_t mesh_id = mesh_id_load(handle.instance_id);

    material_id = material_id_load(mesh_id, handle.tri_id);
    instance_id = handle.instance_id;
  }

  vec3 rel_hit_pos = get_vector(0.0f, 0.0f, 0.0f);
  if (depth < FLT_MAX) {
    rel_hit_pos = scale_vector(task.ray, depth);
  }

  GBufferMetaData meta_data;

  meta_data.depth              = depth;
  meta_data.instance_id        = instance_id;
  meta_data.material_id        = material_id;
  meta_data.rel_hit_x_bfloat16 = optix_float_to_bfloat16(rel_hit_pos.x);
  meta_data.rel_hit_y_bfloat16 = optix_float_to_bfloat16(rel_hit_pos.y);
  meta_data.rel_hit_z_bfloat16 = optix_float_to_bfloat16(rel_hit_pos.z);

  uint4 data;
  data.x = meta_data.instance_id;
  data.y = __float_as_uint(meta_data.depth);
  data.z = (((uint32_t) meta_data.rel_hit_y_bfloat16) << 16) | (meta_data.rel_hit_x_bfloat16);
  data.w = (((uint32_t) meta_data.material_id) << 16) | (meta_data.rel_hit_z_bfloat16);

  __stwt((uint4*) device.ptrs.gbuffer_meta + x + y * ld, data);
}

extern "C" __global__ void __raygen__optix() {
  HANDLE_DEVICE_ABORT();

  const uint16_t trace_task_count = device.ptrs.trace_counts[THREAD_ID];

  for (int i = 0; i < trace_task_count; i++) {
    const int offset      = get_task_address(i);
    const DeviceTask task = task_load(offset);

    TriangleHandle handle;
    const float tmax = trace_preprocess(task, handle);

    OptixKernelFunctionGeometryTracePayload payload;
    payload.depth  = tmax;
    payload.handle = handle;

    optixKernelFunctionGeometryTrace(
      device.optix_bvh, task.origin, task.ray, 0.0f, tmax, 0.0f, OptixVisibilityMask(0xFFFF), OPTIX_RAY_FLAG_NONE, payload);

    optix_write_out_gbuffer_meta(task, payload.depth, payload.handle);

    triangle_handle_store(payload.handle, offset);
    trace_depth_store(payload.depth, offset);
  }
}
