// OptiX translation unit setup
#include "optix_compile_defines.cuh"
//

#include "math.cuh"
#include "memory.cuh"
#include "ocean_utils.cuh"
#include "optix_include.cuh"
#include "utils.cuh"

// Kernels must be named __[SEMANTIC]__..., for example, __raygen__...
// This can be found under function name prefix in the programming guide

////////////////////////////////////////////////////////////////////
// GBufferMetaData
////////////////////////////////////////////////////////////////////

__device__ uint16_t optix_float_to_bfloat16(const float val) {
  return __float_as_uint(val) >> 16;
}

__device__ void optix_write_out_gbuffer_meta(const DeviceTask task, OptixRaytraceResult result) {
  if (device.state.sample_id != 0 || device.state.depth != 0)
    return;

  const uint32_t shift = device.settings.supersampling + 1;
  const uint32_t mask  = (1 << shift) - 1;

  if ((task.index.x & mask) || (task.index.y & mask))
    return;

  const uint16_t x  = task.index.x >> shift;
  const uint16_t y  = task.index.y >> shift;
  const uint32_t ld = device.settings.width >> shift;

  if (device.ocean.active) {
    if (task.origin.y < OCEAN_MIN_HEIGHT || task.origin.y > OCEAN_MAX_HEIGHT) {
      const float short_distance = ocean_short_distance(task.origin, task.ray);

      if (short_distance < result.depth) {
        result.handle.instance_id = HIT_TYPE_REJECT;
        result.depth              = short_distance;
      }
    }
  }

  uint16_t material_id = MATERIAL_ID_INVALID;
  uint32_t instance_id = HIT_TYPE_INVALID;

  if (result.handle.instance_id < HIT_TYPE_TRIANGLE_ID_LIMIT) {
    const uint32_t mesh_id = mesh_id_load(result.handle.instance_id);

    material_id = material_id_load(mesh_id, result.handle.tri_id);
    instance_id = result.handle.instance_id;
  }

  vec3 rel_hit_pos = get_vector(0.0f, 0.0f, 0.0f);
  if (result.depth < FLT_MAX) {
    rel_hit_pos = scale_vector(task.ray, result.depth);
  }

  GBufferMetaData meta_data;

  meta_data.depth              = result.depth;
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

////////////////////////////////////////////////////////////////////
// Raytracing passes
////////////////////////////////////////////////////////////////////

__device__ void optix_raytrace_geometry(const DeviceTask task, OptixRaytraceResult& result) {
  OptixKernelFunctionGeometryTracePayload payload;
  payload.depth  = result.depth;
  payload.handle = result.handle;

  optixKernelFunctionGeometryTrace(
    device.optix_bvh, task.origin, task.ray, 0.0f, result.depth, 0.0f, OptixVisibilityMask(0xFFFF), OPTIX_RAY_FLAG_NONE,
    OPTIX_TRACE_STATUS_EXECUTE, payload);

  result.depth  = payload.depth;
  result.handle = payload.handle;
}

__device__ void optix_raytrace_particles(const DeviceTask task, OptixRaytraceResult& result) {
  OptixTraceStatus trace_status = OPTIX_TRACE_STATUS_EXECUTE;

  // Particles can not be hit by non delta path due to their negligible contribution
  if ((task.state & STATE_FLAG_DELTA_PATH) == 0 || !device.particles.active) {
    trace_status = OPTIX_TRACE_STATUS_ABORT;
  }

  const float time         = quasirandom_sequence_1D_base_float(QUASI_RANDOM_TARGET_CAMERA_TIME, task.index, device.state.sample_id, 0);
  const vec3 motion        = angles_to_direction(device.particles.direction_altitude, device.particles.direction_azimuth);
  const vec3 motion_offset = scale_vector(motion, time * device.particles.speed);

  const vec3 scaled_ray = scale_vector(task.ray, 1.0f / device.particles.scale);
  vec3 pos              = scale_vector(add_vector(task.origin, motion_offset), 1.0f / device.particles.scale);

  // Map our current point into the particle tiling coordinate system
  pos.x = pos.x - floorf(pos.x);
  pos.y = pos.y - floorf(pos.y);
  pos.z = pos.z - floorf(pos.z);

  OptixKernelFunctionParticleTracePayload payload;
  payload.depth       = FLT_MAX;
  payload.instance_id = HIT_TYPE_REJECT;

  optixKernelFunctionParticleTrace(
    device.optix_bvh_particles, pos, scaled_ray, 0.0f, result.depth, 0.0f, OptixVisibilityMask(0xFFFF), OPTIX_RAY_FLAG_NONE, trace_status,
    payload);

  if (payload.instance_id != HIT_TYPE_REJECT) {
    // Hit ID contains the triangle ID but we only store the actual particle / quad ID
    payload.instance_id = HIT_TYPE_PARTICLE_MIN + (payload.instance_id >> 1);

    result.handle = triangle_handle_get(payload.instance_id, 0);
    result.depth  = payload.depth;
  }
}

extern "C" __global__ void __raygen__optix() {
  HANDLE_DEVICE_ABORT();

  const uint16_t trace_task_count = device.ptrs.trace_counts[THREAD_ID];
  const uint16_t task_id          = TASK_ID;

  if (task_id >= trace_task_count)
    return;

  const uint32_t offset = get_task_address(task_id);
  const DeviceTask task = task_load(offset);

  OptixRaytraceResult result;
  result.handle = triangle_handle_get(HIT_TYPE_SKY, 0);
  result.depth  = FLT_MAX;

  optix_raytrace_geometry(task, result);
  optix_raytrace_particles(task, result);

  triangle_handle_store(result.handle, offset);
  trace_depth_store(result.depth, offset);

  optix_write_out_gbuffer_meta(task, result);
}
