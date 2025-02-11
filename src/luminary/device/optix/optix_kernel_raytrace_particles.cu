#define OPTIX_KERNEL

#include "math.cuh"
#include "memory.cuh"
#include "optix_include.cuh"
#include "utils.cuh"

// Kernels must be named __[SEMANTIC]__..., for example, __raygen__...
// This can be found under function name prefix in the programming guide

extern "C" __global__ void __raygen__optix() {
  HANDLE_DEVICE_ABORT();

  const uint16_t trace_task_count = device.ptrs.trace_counts[THREAD_ID];

  for (uint32_t i = 0; i < trace_task_count; i++) {
    const uint32_t offset = get_task_address(i);
    const DeviceTask task = task_load(offset);
    float tmax            = trace_depth_load(offset);

    // Particles can not be hit by non delta path due to their negligible contribution
    if ((task.state & STATE_FLAG_DELTA_PATH) == 0)
      continue;

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
    payload.depth       = tmax;
    payload.instance_id = HIT_TYPE_REJECT;

    optixKernelFunctionParticleTrace(
      device.optix_bvh_particles, pos, scaled_ray, 0.0f, tmax, 0.0f, OptixVisibilityMask(0xFFFF), OPTIX_RAY_FLAG_NONE, payload);

    if (payload.instance_id != HIT_TYPE_REJECT) {
      // Hit ID contains the triangle ID but we only store the actual particle / quad ID
      payload.instance_id = HIT_TYPE_PARTICLE_MIN + (payload.instance_id >> 1);

      const TriangleHandle handle = triangle_handle_get(payload.instance_id, 0);

      triangle_handle_store(handle, offset);
      trace_depth_store(payload.depth, offset);
    }
  }
}
