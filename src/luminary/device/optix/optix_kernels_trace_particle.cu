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

    const float time         = quasirandom_sequence_1D_base_float(QUASI_RANDOM_TARGET_CAMERA_TIME, task.index, device.state.sample_id, 0);
    const vec3 motion        = angles_to_direction(device.particles.direction_altitude, device.particles.direction_azimuth);
    const vec3 motion_offset = scale_vector(motion, time * device.particles.speed);

    const vec3 scaled_ray = scale_vector(task.ray, 1.0f / device.particles.scale);
    const vec3 reference  = scale_vector(add_vector(task.origin, motion_offset), 1.0f / device.particles.scale);

    OptixKernelFunctionParticleTracePayload payload;
    payload.depth       = tmax;
    payload.instance_id = HIT_TYPE_REJECT;

    float t = 64.0f * eps;

    float3 inv_ray;
    inv_ray.x = 1.0f / ((fabsf(scaled_ray.x) > eps) ? scaled_ray.x : copysignf(eps, scaled_ray.x));
    inv_ray.y = 1.0f / ((fabsf(scaled_ray.y) > eps) ? scaled_ray.y : copysignf(eps, scaled_ray.y));
    inv_ray.z = 1.0f / ((fabsf(scaled_ray.z) > eps) ? scaled_ray.z : copysignf(eps, scaled_ray.z));

    for (int i = 0; i < 8; i++) {
      vec3 p = add_vector(reference, scale_vector(scaled_ray, t));

      // Map our current point into the particle tiling coordinate system
      p.x = p.x - floorf(p.x);
      p.y = p.y - floorf(p.y);
      p.z = p.z - floorf(p.z);

      optixKernelFunctionParticleTrace(
        device.optix_bvh_particles, p, scaled_ray, 0.0f, tmax, 0.0f, OptixVisibilityMask(0xFFFF), OPTIX_RAY_FLAG_NONE, payload);

      if (payload.depth < tmax)
        break;

      const float tx = inv_ray.x * (((scaled_ray.x < 0.0f) ? 0.0f : 1.0f) - p.x);
      const float ty = inv_ray.y * (((scaled_ray.y < 0.0f) ? 0.0f : 1.0f) - p.y);
      const float tz = inv_ray.z * (((scaled_ray.z < 0.0f) ? 0.0f : 1.0f) - p.z);

      const float step = fminf(fminf(tx, ty), tz) + 128.0f * eps;

      t += step;
      tmax -= step;

      if (tmax < 0.0f)
        break;
    }

    if (payload.instance_id != HIT_TYPE_REJECT) {
      // Hit ID contains the triangle ID but we only store the actual particle / quad ID
      payload.instance_id = HIT_TYPE_PARTICLE_MIN + (payload.instance_id >> 1);

      const TriangleHandle handle = triangle_handle_get(payload.instance_id, 0);

      triangle_handle_store(handle, offset);
      trace_depth_store(t + payload.depth, offset);
    }
  }
}
