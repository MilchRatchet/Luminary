#define OPTIX_KERNEL

#define OPTIX_PAYLOAD_DEPTH 0
#define OPTIX_PAYLOAD_INSTANCE_ID 1

#include "math.cuh"
#include "memory.cuh"
#include "optix_utils.cuh"
#include "utils.cuh"

__device__ bool particle_opacity_cutout(const float2 coord) {
  const float dx = fabsf(coord.x - 0.5f);
  const float dy = fabsf(coord.y - 0.5f);

  const float r = dx * dx + dy * dy;

  return (r > 0.25f);
}

// Kernels must be named __[SEMANTIC]__..., for example, __raygen__...
// This can be found under function name prefix in the programming guide

extern "C" __global__ void __raygen__optix() {
  const uint16_t trace_task_count = device.ptrs.trace_counts[THREAD_ID];

  const float time         = quasirandom_sequence_1D_global(QUASI_RANDOM_TARGET_CAMERA_TIME);
  const vec3 motion        = angles_to_direction(device.particles.direction_altitude, device.particles.direction_azimuth);
  const vec3 motion_offset = scale_vector(motion, time * device.particles.speed);

  for (uint32_t i = 0; i < trace_task_count; i++) {
    const uint32_t offset    = get_task_address(i);
    const TraceTask task     = load_trace_task(offset);
    const TraceResult result = load_trace_result(offset);

    const vec3 scaled_ray = scale_vector(task.ray, 1.0f / device.particles.scale);
    const vec3 reference  = scale_vector(add_vector(task.origin, motion_offset), 1.0f / device.particles.scale);

    const float3 origin = make_float3(reference.x, reference.y, reference.z);
    const float3 ray    = make_float3(scaled_ray.x, scaled_ray.y, scaled_ray.z);

    float tmax = result.depth;

    unsigned int depth       = __float_as_uint(result.depth);
    unsigned int instance_id = result.instance_id;

    const unsigned int vis_mask = OptixVisibilityMask(0xFFFF);

    float t = 64.0f * eps;

    float3 inv_ray;
    inv_ray.x = 1.0f / ((fabsf(ray.x) > eps) ? ray.x : copysignf(eps, ray.x));
    inv_ray.y = 1.0f / ((fabsf(ray.y) > eps) ? ray.y : copysignf(eps, ray.y));
    inv_ray.z = 1.0f / ((fabsf(ray.z) > eps) ? ray.z : copysignf(eps, ray.z));

    for (int i = 0; i < 8; i++) {
      float3 p = make_float3(origin.x + ray.x * t, origin.y + ray.y * t, origin.z + ray.z * t);

      // Map our current point into the particle tiling coordinate system
      p.x = p.x - floorf(p.x);
      p.y = p.y - floorf(p.y);
      p.z = p.z - floorf(p.z);

      OPTIX_PAYLOAD_INDEX_REQUIRE(OPTIX_PAYLOAD_DEPTH, 0);
      OPTIX_PAYLOAD_INDEX_REQUIRE(OPTIX_PAYLOAD_INSTANCE_ID, 1);
      optixTrace(device.optix_bvh_particles, p, ray, 0.0f, tmax, 0.0f, vis_mask, OPTIX_RAY_FLAG_NONE, 0, 0, 0, depth, instance_id);

      const float intersection_dist = __uint_as_float(depth);

      if (intersection_dist < tmax) {
        // Hit ID contains the triangle ID but we only store the actual particle / quad ID
        instance_id = HIT_TYPE_PARTICLE_MIN + (instance_id >> 1);

        TraceResult trace_result;
        trace_result.depth       = t + intersection_dist;
        trace_result.instance_id = instance_id;
        trace_result.tri_id      = 0;

        store_trace_result(trace_result, offset);
        break;
      }
      else {
        const float tx = inv_ray.x * (((ray.x < 0.0f) ? 0.0f : 1.0f) - p.x);
        const float ty = inv_ray.y * (((ray.y < 0.0f) ? 0.0f : 1.0f) - p.y);
        const float tz = inv_ray.z * (((ray.z < 0.0f) ? 0.0f : 1.0f) - p.z);

        const float step = fminf(fminf(tx, ty), tz) + 128.0f * eps;

        t += step;
        tmax -= step;

        if (tmax < 0.0f)
          break;
      }
    }
  }
}

extern "C" __global__ void __anyhit__optix() {
  if (particle_opacity_cutout(optixGetTriangleBarycentrics())) {
    optixIgnoreIntersection();
  }
}

extern "C" __global__ void __closesthit__optix() {
  optixSetPayloadGeneric(OPTIX_PAYLOAD_DEPTH, __float_as_uint(optixGetRayTmax()));
  optixSetPayloadGeneric(OPTIX_PAYLOAD_INSTANCE_ID, optixGetPrimitiveIndex());
}
