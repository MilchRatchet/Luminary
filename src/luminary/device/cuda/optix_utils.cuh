#ifndef CU_OPTIX_UTILS_H
#define CU_OPTIX_UTILS_H

#include "../optix_shared.h"
#include "utils.cuh"

enum OptixTraceStatus { OPTIX_TRACE_STATUS_EXECUTE, OPTIX_TRACE_STATUS_ABORT, OPTIX_TRACE_STATUS_OPTIONAL_UNUSED } typedef OptixTraceStatus;

#ifdef OPTIX_KERNEL

////////////////////////////////////////////////////////////////////
// Payload Type bitmasks
////////////////////////////////////////////////////////////////////

enum OptixKernelFunctionPayloadTypeID {
  OPTIX_KERNEL_FUNCTION_PAYLOAD_TYPE_ID_GEOMETRY_TRACE   = (1u << OPTIX_KERNEL_FUNCTION_GEOMETRY_TRACE),
  OPTIX_KERNEL_FUNCTION_PAYLOAD_TYPE_ID_PARTICLE_TRACE   = (1u << OPTIX_KERNEL_FUNCTION_PARTICLE_TRACE),
  OPTIX_KERNEL_FUNCTION_PAYLOAD_TYPE_ID_LIGHT_BSDF_TRACE = (1u << OPTIX_KERNEL_FUNCTION_LIGHT_BSDF_TRACE),
  OPTIX_KERNEL_FUNCTION_PAYLOAD_TYPE_ID_SHADOW_TRACE     = (1u << OPTIX_KERNEL_FUNCTION_SHADOW_TRACE),
  OPTIX_KERNEL_FUNCTION_PAYLOAD_TYPE_ID_SHADOW_SUN_TRACE = (1u << OPTIX_KERNEL_FUNCTION_SHADOW_SUN_TRACE)
} typedef OptixKernelFunctionPayloadTypeID;

#define OPTIX_TRACE_PAYLOAD_ID(__macro_optix_kernel_function_name) \
  ((OptixPayloadTypeID) (OPTIX_KERNEL_FUNCTION_PAYLOAD_TYPE_ID_##__macro_optix_kernel_function_name))

////////////////////////////////////////////////////////////////////
// SBT Offsets
////////////////////////////////////////////////////////////////////

enum OptixKernelFunctionSBTOffset {
  OPTIX_KERNEL_FUNCTION_SBT_OFFSET_GEOMETRY_TRACE   = OPTIX_KERNEL_FUNCTION_GEOMETRY_TRACE,
  OPTIX_KERNEL_FUNCTION_SBT_OFFSET_PARTICLE_TRACE   = OPTIX_KERNEL_FUNCTION_PARTICLE_TRACE,
  OPTIX_KERNEL_FUNCTION_SBT_OFFSET_LIGHT_BSDF_TRACE = OPTIX_KERNEL_FUNCTION_LIGHT_BSDF_TRACE,
  OPTIX_KERNEL_FUNCTION_SBT_OFFSET_SHADOW_TRACE     = OPTIX_KERNEL_FUNCTION_SHADOW_TRACE,
  OPTIX_KERNEL_FUNCTION_SBT_OFFSET_SHADOW_SUN_TRACE = OPTIX_KERNEL_FUNCTION_SHADOW_SUN_TRACE
} typedef OptixKernelFunctionSBTOffset;

#define OPTIX_TRACE_SBT_OFFSET(__macro_optix_kernel_function_name) \
  ((uint32_t) (OPTIX_KERNEL_FUNCTION_SBT_OFFSET_##__macro_optix_kernel_function_name))

////////////////////////////////////////////////////////////////////
// Trace call definitions
////////////////////////////////////////////////////////////////////

static __forceinline__ __device__ void optixKernelFunctionGeometryTrace(
  const OptixTraversableHandle handle, const vec3 origin, const vec3 ray, const float tmin, const float tmax, const float rayTime,
  const OptixVisibilityMask visibilityMask, const uint32_t rayFlags, const OptixTraceStatus status,
  OptixKernelFunctionGeometryTracePayload& payload) {
  const float actual_tmax = (status == OPTIX_TRACE_STATUS_EXECUTE) ? tmax : -1.0f;
  optixTrace(
    OPTIX_TRACE_PAYLOAD_ID(GEOMETRY_TRACE), handle, make_float3(origin.x, origin.y, origin.z), make_float3(ray.x, ray.y, ray.z), tmin,
    actual_tmax, rayTime, visibilityMask, rayFlags, OPTIX_TRACE_SBT_OFFSET(GEOMETRY_TRACE), 0, 0, payload.v0, payload.v1, payload.v2);
}

static __forceinline__ __device__ void optixKernelFunctionParticleTrace(
  const OptixTraversableHandle handle, const vec3 origin, const vec3 ray, const float tmin, const float tmax, const float rayTime,
  const OptixVisibilityMask visibilityMask, const uint32_t rayFlags, const OptixTraceStatus status,
  OptixKernelFunctionParticleTracePayload& payload) {
  const float actual_tmax = (status == OPTIX_TRACE_STATUS_EXECUTE) ? tmax : -1.0f;
  optixTrace(
    OPTIX_TRACE_PAYLOAD_ID(PARTICLE_TRACE), handle, make_float3(origin.x, origin.y, origin.z), make_float3(ray.x, ray.y, ray.z), tmin,
    actual_tmax, rayTime, visibilityMask, rayFlags, OPTIX_TRACE_SBT_OFFSET(PARTICLE_TRACE), 0, 0, payload.v0, payload.v1);
}

static __forceinline__ __device__ void optixKernelFunctionLightBSDFTrace(
  const OptixTraversableHandle handle, const vec3 origin, const vec3 ray, const float tmin, const float tmax, const float rayTime,
  const OptixVisibilityMask visibilityMask, const uint32_t rayFlags, const OptixTraceStatus status,
  OptixKernelFunctionLightBSDFTracePayload& payload) {
  const float actual_tmax = (status == OPTIX_TRACE_STATUS_EXECUTE) ? tmax : -1.0f;
  optixTrace(
    OPTIX_TRACE_PAYLOAD_ID(LIGHT_BSDF_TRACE), handle, make_float3(origin.x, origin.y, origin.z), make_float3(ray.x, ray.y, ray.z), tmin,
    actual_tmax, rayTime, visibilityMask, rayFlags, OPTIX_TRACE_SBT_OFFSET(LIGHT_BSDF_TRACE), 0, 0, payload.v0);
}

static __forceinline__ __device__ void optixKernelFunctionShadowTrace(
  const OptixTraversableHandle handle, const vec3 origin, const vec3 ray, const float tmin, const float tmax, const float rayTime,
  const OptixVisibilityMask visibilityMask, const uint32_t rayFlags, const OptixTraceStatus status,
  OptixKernelFunctionShadowTracePayload& payload) {
  const float actual_tmax = (status == OPTIX_TRACE_STATUS_EXECUTE) ? tmax : -1.0f;
  optixTrace(
    OPTIX_TRACE_PAYLOAD_ID(SHADOW_TRACE), handle, make_float3(origin.x, origin.y, origin.z), make_float3(ray.x, ray.y, ray.z), tmin,
    actual_tmax, rayTime, visibilityMask, rayFlags, OPTIX_TRACE_SBT_OFFSET(SHADOW_TRACE), 0, 0, payload.v0, payload.v1, payload.v2,
    payload.v3, payload.v4);
}

static __forceinline__ __device__ void optixKernelFunctionShadowSunTrace(
  const OptixTraversableHandle handle, const vec3 origin, const vec3 ray, const float tmin, const float tmax, const float rayTime,
  const OptixVisibilityMask visibilityMask, const uint32_t rayFlags, const OptixTraceStatus status,
  OptixKernelFunctionShadowSunTracePayload& payload) {
  const float actual_tmax = (status == OPTIX_TRACE_STATUS_EXECUTE) ? tmax : -1.0f;
  optixTrace(
    OPTIX_TRACE_PAYLOAD_ID(SHADOW_SUN_TRACE), handle, make_float3(origin.x, origin.y, origin.z), make_float3(ray.x, ray.y, ray.z), tmin,
    actual_tmax, rayTime, visibilityMask, rayFlags, OPTIX_TRACE_SBT_OFFSET(SHADOW_SUN_TRACE), 0, 0, payload.v0, payload.v1, payload.v2);
}

////////////////////////////////////////////////////////////////////
// Utility functions
////////////////////////////////////////////////////////////////////

__device__ TriangleHandle optixGetTriangleHandle() {
  return triangle_handle_get(optixGetInstanceId(), optixGetPrimitiveIndex());
}

////////////////////////////////////////////////////////////////////
// Payload generics
////////////////////////////////////////////////////////////////////

// IMPORTANT: Verify that OptiX still implements these intrinsics in this way whenever updating OptiX.
// IMPORTANT: I suppose OptiX implemented these functions in this particular way because the index must be a compile time constant,
// so I better make sure that I honor that :D
// Intellisense does not understand the usage of volatile here :(

#ifndef __INTELLISENSE__
static __device__ uint32_t optixGetPayloadGeneric(const uint32_t index) {
  uint32_t result;
  asm volatile("call (%0), _optix_get_payload, (%1);" : "=r"(result) : "r"(index) :);
  return result;
}

static __forceinline__ __device__ void optixSetPayloadGeneric(const uint32_t index, const uint32_t value) {
  asm volatile("call _optix_set_payload, (%0, %1);" : : "r"(index), "r"(value) :);
}
#else
// Dummy functions for intellisense
static __device__ uint32_t optixGetPayloadGeneric(const uint32_t index) {
  return 0;
}

static __forceinline__ __device__ void optixSetPayloadGeneric(const uint32_t index, const uint32_t value) {
}
#endif

////////////////////////////////////////////////////////////////////
// Payload definitions
////////////////////////////////////////////////////////////////////

__device__ TriangleHandle optixGetPayloadTriangleHandle(const uint32_t kernel_function_triangle_handle_index) {
  TriangleHandle handle;
  handle.instance_id = optixGetPayloadGeneric(kernel_function_triangle_handle_index);
  handle.tri_id      = optixGetPayloadGeneric(kernel_function_triangle_handle_index + 1);

  return handle;
}

__device__ void optixSetPayloadTriangleHandle(const uint32_t kernel_function_triangle_handle_index, const TriangleHandle handle) {
  optixSetPayloadGeneric(kernel_function_triangle_handle_index, handle.instance_id);
  optixSetPayloadGeneric(kernel_function_triangle_handle_index + 1, handle.tri_id);
}

__device__ uint32_t optixGetPayloadInstanceID(const uint32_t kernel_function_instance_id_index) {
  return optixGetPayloadGeneric(kernel_function_instance_id_index);
}

__device__ void optixSetPayloadInstanceID(const uint32_t kernel_function_instance_id_index, const uint32_t instance_id) {
  optixSetPayloadGeneric(kernel_function_instance_id_index, instance_id);
}

__device__ uint32_t optixGetPayloadTriangleID(const uint32_t kernel_function_triangle_id_index) {
  return optixGetPayloadGeneric(kernel_function_triangle_id_index);
}

__device__ void optixSetPayloadTriangleID(const uint32_t kernel_function_triangle_id_index, const uint32_t triangle_id) {
  optixSetPayloadGeneric(kernel_function_triangle_id_index, triangle_id);
}

__device__ RGBF optixGetPayloadColor(const uint32_t kernel_function_color_index) {
  RGBF color;
  color.r = __uint_as_float(optixGetPayloadGeneric(kernel_function_color_index + 0));
  color.g = __uint_as_float(optixGetPayloadGeneric(kernel_function_color_index + 1));
  color.b = __uint_as_float(optixGetPayloadGeneric(kernel_function_color_index + 2));

  return color;
}

__device__ void optixSetPayloadColor(const uint32_t kernel_function_color_index, const RGBF color) {
  optixSetPayloadGeneric(kernel_function_color_index + 0, __float_as_uint(color.r));
  optixSetPayloadGeneric(kernel_function_color_index + 1, __float_as_uint(color.g));
  optixSetPayloadGeneric(kernel_function_color_index + 2, __float_as_uint(color.b));
}

__device__ float optixGetPayloadDepth(const uint32_t kernel_function_depth_index) {
  return __uint_as_float(optixGetPayloadGeneric(kernel_function_depth_index));
}

__device__ void optixSetPayloadDepth(const uint32_t kernel_function_depth_index, const float depth) {
  optixSetPayloadGeneric(kernel_function_depth_index, __float_as_uint(depth));
}

#endif

#endif /* CU_OPTIX_UTILS_H */
