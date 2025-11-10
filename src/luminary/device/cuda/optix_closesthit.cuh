#ifndef CU_LUMINARY_OPTIX_CLOSESTHIT_H
#define CU_LUMINARY_OPTIX_CLOSESTHIT_H

#if defined(OPTIX_KERNEL)

#include "optix_common.cuh"
#include "optix_utils.cuh"

#define OPTIX_CLOSESTHIT_FUNC_NAME(X) __closesthit__##X

////////////////////////////////////////////////////////////////////
// OPTIX_SBT_OFFSET_GEOMETRY_TRACE
////////////////////////////////////////////////////////////////////

extern "C" __global__ void OPTIX_CLOSESTHIT_FUNC_NAME(geometry_trace)() {
  optixSetPayloadTypes(OPTIX_KERNEL_FUNCTION_PAYLOAD_TYPE_ID_GEOMETRY_TRACE);

  optixSetPayloadFloat(OPTIX_KERNEL_FUNCTION_GEOMETRY_TRACE_PAYLOAD_VALUE_DEPTH, optixGetRayTmax());

  TriangleHandle tri_handle;

  tri_handle.instance_id = optixGetInstanceId();
  tri_handle.tri_id      = optixGetPrimitiveIndex();

  optixSetPayloadTriangleHandle(OPTIX_KERNEL_FUNCTION_GEOMETRY_TRACE_PAYLOAD_VALUE_TRIANGLE_HANDLE, tri_handle);
}

////////////////////////////////////////////////////////////////////
// OPTIX_SBT_OFFSET_PARTICLE_TRACE
////////////////////////////////////////////////////////////////////

extern "C" __global__ void OPTIX_CLOSESTHIT_FUNC_NAME(particle_trace)() {
  optixSetPayloadTypes(OPTIX_KERNEL_FUNCTION_PAYLOAD_TYPE_ID_PARTICLE_TRACE);

  optixSetPayloadFloat(OPTIX_KERNEL_FUNCTION_PARTICLE_TRACE_PAYLOAD_VALUE_DEPTH, optixGetRayTmax());
  optixSetPayloadInstanceID(OPTIX_KERNEL_FUNCTION_PARTICLE_TRACE_PAYLOAD_VALUE_INSTANCE_ID, optixGetPrimitiveIndex());
}

////////////////////////////////////////////////////////////////////
// OPTIX_SBT_OFFSET_SHADOW_TRACE
////////////////////////////////////////////////////////////////////

extern "C" __global__ void OPTIX_CLOSESTHIT_FUNC_NAME(shadow_trace)() {
  optixSetPayloadTypes(OPTIX_KERNEL_FUNCTION_PAYLOAD_TYPE_ID_SHADOW_TRACE);

  optixSetPayloadInstanceID(OPTIX_KERNEL_FUNCTION_SHADOW_TRACE_PAYLOAD_VALUE_TRIANGLE_HANDLE, HIT_TYPE_REJECT);
}

////////////////////////////////////////////////////////////////////
// OPTIX_SBT_OFFSET_SHADOW_SUN_TRACE
////////////////////////////////////////////////////////////////////

extern "C" __global__ void OPTIX_CLOSESTHIT_FUNC_NAME(shadow_sun_trace)() {
  optixSetPayloadTypes(OPTIX_KERNEL_FUNCTION_PAYLOAD_TYPE_ID_SHADOW_SUN_TRACE);

  optixSetPayloadColor(OPTIX_KERNEL_FUNCTION_SHADOW_SUN_TRACE_PAYLOAD_VALUE_THROUGHPUT, splat_color(0.0f));
}

////////////////////////////////////////////////////////////////////
// OPTIX_SBT_OFFSET_LIGHT_BSDF_TRACE
////////////////////////////////////////////////////////////////////

extern "C" __global__ void OPTIX_CLOSESTHIT_FUNC_NAME(light_bsdf_trace)() {
  optixSetPayloadTypes(OPTIX_KERNEL_FUNCTION_PAYLOAD_TYPE_ID_LIGHT_BSDF_TRACE);
}

#endif /* OPTIX_KERNEL */

#endif /* CU_LUMINARY_OPTIX_CLOSESTHIT_H */
