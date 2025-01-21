#ifndef CU_LUMINARY_OPTIX_CLOSESTHIT_H
#define CU_LUMINARY_OPTIX_CLOSESTHIT_H

#include "optix_common.cuh"
#include "optix_utils.cuh"

#define OPTIX_CLOSESTHIT_FUNC_NAME(X) __closesthit__##X

////////////////////////////////////////////////////////////////////
// OPTIX_SBT_OFFSET_GEOMETRY_TRACE
////////////////////////////////////////////////////////////////////

extern "C" __global__ void OPTIX_CLOSESTHIT_FUNC_NAME(geometry_trace)() {
  const float depth = optixGetRayTmax();

  optixSetPayloadDepth(depth);

  TriangleHandle tri_handle;

  tri_handle.instance_id = optixGetInstanceId();
  tri_handle.tri_id      = optixGetPrimitiveIndex();

  optixSetPayloadTriangleHandle(tri_handle);
}

////////////////////////////////////////////////////////////////////
// OPTIX_SBT_OFFSET_PARTICLE_TRACE
////////////////////////////////////////////////////////////////////

extern "C" __global__ void OPTIX_CLOSESTHIT_FUNC_NAME(particle_trace)() {
  optixSetPayloadGeneric(OPTIX_PAYLOAD_DEPTH, __float_as_uint(optixGetRayTmax()));
  optixSetPayloadGeneric(OPTIX_PAYLOAD_INSTANCE_ID, optixGetPrimitiveIndex());
}

////////////////////////////////////////////////////////////////////
// OPTIX_SBT_OFFSET_LIGHT_BSDF_TRACE
////////////////////////////////////////////////////////////////////

extern "C" __global__ void OPTIX_CLOSESTHIT_FUNC_NAME(light_bsdf_trace)() {
  optixSetPayloadGeneric(OPTIX_PAYLOAD_TRIANGLE_ID, optixGetPrimitiveIndex());
}

////////////////////////////////////////////////////////////////////
// OPTIX_SBT_OFFSET_SHADOW_TRACE
////////////////////////////////////////////////////////////////////

extern "C" __global__ void OPTIX_CLOSESTHIT_FUNC_NAME(shadow_trace)() {
  optixSetPayloadGeneric(OPTIX_PAYLOAD_TRIANGLE_HANDLE, HIT_TYPE_REJECT);
}

#endif /* CU_LUMINARY_OPTIX_CLOSESTHIT_H */
