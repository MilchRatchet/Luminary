#ifndef CU_OPTIX_UTILS_H
#define CU_OPTIX_UTILS_H

#include "utils.cuh"

#ifdef OPTIX_KERNEL
__device__ TriangleHandle optixGetTriangleHandle() {
  return triangle_handle_get(optixGetInstanceId(), optixGetPrimitiveIndex());
}

#define OPTIX_PAYLOAD_INDEX_REQUIRE(payload, index) static_assert(payload == index, "OptiX Payload is not in the correct index.")

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

#ifdef OPTIX_PAYLOAD_TRIANGLE_HANDLE
__device__ TriangleHandle optixGetPayloadTriangleHandle() {
  TriangleHandle handle;
  handle.instance_id = optixGetPayloadGeneric(OPTIX_PAYLOAD_TRIANGLE_HANDLE);
  handle.tri_id      = optixGetPayloadGeneric(OPTIX_PAYLOAD_TRIANGLE_HANDLE + 1);

  return handle;
}

__device__ void optixSetPayloadTriangleHandle(const TriangleHandle handle) {
  optixSetPayloadGeneric(OPTIX_PAYLOAD_TRIANGLE_HANDLE, handle.instance_id);
  optixSetPayloadGeneric(OPTIX_PAYLOAD_TRIANGLE_HANDLE + 1, handle.tri_id);
}
#endif

#ifdef OPTIX_PAYLOAD_COMPRESSED_ALPHA
__device__ CompressedAlpha optixGetPayloadCompressedAlpha() {
  CompressedAlpha alpha;
  alpha.data0 = optixGetPayloadGeneric(OPTIX_PAYLOAD_COMPRESSED_ALPHA);
  alpha.data1 = optixGetPayloadGeneric(OPTIX_PAYLOAD_COMPRESSED_ALPHA + 1);

  return alpha;
}

__device__ void optixSetPayloadCompressedAlpha(const CompressedAlpha alpha) {
  optixSetPayloadGeneric(OPTIX_PAYLOAD_COMPRESSED_ALPHA, alpha.data0);
  optixSetPayloadGeneric(OPTIX_PAYLOAD_COMPRESSED_ALPHA + 1, alpha.data1);
}
#endif

#ifdef OPTIX_PAYLOAD_DEPTH
__device__ float optixGetPayloadDepth() {
  return optixGetPayloadGeneric(OPTIX_PAYLOAD_DEPTH);
}

__device__ void optixSetPayloadDepth(const float depth) {
  optixSetPayloadGeneric(OPTIX_PAYLOAD_DEPTH, __float_as_uint(depth));
}
#endif

#endif

#endif /* CU_OPTIX_UTILS_H */
