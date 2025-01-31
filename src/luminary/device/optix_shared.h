#ifndef LUMINARY_OPTIX_SHARED_H
#define LUMINARY_OPTIX_SHARED_H

enum OptixKernelFunction {
  OPTIX_KERNEL_FUNCTION_GEOMETRY_TRACE,
  OPTIX_KERNEL_FUNCTION_PARTICLE_TRACE,
  OPTIX_KERNEL_FUNCTION_LIGHT_BSDF_TRACE,
  OPTIX_KERNEL_FUNCTION_SHADOW_TRACE,
  OPTIX_KERNEL_FUNCTION_COUNT
} typedef OptixKernelFunction;

struct OptixKernelFunctionGeometryTracePayload {
  union {
    float depth;
    unsigned int v0;
  };
  union {
    TriangleHandle handle;
    struct {
      unsigned int v1;
      unsigned int v2;
    };
  };
} typedef OptixKernelFunctionGeometryTracePayload;

enum OptixKernelFunctionGeometryTracePayloadValue {
  OPTIX_KERNEL_FUNCTION_GEOMETRY_TRACE_PAYLOAD_VALUE_DEPTH,
  OPTIX_KERNEL_FUNCTION_GEOMETRY_TRACE_PAYLOAD_VALUE_TRIANGLE_HANDLE,
  OPTIX_KERNEL_FUNCTION_GEOMETRY_TRACE_PAYLOAD_VALUE_TRIANGLE_HANDLE2,
  OPTIX_KERNEL_FUNCTION_GEOMETRY_TRACE_PAYLOAD_VALUE_COUNT
} typedef OptixKernelFunctionGeometryTracePayloadValue;

struct OptixKernelFunctionParticleTracePayload {
  union {
    float depth;
    unsigned int v0;
  };
  union {
    uint32_t instance_id;
    unsigned int v1;
  };
} typedef OptixKernelFunctionParticleTracePayload;

enum OptixKernelFunctionParticleTracePayloadValue {
  OPTIX_KERNEL_FUNCTION_PARTICLE_TRACE_PAYLOAD_VALUE_DEPTH,
  OPTIX_KERNEL_FUNCTION_PARTICLE_TRACE_PAYLOAD_VALUE_INSTANCE_ID,
  OPTIX_KERNEL_FUNCTION_PARTICLE_TRACE_PAYLOAD_VALUE_COUNT
} typedef OptixKernelFunctionParticleTracePayloadValue;

struct OptixKernelFunctionLightBSDFTracePayload {
  union {
    uint32_t triangle_id;
    unsigned int v0;
  };
} typedef OptixKernelFunctionLightBSDFTracePayload;

enum OptixKernelFunctionLightBSDFTracePayloadValue {
  OPTIX_KERNEL_FUNCTION_LIGHT_BSDF_TRACE_PAYLOAD_VALUE_TRIANGLE_ID,
  OPTIX_KERNEL_FUNCTION_LIGHT_BSDF_TRACE_PAYLOAD_VALUE_COUNT
} typedef OptixKernelFunctionLightBSDFTracePayloadValue;

struct OptixKernelFunctionShadowTracePayload {
  union {
    TriangleHandle handle;
    struct {
      unsigned int v0;
      unsigned int v1;
    };
  };
  union {
    RGBF throughput;
    struct {
      unsigned int v2;
      unsigned int v3;
      unsigned int v4;
    };
  };
} typedef OptixKernelFunctionShadowTracePayload;

enum OptixKernelFunctionShadowTracePayloadValue {
  OPTIX_KERNEL_FUNCTION_SHADOW_TRACE_PAYLOAD_VALUE_TRIANGLE_HANDLE,
  OPTIX_KERNEL_FUNCTION_SHADOW_TRACE_PAYLOAD_VALUE_TRIANGLE_HANDLE2,
  OPTIX_KERNEL_FUNCTION_SHADOW_TRACE_PAYLOAD_VALUE_THROUGHPUT,
  OPTIX_KERNEL_FUNCTION_SHADOW_TRACE_PAYLOAD_VALUE_THROUGHPUT2,
  OPTIX_KERNEL_FUNCTION_SHADOW_TRACE_PAYLOAD_VALUE_THROUGHPUT3,
  OPTIX_KERNEL_FUNCTION_SHADOW_TRACE_PAYLOAD_VALUE_COUNT
} typedef OptixKernelFunctionShadowTracePayloadValue;

#endif /* LUMINARY_OPTIX_SHARED_H */
