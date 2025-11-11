#ifndef CU_UTILS_H
#define CU_UTILS_H

#include <cuda_fp16.h>

#include "../device_utils.h"
#include "../kernel_args.h"

// Part of compile configuration
#ifndef LUMINARY_MIN_BLOCKS_PER_SM
#define LUMINARY_MIN_BLOCKS_PER_SM 1
#endif /* LUMINARY_MIN_BLOCKS_PER_SM */

#define NUM_THREADS (THREADS_PER_BLOCK * device.config.num_blocks)

#define WARP_SIZE_LOG 5
#define WARP_SIZE (1 << WARP_SIZE_LOG)
#define WARP_SIZE_MASK (WARP_SIZE - 1)

#define NUM_WARPS (NUM_THREADS >> WARP_SIZE_LOG)

#ifndef OPTIX_KERNEL
#define THREAD_ID (threadIdx.x + blockIdx.x * blockDim.x)
#else /* !OPTIX_KERNEL */
#define THREAD_ID (optixGetLaunchIndex().x + optixGetLaunchIndex().y * optixGetLaunchDimensions().x)
#endif /* OPTIX_KERNEL */

#ifdef OPTIX_KERNEL
#define TASK_ID optixGetLaunchIndex().z
#endif /* OPTIX_KERNEL */

#define LUMINARY_KERNEL extern "C" __global__ __launch_bounds__(THREADS_PER_BLOCK, LUMINARY_MIN_BLOCKS_PER_SM)
#define LUMINARY_KERNEL_NO_BOUNDS extern "C" __global__

#define LUMINARY_FUNCTION __device__ __forceinline__
#define LUMINARY_FUNCTION_NO_INLINE __device__ __noinline__

#define LUMINARY_ENABLE_SMEM_SPILLING asm volatile(".pragma \"enable_smem_spilling\";")

#ifndef eps
#define eps FLT_EPSILON
#endif /* eps */

#define GEOMETRY_DELTA_PATH_CUTOFF (0.05f)
#define BSDF_ROUGHNESS_CLAMP (2e-2f)

enum OptixTraceStatus { OPTIX_TRACE_STATUS_EXECUTE, OPTIX_TRACE_STATUS_ABORT, OPTIX_TRACE_STATUS_OPTIONAL_UNUSED } typedef OptixTraceStatus;

enum HitType : uint32_t {
  HIT_TYPE_INVALID           = 0xFFFFFFFFu,
  HIT_TYPE_SKY               = 0xFFFFFFFEu,
  HIT_TYPE_OCEAN             = 0xFFFFFFFDu,
  HIT_TYPE_PARTICLE          = 0xFFFFFFFCu,
  HIT_TYPE_LIGHT_BSDF_HINT   = 0xFFFFFFFBu,
  HIT_TYPE_REJECT            = 0xFFFFFFF0u,
  HIT_TYPE_PARTICLE_MAX      = 0xeFFFFFFFu,
  HIT_TYPE_PARTICLE_MIN      = 0x80000000u,
  HIT_TYPE_PARTICLE_MASK     = 0x7FFFFFFFu,
  HIT_TYPE_TRIANGLE_ID_LIMIT = 0x7FFFFFFFu,
  HIT_TYPE_VOLUME_BASE       = 0xFFFE0000u,
  HIT_TYPE_VOLUME_MASK       = 0x0000FFFFu,
  HIT_TYPE_VOLUME_MAX        = 0xFFFEFFFFu
} typedef HitType;

enum ShadingTaskIndex {
  SHADING_TASK_INDEX_GEOMETRY,
  SHADING_TASK_INDEX_PARTICLE,
  SHADING_TASK_INDEX_OCEAN,
  SHADING_TASK_INDEX_VOLUME,
  SHADING_TASK_INDEX_SKY,
  SHADING_TASK_INDEX_TOTAL,
  SHADING_TASK_INDEX_INVALID = 0xFF
} typedef ShadingTaskIndex;

#define TASK_ADDRESS_OFFSET_IMPL(__internal_macro_shading_task_index) (NUM_THREADS * __internal_macro_shading_task_index + THREAD_ID)

#define TASK_ADDRESS_OFFSET_GEOMETRY TASK_ADDRESS_OFFSET_IMPL(SHADING_TASK_INDEX_GEOMETRY)
#define TASK_ADDRESS_OFFSET_PARTICLE TASK_ADDRESS_OFFSET_IMPL(SHADING_TASK_INDEX_PARTICLE)
#define TASK_ADDRESS_OFFSET_OCEAN TASK_ADDRESS_OFFSET_IMPL(SHADING_TASK_INDEX_OCEAN)
#define TASK_ADDRESS_OFFSET_VOLUME TASK_ADDRESS_OFFSET_IMPL(SHADING_TASK_INDEX_VOLUME)
#define TASK_ADDRESS_OFFSET_SKY TASK_ADDRESS_OFFSET_IMPL(SHADING_TASK_INDEX_SKY)

#define VOLUME_HIT_ID_CHECK(X) (((X) >= HIT_TYPE_VOLUME_BASE) && ((X) <= HIT_TYPE_VOLUME_MAX))
#define VOLUME_ID_TO_HIT_ID(X) (HIT_TYPE_VOLUME_BASE | (X))
#define PARTICLE_HIT_ID_CHECK(X) (((X) <= HIT_TYPE_PARTICLE_MAX) && ((X) >= HIT_TYPE_PARTICLE_MIN))
#define IS_PRIMARY_RAY (device.state.depth == 0)

//
// Usage documentation:
//
// STATE_FLAG_DELTA_PATH: This flag is set for paths whose vertices generated bounce rays only from delta (or near-delta) distributions.
//                        This flag is used for firefly clamping as it only applies to light gathered on path suffixes of non-delta paths.
//
// STATE_FLAG_CAMERA_DIRECTION: This flag is set while the current path is just a line along the original camera direction.
//                              This flag is used to allow light to be gathered through non-refractive transparencies when coming directly
//                              from the camera where no DL is executed.
//
// STATE_FLAG_VOLUME_SCATTERED: This flag is set for paths that have at least one vertex that is a volume scattering event.
//                              This flag is used to limit ocean volumes to single scattering for performance reasons.
//                              This flag is used to limit DL after volume scattering for convergence reasons.
//
// STATE_FLAG_ALLOW_EMISSION: This flag is set for rays that are allowed to include emission in bounce rays.
//                            This flag is used on the ocean surface because there is no DL on it.
//
// STATE_FLAG_ALLOW_AMBIENT: This flag is set for rays that are allowed to include sky contribution
//
// STATE_FLAG_USE_IGNORE_HANDLE: This flag is set for rays that need to ignore a triangle in the next trace.
//                               This flag is used to avoid loading the DeviceTaskAux data which currently only holds this handle
//                               and is not written/sorted/loaded consistently for performance reasons. This will be removed when
//                               nested dielectrics are implemented.
//

enum StateFlag {
  STATE_FLAG_DELTA_PATH        = 0b00000001u,
  STATE_FLAG_CAMERA_DIRECTION  = 0b00000010u,
  STATE_FLAG_VOLUME_SCATTERED  = 0b00000100u,
  STATE_FLAG_ALLOW_EMISSION    = 0b00001000u,
  STATE_FLAG_ALLOW_AMBIENT     = 0b00010000u,
  STATE_FLAG_USE_IGNORE_HANDLE = 0b00100000u
} typedef StateFlag;

struct OptixRaytraceResult {
  TriangleHandle handle;
  float depth;
} typedef OptixRaytraceResult;

//===========================================================================================
// Device Variables
//===========================================================================================

#ifndef OPTIX_KERNEL
__constant__ DeviceConstantMemory device;
#else
extern "C" static __constant__ DeviceConstantMemory device;
#endif

//===========================================================================================
// PathID
//===========================================================================================

#define PATH_ID_SENSOR_BITS 14
#define PATH_ID_SAMPLE_BITS 20
#define PATH_ID_EXTRA_BITS (16 - PATH_ID_SENSOR_BITS)
#define PATH_ID_SENSOR_MASK ((1u << PATH_ID_SENSOR_BITS) - 1)
#define PATH_ID_SAMPLE_MASK ((1u << PATH_ID_SAMPLE_BITS) - 1)
#define PATH_ID_EXTRA_MASK ((1u << PATH_ID_EXTRA_BITS) - 1)

LUMINARY_FUNCTION PathID path_id_get(const uint32_t x, const uint32_t y, const uint32_t sample_id) {
  PathID path_id;
  path_id.x = (x & PATH_ID_SENSOR_MASK) | ((sample_id >> (16 + PATH_ID_EXTRA_BITS * 0)) & PATH_ID_EXTRA_MASK);
  path_id.y = (y & PATH_ID_SENSOR_MASK) | ((sample_id >> (16 + PATH_ID_EXTRA_BITS * 1)) & PATH_ID_EXTRA_MASK);
  path_id.z = sample_id & PATH_ID_SAMPLE_MASK;

  return path_id;
}

LUMINARY_FUNCTION ushort2 path_id_get_pixel(const PathID& path_id) {
  return make_ushort2(path_id.x & PATH_ID_SENSOR_MASK, path_id.y & PATH_ID_SENSOR_MASK);
}

LUMINARY_FUNCTION uint32_t path_id_get_pixel_index(const PathID& path_id) {
  const ushort2 pixel = path_id_get_pixel(path_id);

  const uint32_t x = pixel.x;
  const uint32_t y = pixel.y;

  return x + device.settings.width * y;
}

LUMINARY_FUNCTION uint32_t path_id_get_sample_id(const PathID& path_id) {
  const uint32_t x = path_id.x;
  const uint32_t y = path_id.y;

  uint32_t sample_id = path_id.z;
  sample_id |= (x >> PATH_ID_SENSOR_BITS) << (16 + PATH_ID_EXTRA_BITS * 0);
  sample_id |= (y >> PATH_ID_SENSOR_BITS) << (16 + PATH_ID_EXTRA_BITS * 1);

  return sample_id;
}

//===========================================================================================
// Functions
//===========================================================================================

#ifdef __builtin_assume
#define LUMINARY_ASSUME(__internal_macro_expression) __builtin_assume(__internal_macro_expression)
#else /* __builtin_assume */
#define LUMINARY_ASSUME(__internal_macro_expression) __assume(__internal_macro_expression)
#endif /* !__builtin_assume */

#define UTILS_NO_PIXEL_SELECTED (make_ushort2(0xFFFF, 0xFFFF))

LUMINARY_FUNCTION bool is_selected_pixel(const ushort2 index) {
  if (device.state.user_selected_x == UTILS_NO_PIXEL_SELECTED.x && device.state.user_selected_y == UTILS_NO_PIXEL_SELECTED.y)
    return false;

  // Only the top left subpixel of a pixel can be selected.
  return (index.x == device.state.user_selected_x && index.y == device.state.user_selected_y);
}

LUMINARY_FUNCTION bool is_selected_pixel_lenient(const ushort2 index) {
  if (device.state.user_selected_x == UTILS_NO_PIXEL_SELECTED.x && device.state.user_selected_y == UTILS_NO_PIXEL_SELECTED.y)
    return true;

  return is_selected_pixel(index);
}

LUMINARY_FUNCTION bool is_center_pixel(const PathID& path_id) {
  const ushort2 pixel = path_id_get_pixel(path_id);

  const uint32_t width  = device.settings.width >> device.settings.supersampling;
  const uint32_t height = device.settings.height >> device.settings.supersampling;

  return ((pixel.x == width) && (pixel.y == height));
}

LUMINARY_FUNCTION bool is_non_finite(const float a) {
#ifndef __INTELLISENSE__
  return isnan(a) || isinf(a);
#else
  const bool is_nan = a != a;
  const bool is_inf = a < -FLT_MAX || a > FLT_MAX;

  return is_nan || is_inf;
#endif
}

#define LIGHTS_ARE_PRESENT (device.ptrs.light_tree_root != nullptr)

#define HANDLE_DEVICE_ABORT()                                                                                         \
  if (__ldcv(device.ptrs.abort_flag) != 0 && ((device.state.undersampling & UNDERSAMPLING_FIRST_SAMPLE_MASK) == 0)) { \
    return;                                                                                                           \
  }

LUMINARY_FUNCTION ShadingTaskIndex shading_task_index_from_instance_id(const uint32_t instance_id) {
  if (instance_id <= HIT_TYPE_TRIANGLE_ID_LIMIT) {
    return SHADING_TASK_INDEX_GEOMETRY;
  }
  else if (instance_id == HIT_TYPE_OCEAN) {
    return SHADING_TASK_INDEX_OCEAN;
  }
  else if (VOLUME_HIT_ID_CHECK(instance_id)) {
    return SHADING_TASK_INDEX_VOLUME;
  }
  else if (instance_id <= HIT_TYPE_PARTICLE_MAX) {
    return SHADING_TASK_INDEX_PARTICLE;
  }
  else if (instance_id == HIT_TYPE_SKY) {
    return SHADING_TASK_INDEX_SKY;
  }
  else {
    return SHADING_TASK_INDEX_INVALID;
  }
}

//===========================================================================================
// Debug utils
//===========================================================================================

// #define UTILS_DEBUG_MODE

#ifdef UTILS_DEBUG_MODE

#define UTILS_DEBUG_NAN_COLOR (get_color(1.0f, 0.0f, 0.0f))

LUMINARY_FUNCTION bool _utils_debug_nans(const RGBF color, const char* func, const uint32_t line, const char* var) {
  const float sum_color = color.r + color.g + color.b;
  if (isnan(sum_color) || isinf(sum_color)) {
    printf("[%s:%u] Failed NaN check. %s = (%f %f %f).\n", func, line, var, color.r, color.g, color.b);
    return true;
  }

  return false;
}

LUMINARY_FUNCTION bool _utils_debug_nans(const float value, const char* func, const uint32_t line, const char* var) {
  if (isnan(value) || isinf(value)) {
    printf("[%s:%u] Failed NaN check. %s = %f.\n", func, line, var, value);

    return true;
  }

  return false;
}

#define UTILS_CHECK_NANS(pixel, var) (is_selected_pixel_lenient(pixel) && _utils_debug_nans(var, __func__, __LINE__, #var))

#else /* UTILS_DEBUG_MODE */

#define UTILS_DEBUG_NAN_COLOR (get_color(0.0f, 0.0f, 0.0f))

#define UTILS_CHECK_NANS(pixel, var)

#endif /* !UTILS_DEBUG_MODE */

//===========================================================================================
// Triangle addressing
//===========================================================================================

/* instance_id refers to the light instance id, the actual instance_id is obtain through device.ptrs.light_instance_map. */
typedef TriangleHandle LightTriangleHandle;

LUMINARY_FUNCTION TriangleHandle triangle_handle_get(const uint32_t instance_id, const uint32_t tri_id) {
  TriangleHandle handle;

  handle.instance_id = instance_id;
  handle.tri_id      = tri_id;

  return handle;
}

LUMINARY_FUNCTION bool triangle_handle_equal(const TriangleHandle handle1, const TriangleHandle handle2) {
  return (handle1.instance_id == handle2.instance_id) && (handle1.tri_id == handle2.tri_id);
}

#define TRIANGLE_HANDLE_INVALID (triangle_handle_get(INSTANCE_ID_INVALID, 0))

#endif /* CU_UTILS_H */
