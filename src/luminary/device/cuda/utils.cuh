#ifndef CU_UTILS_H
#define CU_UTILS_H

#include <cuda_fp16.h>

#include "../device_utils.h"
#include "../kernel_args.h"

#define NUM_THREADS (THREADS_PER_BLOCK * BLOCKS_PER_GRID)

#ifndef OPTIX_KERNEL
#define THREAD_ID (threadIdx.x + blockIdx.x * blockDim.x)
#else
#define THREAD_ID (optixGetLaunchIndex().x + optixGetLaunchIndex().y * optixGetLaunchDimensions().x)
#endif

#define LUMINARY_KERNEL extern "C" __global__ __launch_bounds__(THREADS_PER_BLOCK)
#define LUMINARY_KERNEL_NO_BOUNDS extern "C" __global__

#ifndef eps
#define eps 0.000001f
#endif /* eps */

#define GEOMETRY_DELTA_PATH_CUTOFF (0.05f)
#define BSDF_ROUGHNESS_CLAMP (0.025f)

enum HitType : uint32_t {
  HIT_TYPE_INVALID           = 0xFFFFFFFFu,  // TODO: Shift all values so that invalid is all bits set
  HIT_TYPE_SKY               = 0xffffffffu,
  HIT_TYPE_OCEAN             = 0xfffffffeu,
  HIT_TYPE_PARTICLE          = 0xfffffffdu,
  HIT_TYPE_VOLUME_OCEAN      = 0xfffffff3u,
  HIT_TYPE_VOLUME_FOG        = 0xfffffff2u,
  HIT_TYPE_REJECT            = 0xfffffff0u,
  HIT_TYPE_LIGHT_BSDF_HINT   = 0xffffffefu,
  HIT_TYPE_PARTICLE_MAX      = 0xefffffffu,
  HIT_TYPE_PARTICLE_MIN      = 0x80000000u,
  HIT_TYPE_PARTICLE_MASK     = 0x7fffffffu,
  HIT_TYPE_TRIANGLE_ID_LIMIT = 0x7fffffffu
} typedef HitType;

enum ShadingTaskIndex {
  SHADING_TASK_INDEX_GEOMETRY,
  SHADING_TASK_INDEX_VOLUME,
  SHADING_TASK_INDEX_PARTICLE,
  SHADING_TASK_INDEX_SKY,
  SHADING_TASK_INDEX_TOTAL
} typedef ShadingTaskIndex;

#define TASK_ADDRESS_OFFSET_IMPL(__internal_macro_shading_task_index) (NUM_THREADS * __internal_macro_shading_task_index + THREAD_ID)

#define TASK_ADDRESS_OFFSET_GEOMETRY TASK_ADDRESS_OFFSET_IMPL(SHADING_TASK_INDEX_GEOMETRY)
#define TASK_ADDRESS_OFFSET_VOLUME TASK_ADDRESS_OFFSET_IMPL(SHADING_TASK_INDEX_VOLUME)
#define TASK_ADDRESS_OFFSET_PARTICLE TASK_ADDRESS_OFFSET_IMPL(SHADING_TASK_INDEX_PARTICLE)
#define TASK_ADDRESS_OFFSET_SKY TASK_ADDRESS_OFFSET_IMPL(SHADING_TASK_INDEX_SKY)
#define TASK_ADDRESS_OFFSET_TOTAL TASK_ADDRESS_OFFSET_IMPL(SHADING_TASK_INDEX_TOTAL)

#define VOLUME_HIT_CHECK(X) ((X == HIT_TYPE_VOLUME_FOG) || (X == HIT_TYPE_VOLUME_OCEAN))
#define VOLUME_HIT_TYPE(X) ((X <= HIT_TYPE_PARTICLE_MAX) ? VOLUME_TYPE_PARTICLE : ((VolumeType) (X & 0x00000001u)))
#define PARTICLE_HIT_CHECK(X) ((X <= HIT_TYPE_PARTICLE_MAX) && (X >= HIT_TYPE_PARTICLE_MIN))
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
// STATE_FLAG_OCEAN_SCATTERED: This flag is set for paths that have at least one vertex that is a ocean volume scattering event.
//                             This flag is used to limit ocean volumes to single scattering for performance reasons.
//
enum StateFlag {
  STATE_FLAG_DELTA_PATH       = 0b00000001u,
  STATE_FLAG_CAMERA_DIRECTION = 0b00000010u,
  STATE_FLAG_OCEAN_SCATTERED  = 0b00000100u
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
// Functions
//===========================================================================================

#define UTILS_NO_PIXEL_SELECTED (make_ushort2(0xFFFF, 0xFFFF))

__device__ bool is_selected_pixel(const ushort2 index) {
  if (device.state.user_selected_x == UTILS_NO_PIXEL_SELECTED.x && device.state.user_selected_y == UTILS_NO_PIXEL_SELECTED.y)
    return false;

  // Only the top left subpixel of a pixel can be selected.
  return (index.x == device.state.user_selected_x && index.y == device.state.user_selected_y);
}

__device__ bool is_selected_pixel_lenient(const ushort2 index) {
  if (device.state.user_selected_x == UTILS_NO_PIXEL_SELECTED.x && device.state.user_selected_y == UTILS_NO_PIXEL_SELECTED.y)
    return true;

  return is_selected_pixel(index);
}

__device__ uint32_t get_pixel_id(const ushort2 pixel) {
  return pixel.x + device.settings.width * pixel.y;
}

__device__ uint32_t get_task_address_of_thread(const uint32_t thread_id, const uint32_t block_id, const uint32_t number) {
  static_assert(THREADS_PER_BLOCK == 128, "I wrote this using that we have 4 warps per block, this is also used in the 0x3!");

  const uint32_t threads_per_warp  = 32;
  const uint32_t warp_id           = ((thread_id >> 5) & 0x3) + block_id * 4;
  const uint32_t thread_id_in_warp = (thread_id & 0x1f);
  return threads_per_warp * device.pixels_per_thread * warp_id + threads_per_warp * number + thread_id_in_warp;
}

__device__ uint32_t get_task_address(const uint32_t number) {
#ifndef OPTIX_KERNEL
  return get_task_address_of_thread(threadIdx.x, blockIdx.x, number);
#else
  const uint3 idx = optixGetLaunchIndex();
  return get_task_address_of_thread(idx.x, idx.y, number);
#endif
}

__device__ bool is_non_finite(const float a) {
#ifndef __INTELLISENSE__
  return isnan(a) || isinf(a);
#else
  const bool is_nan = a != a;
  const bool is_inf = a < -FLT_MAX || a > FLT_MAX;

  return is_nan || is_inf;
#endif
}

#define LIGHTS_ARE_PRESENT (device.ptrs.light_tree_nodes != nullptr)

#define HANDLE_DEVICE_ABORT()                                                                                         \
  if (__ldcv(device.ptrs.abort_flag) != 0 && ((device.state.undersampling & UNDERSAMPLING_FIRST_SAMPLE_MASK) == 0)) { \
    return;                                                                                                           \
  }

//===========================================================================================
// Debug utils
//===========================================================================================

// #define UTILS_DEBUG_MODE

#ifdef UTILS_DEBUG_MODE

#define UTILS_DEBUG_NAN_COLOR (get_color(1.0f, 0.0f, 0.0f))

__device__ bool _utils_debug_nans(const RGBF color, const char* func, const uint32_t line, const char* var) {
  const float sum_color = color.r + color.g + color.b;
  if (isnan(sum_color) || isinf(sum_color)) {
    printf("[%s:%u] Failed NaN check. %s = (%f %f %f).\n", func, line, var, color.r, color.g, color.b);
    return true;
  }

  return false;
}

__device__ bool _utils_debug_nans(const float value, const char* func, const uint32_t line, const char* var) {
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

__device__ TriangleHandle triangle_handle_get(const uint32_t instance_id, const uint32_t tri_id) {
  TriangleHandle handle;

  handle.instance_id = instance_id;
  handle.tri_id      = tri_id;

  return handle;
}

__device__ bool triangle_handle_equal(const TriangleHandle handle1, const TriangleHandle handle2) {
  return (handle1.instance_id == handle2.instance_id) && (handle1.tri_id == handle2.tri_id);
}

#endif /* CU_UTILS_H */
