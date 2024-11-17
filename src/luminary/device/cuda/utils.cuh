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

#ifndef eps
#define eps 0.000001f
#endif /* eps */

#define GEOMETRY_DELTA_PATH_CUTOFF (0.05f)
#define BSDF_ROUGHNESS_CLAMP (0.025f)

enum HitType : uint32_t {
  HIT_TYPE_SKY               = 0xffffffffu,
  HIT_TYPE_OCEAN             = 0xfffffffeu,
  HIT_TYPE_TOY               = 0xfffffffdu,
  HIT_TYPE_PARTICLE          = 0xfffffffcu,
  HIT_TYPE_VOLUME_OCEAN      = 0xfffffff3u,
  HIT_TYPE_VOLUME_FOG        = 0xfffffff2u,
  HIT_TYPE_REJECT            = 0xfffffff0u,
  HIT_TYPE_LIGHT_BSDF_HINT   = 0xffffffefu,
  HIT_TYPE_PARTICLE_MAX      = 0xefffffffu,
  HIT_TYPE_PARTICLE_MIN      = 0x80000000u,
  HIT_TYPE_PARTICLE_MASK     = 0x7fffffffu,
  HIT_TYPE_TRIANGLE_ID_LIMIT = 0x7fffffffu
} typedef HitType;

// TODO: Task addresses should be interleaved. This will reduce time to load task list meta data.
enum TaskAddressOffset {
  TASK_ADDRESS_OFFSET_GEOMETRY   = 0,
  TASK_ADDRESS_OFFSET_VOLUME     = 1,
  TASK_ADDRESS_OFFSET_PARTICLE   = 2,
  TASK_ADDRESS_OFFSET_SKY        = 3,
  TASK_ADDRESS_OFFSET_TOTALCOUNT = 4,
  TASK_ADDRESS_OFFSET_STRIDE     = 4,
  TASK_ADDRESS_COUNT_STRIDE      = 5
} typedef TaskAddressOffset;

struct CompressedAlpha {
  // Unsigned int for OptiX compatibility
  unsigned int data0;
  unsigned int data1;
} typedef CompressedAlpha;

#define VOLUME_HIT_CHECK(X) ((X == HIT_TYPE_VOLUME_FOG) || (X == HIT_TYPE_VOLUME_OCEAN))
#define VOLUME_HIT_TYPE(X) ((X <= HIT_TYPE_PARTICLE_MAX) ? VOLUME_TYPE_PARTICLE : ((VolumeType) (X & 0x00000001u)))
#define PARTICLE_HIT_CHECK(X) ((X <= HIT_TYPE_PARTICLE_MAX) && (X >= HIT_TYPE_PARTICLE_MIN))
#define IS_PRIMARY_RAY (device.depth == 0)

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

#define OUTPUT_DIM(dim) (dim >> 1)

__device__ bool is_selected_pixel(const ushort2 index) {
  // Only the top left subpixel of a pixel can be selected.
  return (index.x == (device.user_selected_x << 1) && index.y == (device.user_selected_y << 1));
}

__device__ bool is_selected_pixel_lenient(const ushort2 index) {
  if (device.user_selected_x == UTILS_NO_PIXEL_SELECTED.x && device.user_selected_y == UTILS_NO_PIXEL_SELECTED.y)
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

__device__ bool is_finite(const float a) {
  const bool is_nan = a != a;
  const bool is_inf = a < -FLT_MAX || a > FLT_MAX;

  return is_nan || is_inf;
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
