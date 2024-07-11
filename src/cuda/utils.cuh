#ifndef CU_UTILS_H
#define CU_UTILS_H

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include "log.h"
#include "utils.h"

#define NUM_THREADS (THREADS_PER_BLOCK * BLOCKS_PER_GRID)

#ifndef OPTIX_KERNEL
#define THREAD_ID (threadIdx.x + blockIdx.x * blockDim.x)
#else
#define THREAD_ID (optixGetLaunchIndex().x + optixGetLaunchIndex().y * optixGetLaunchDimensions().x)
#endif

#define LUMINARY_KERNEL __global__ __launch_bounds__(THREADS_PER_BLOCK)

#ifndef eps
#define eps 0.000001f
#endif /* eps */

#define GEOMETRY_DELTA_PATH_CUTOFF (0.025f)

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

enum TaskAddressOffset {
  TASK_ADDRESS_OFFSET_GEOMETRY   = 0,
  TASK_ADDRESS_OFFSET_VOLUME     = 1,
  TASK_ADDRESS_OFFSET_OCEAN      = 2,
  TASK_ADDRESS_OFFSET_SKY        = 3,
  TASK_ADDRESS_OFFSET_TOTALCOUNT = 4,
  TASK_ADDRESS_OFFSET_STRIDE     = 4,
  TASK_ADDRESS_COUNT_STRIDE      = 5
} typedef TaskAddressOffset;

#define VOLUME_HIT_CHECK(X) ((X == HIT_TYPE_VOLUME_FOG) || (X == HIT_TYPE_VOLUME_OCEAN))
#define VOLUME_HIT_TYPE(X) ((X <= HIT_TYPE_PARTICLE_MAX) ? VOLUME_TYPE_PARTICLE : ((VolumeType) (X & 0x00000001u)))
#define PARTICLE_HIT_CHECK(X) ((X <= HIT_TYPE_PARTICLE_MAX) && (X >= HIT_TYPE_PARTICLE_MIN))
#define IS_PRIMARY_RAY (device.depth == 0)

//===========================================================================================
// Device Variables
//===========================================================================================

#ifndef UTILS_NO_DEVICE_TABLE
__constant__ DeviceConstantMemory device;
#endif

//===========================================================================================
// Functions
//===========================================================================================

__device__ static bool is_selected_pixel(const ushort2 index) {
  return (index.x == device.user_selected_x && index.y == device.user_selected_y);
}

__device__ static uint32_t get_pixel_id(const int x, const int y) {
  return x + device.width * y;
}

__device__ static int get_task_address_of_thread(const int thread_id, const int block_id, const int number) {
  static_assert(THREADS_PER_BLOCK == 128, "I wrote this using that we have 4 warps per block, this is also used in the 0x3!");

  const uint32_t threads_per_warp  = 32;
  const uint32_t warp_id           = ((thread_id >> 5) & 0x3) + block_id * 4;
  const uint32_t thread_id_in_warp = (thread_id & 0x1f);
  return threads_per_warp * device.pixels_per_thread * warp_id + threads_per_warp * number + thread_id_in_warp;
}

__device__ static int get_task_address(const int number) {
#ifndef OPTIX_KERNEL
  return get_task_address_of_thread(threadIdx.x, blockIdx.x, number);
#else
  const uint3 idx = optixGetLaunchIndex();
  return get_task_address_of_thread(idx.x, idx.y, number);
#endif
}

#endif /* CU_UTILS_H */
