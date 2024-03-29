#ifndef CU_UTILS_H
#define CU_UTILS_H

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include "log.h"
#include "utils.h"

#define THREADS_PER_BLOCK 128
#define BLOCKS_PER_GRID 1024
#define NUM_THREADS (THREADS_PER_BLOCK * BLOCKS_PER_GRID)
#define THREAD_ID (threadIdx.x + blockIdx.x * blockDim.x)

#define LUMINARY_KERNEL __global__ __launch_bounds__(THREADS_PER_BLOCK)

#ifndef eps
#define eps 0.000001f
#endif /* eps */

enum HitType : uint32_t {
  HIT_TYPE_SKY               = 0xffffffffu,
  HIT_TYPE_OCEAN             = 0xfffffffeu,
  HIT_TYPE_TOY               = 0xfffffffdu,
  HIT_TYPE_PARTICLE          = 0xfffffffcu,
  HIT_TYPE_VOLUME_OCEAN      = 0xfffffff3u,
  HIT_TYPE_VOLUME_FOG        = 0xfffffff2u,
  HIT_TYPE_REJECT            = 0xfffffff0u,
  HIT_TYPE_PARTICLE_MAX      = 0xefffffffu,
  HIT_TYPE_PARTICLE_MIN      = 0x80000000u,
  HIT_TYPE_PARTICLE_MASK     = 0x7fffffffu,
  HIT_TYPE_TRIANGLE_ID_LIMIT = 0x7fffffffu
} typedef HitType;

enum TaskAddressOffset {
  TASK_ADDRESS_OFFSET_GEOMETRY   = 0,
  TASK_ADDRESS_OFFSET_PARTICLE   = 1,
  TASK_ADDRESS_OFFSET_OCEAN      = 2,
  TASK_ADDRESS_OFFSET_SKY        = 3,
  TASK_ADDRESS_OFFSET_TOY        = 4,
  TASK_ADDRESS_OFFSET_VOLUME     = 5,
  TASK_ADDRESS_OFFSET_TOTALCOUNT = 6,
  TASK_ADDRESS_OFFSET_STRIDE     = 6,
  TASK_ADDRESS_COUNT_STRIDE      = 7
} typedef TaskAddressOffset;

#define VOLUME_HIT_CHECK(X) ((X == HIT_TYPE_VOLUME_FOG) || (X == HIT_TYPE_VOLUME_OCEAN))
#define VOLUME_HIT_TYPE(X) ((X <= HIT_TYPE_PARTICLE_MAX) ? VOLUME_TYPE_PARTICLE : ((VolumeType) (X & 0x00000001u)))
#define PARTICLE_HIT_CHECK(X) ((X <= HIT_TYPE_PARTICLE_MAX) && (X >= HIT_TYPE_PARTICLE_MIN))

//===========================================================================================
// Device Variables
//===========================================================================================

#ifndef UTILS_NO_DEVICE_TABLE
__constant__ DeviceConstantMemory device;
#endif

//===========================================================================================
// Functions
//===========================================================================================

#ifndef UTILS_NO_DEVICE_FUNCTIONS
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
  return get_task_address_of_thread(threadIdx.x, blockIdx.x, number);
}

__device__ static int get_task_address2(const unsigned int x, const unsigned int y, const int number) {
  return get_task_address_of_thread(x, y, number);
}

__device__ static int is_first_ray() {
  return (device.iteration_type == TYPE_CAMERA);
}

__device__ static bool proper_light_sample(const uint32_t target_light, const uint32_t source_light) {
  return (
    (device.iteration_type == TYPE_CAMERA) || (device.iteration_type == TYPE_BOUNCE)
    || ((device.iteration_type == TYPE_LIGHT) && (target_light == source_light)) || target_light == LIGHT_ID_ANY);
}
#endif

#endif /* CU_UTILS_H */
