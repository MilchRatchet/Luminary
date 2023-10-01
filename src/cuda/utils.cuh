#ifndef CU_UTILS_H
#define CU_UTILS_H

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include "log.h"
#include "utils.h"

#define THREADS_PER_BLOCK 128
#define BLOCKS_PER_GRID 1024

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
  HIT_TYPE_TRIANGLE_ID_LIMIT = 0xefffffffu
} typedef HitType;

// VOLUME_FOG_HIT is supposed to always have the volume type bits set to 0.
#define VOLUME_HIT_CHECK(X) ((X & HIT_TYPE_VOLUME_FOG) == HIT_TYPE_VOLUME_FOG)
#define VOLUME_HIT_TYPE(X) ((VolumeType) (X & 0x00000001u))

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
  const int warp_id       = (((thread_id & 0x60) >> 5) + block_id * (THREADS_PER_BLOCK / 32));
  const int thread_offset = (thread_id & 0x1f);
  return 32 * device.pixels_per_thread * warp_id + 32 * number + thread_offset;
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
