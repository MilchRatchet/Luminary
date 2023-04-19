#ifndef CU_UTILS_H
#define CU_UTILS_H

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>

#include "log.h"
#include "utils.h"

#define THREADS_PER_BLOCK 128
#define BLOCKS_PER_GRID 1024

#ifndef eps
#define eps 0.001f
#endif /* eps */

//===========================================================================================
// Bit Masks
//===========================================================================================

#define RANDOM_INDEX 0x0000ffffu
#define DEPTH_LEFT 0xffff0000u
#define SKY_HIT 0xffffffffu
#define OCEAN_HIT 0xfffffffeu
#define TOY_HIT 0xfffffffdu
#define FOG_HIT 0xfffffffcu
#define REJECT_HIT 0xfffffff0u
#define TRIANGLE_ID_LIMIT 0xefffffffu
#define LIGHT_ID_ANY 0xfffffff0u
#define LIGHT_ID_ANY_NO_SUN 0xfffffff1u
#define STATE_ALBEDO 0b1u
#define STATE_LIGHT_OCCUPIED 0b10u
#define STATE_BOUNCE_OCCUPIED 0b100u

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
    device.iteration_type == TYPE_CAMERA || ((device.iteration_type == TYPE_LIGHT) && (target_light == source_light))
    || target_light == LIGHT_ID_ANY || (source_light != LIGHT_ID_SUN && target_light == LIGHT_ID_ANY_NO_SUN));
}
#endif

#endif /* CU_UTILS_H */
