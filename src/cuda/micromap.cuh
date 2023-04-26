#ifndef CU_MICROMAP_H
#define CU_MICROMAP_H

#include <optix.h>
#include <optix_stubs.h>

#include "buffer.h"
#include "device.h"
#include "utils.cuh"

#define OPACITY_MICROMAP_STATE_SIZE(__level__, __format__) \
  (((1 << (__level__ * 2)) * ((__format__ == OPTIX_OPACITY_MICROMAP_FORMAT_2_STATE) ? 1 : 2) + 7) / 8)

// Currently this kernel simply sets everything to opague_unknown or opague based on the format
// Assumption is that the state of each triangle is a multiple of 8 bits in size
// that means no FORMAT_2 with level 0 or 1 and no FORMAT_4 with level 0
__global__ void micromap_opacity_compute(const uint32_t level, const OptixOpacityMicromapFormat format, uint8_t* dst) {
  int id                        = threadIdx.x + blockIdx.x * blockDim.x;
  const uint32_t triangle_count = device.scene.triangle_data.triangle_count;

  const uint32_t state_size = OPACITY_MICROMAP_STATE_SIZE(level, format);

  while (id < triangle_count) {
    uint8_t* ptr = dst + id * state_size;

    if (format == OPTIX_OPACITY_MICROMAP_FORMAT_2_STATE) {
      for (int i = 0; i < state_size; i++) {
        ptr[i] = 0xFF;
      }
    }
    else {
      for (int i = 0; i < state_size; i++) {
        ptr[i] = 0xFF;
      }
    }

    id += blockDim.x * gridDim.x;
  }
}

void micromap_opacity_build(RaytraceInstance* instance, void** ptr) {
  const uint32_t level                    = 4;
  const OptixOpacityMicromapFormat format = OPTIX_OPACITY_MICROMAP_FORMAT_4_STATE;

  const uint32_t state_size = OPACITY_MICROMAP_STATE_SIZE(level, format);

  device_malloc(ptr, state_size * instance->scene.triangle_data.triangle_count);

  micromap_opacity_compute<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(level, format, (uint8_t*) *ptr);

  gpuErrchk(cudaDeviceSynchronize());
}

#endif /* CU_MICROMAP_H */
