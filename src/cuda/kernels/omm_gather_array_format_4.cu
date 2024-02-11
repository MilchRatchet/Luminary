#include "micromap_utils.cuh"
#include "utils.cuh"

__global__ void omm_gather_array_format_4(
  uint8_t* dst, const uint8_t* src, const uint32_t level, const uint8_t* level_record, const OptixOpacityMicromapDesc* desc) {
  int id                        = THREAD_ID;
  const uint32_t triangle_count = device.scene.triangle_data.triangle_count;
  const uint32_t state_size     = OMM_STATE_SIZE(level, OPTIX_OPACITY_MICROMAP_FORMAT_4_STATE);

  while (id < triangle_count) {
    if (level_record[id] != level) {
      id += blockDim.x * gridDim.x;
      continue;
    }

    for (uint32_t j = 0; j < state_size; j++) {
      dst[desc[id].byteOffset + j] = src[id * state_size + j];
    }

    id += blockDim.x * gridDim.x;
  }
}
