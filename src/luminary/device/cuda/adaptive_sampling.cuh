#ifndef CU_LUMINARY_ADAPTIVE_SAMPLING_H
#define CU_LUMINARY_ADAPTIVE_SAMPLING_H

#include "utils.cuh"

LUMINARY_FUNCTION uint32_t adapative_sampling_get_sample_offset(const uint32_t x, const uint32_t y) {
  const uint32_t adaptive_sampling_width = device.settings.width >> ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG;
  const uint32_t adaptive_sampling_x     = x >> ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG;
  const uint32_t adaptive_sampling_y     = y >> ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG;

  const uint32_t adaptive_sampling_block = adaptive_sampling_x + adaptive_sampling_y * adaptive_sampling_width;

  const uint32_t adaptive_sampling_counts = device.ptrs.stage_sample_counts[adaptive_sampling_block];

  uint32_t sample_id = device.state.sample_allocation.stage_sample_offsets[0];

  for (uint32_t stage_id = 0; stage_id < ADAPTIVE_SAMPLER_NUM_STAGES; stage_id++) {
    const uint32_t stage_sample_offset = device.state.sample_allocation.stage_sample_offsets[stage_id + 1];
    const uint32_t stage_sample_count  = ((adaptive_sampling_counts >> (stage_id * 8)) & 0xFF) + 1;

    sample_id += stage_sample_offset * stage_sample_count;
  }

  return sample_id;
}

LUMINARY_FUNCTION uint32_t adaptive_sampling_get_sample_count(const uint32_t x, const uint32_t y) {
  // TODO: This has to be different because for multi-device the sample count is not computed from the sample allocation.

  const uint32_t adaptive_sampling_width = device.settings.width >> ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG;
  const uint32_t adaptive_sampling_x     = x >> ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG;
  const uint32_t adaptive_sampling_y     = y >> ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG;

  const uint32_t adaptive_sampling_block = adaptive_sampling_x + adaptive_sampling_y * adaptive_sampling_width;

  const uint32_t adaptive_sampling_counts = device.ptrs.stage_sample_counts[adaptive_sampling_block];

  uint32_t sample_id = device.state.sample_allocation.stage_sample_offsets[0];

  for (uint32_t stage_id = 0; stage_id < ADAPTIVE_SAMPLER_NUM_STAGES; stage_id++) {
    const uint32_t stage_sample_offset = device.state.sample_allocation.stage_sample_offsets[stage_id + 1];
    const uint32_t stage_sample_count  = ((adaptive_sampling_counts >> (stage_id * 8)) & 0xFF) + 1;

    sample_id += stage_sample_offset * stage_sample_count;
  }

  return sample_id;
}

#endif /* CU_LUMINARY_ADAPTIVE_SAMPLING_H */
