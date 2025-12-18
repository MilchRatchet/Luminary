
#ifndef CU_LUMINARY_MEDIUM_STACK_H
#define CU_LUMINARY_MEDIUM_STACK_H

#include "math.cuh"
#include "utils.cuh"

// This whole code assumes that we actually have stack behaviour, i.e., we enter materials in the same order that we leave them. This is
// fine for now but it might be desirable to extend this in the future to support arbitrary enter/leave ordering.

LUMINARY_FUNCTION float medium_stack_ior_peek(const DeviceTaskMediumStack& medium, const bool peek_previous) {
  uint32_t compressed_ior = (peek_previous) ? medium.ior >> 8 : medium.ior;

  compressed_ior = compressed_ior & 0xFF;

  return ior_decompress(compressed_ior);
}

LUMINARY_FUNCTION void medium_stack_ior_modify(DeviceTaskMediumStack& medium, const float ior, const bool push) {
  if (push) {
    medium.ior = medium.ior << 8;
    medium.ior |= ior_compress(ior);
  }
  else {
    medium.ior = medium.ior >> 8;
  }
}

LUMINARY_FUNCTION uint16_t medium_stack_volume_peek(const DeviceTaskMediumStack& medium, const bool peek_previous) {
  return (peek_previous) ? medium.volume_id_01 >> 16 : medium.volume_id_01 & 0xFFFF;
}

LUMINARY_FUNCTION void medium_stack_volume_modify(DeviceTaskMediumStack& medium, const uint16_t volume_id, const bool push) {
  if (push) {
    medium.volume_id_23 = medium.volume_id_23 << 16;
    medium.volume_id_23 |= medium.volume_id_01 >> 16;
    medium.volume_id_01 = medium.volume_id_01 << 16;
    medium.volume_id_01 |= volume_id;
  }
  else {
    medium.volume_id_01 = medium.volume_id_01 >> 16;
    medium.volume_id_01 |= medium.volume_id_23 << 16;
    medium.volume_id_23 = medium.volume_id_23 >> 16;
  }
}

#endif /* CU_LUMINARY_MEDIUM_STACK_H */
