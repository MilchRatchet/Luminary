#ifndef CU_RIS_H
#define CU_RIS_H

#include "math.cuh"
#include "random.cuh"
#include "utils.cuh"

__device__ uint32_t ris_sample_light(const GBufferData data, const ushort2 pixel) {
  // TODO: Implement RIS
  return (quasirandom_sequence_1D(QUASI_RANDOM_TARGET_TBD_0, pixel) * device.scene.triangle_lights_count) + 0.499f;
}

#endif /* CU_RIS_H */
