#ifndef CU_RIS_H
#define CU_RIS_H

#include "math.cuh"
#include "random.cuh"
#include "utils.cuh"

__device__ uint32_t ris_sample_light(const GBufferData data, const ushort2 pixel, const uint32_t light_ray_index) {
  // TODO: Implement RIS
  return (quasirandom_sequence_1D(QUASI_RANDOM_TARGET_TBD_0 + light_ray_index, pixel) * device.scene.triangle_lights_count) + 0.499f;
}

#endif /* CU_RIS_H */
