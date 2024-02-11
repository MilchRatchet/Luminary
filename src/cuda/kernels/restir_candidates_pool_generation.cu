#include "random.cuh"
#include "utils.cuh"

__global__ void restir_candidates_pool_generation() {
  int id        = THREAD_ID;
  uint32_t seed = device.ptrs.randoms[THREAD_ID];

  const int light_sample_bin_count = 1 << device.restir.light_candidate_pool_size_log2;

  if (device.scene.triangle_lights_count == 0)
    return;

  while (id < light_sample_bin_count) {
    // TODO: Expose more white noise keys, maybe
    const uint32_t sampled_id =
      random_uint32_t_base(0xfcbd6e15, id + light_sample_bin_count * device.temporal_frames) % device.scene.triangle_lights_count;

    device.ptrs.light_candidates[id]             = sampled_id;
    device.restir.presampled_triangle_lights[id] = device.scene.triangle_lights[sampled_id];

    id += blockDim.x * gridDim.x;
  }

  device.ptrs.randoms[THREAD_ID] = seed;
}
