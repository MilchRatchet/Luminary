#include "memory.cuh"
#include "utils.cuh"

__global__ void dmm_precompute_indices(uint32_t* dst) {
  int id                        = THREAD_ID;
  const uint32_t triangle_count = device.scene.triangle_data.triangle_count;

  while (id < triangle_count) {
    const uint32_t material_id = load_triangle_material_id(id);
    const uint16_t tex         = __ldg(&(device.scene.materials[material_id].normal_map));

    dst[id] = (tex == TEXTURE_NONE) ? 0 : 1;

    id += blockDim.x * gridDim.x;
  }
}
