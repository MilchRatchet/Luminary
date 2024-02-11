#include "micromap_utils.cuh"
#include "utils.cuh"

__global__ void omm_refine_format_4(uint8_t* dst, const uint8_t* src, uint8_t* level_record, const uint32_t src_level) {
  int id                        = THREAD_ID;
  const uint32_t triangle_count = device.scene.triangle_data.triangle_count;

  while (id < triangle_count) {
    if (level_record[id] != 0xFF) {
      id += blockDim.x * gridDim.x;
      continue;
    }

    OMMTextureTriangle tri = micromap_get_ommtexturetriangle(id);

    const uint32_t src_tri_count  = 1 << (2 * src_level);
    const uint32_t src_state_size = (src_tri_count + 3) / 4;
    const uint32_t dst_state_size = src_tri_count;

    const uint8_t* src_tri_ptr = src + id * src_state_size;
    uint8_t* dst_tri_ptr       = dst + id * dst_state_size;

    bool unknowns_left = false;

    for (uint32_t i = 0; i < src_tri_count; i++) {
      uint8_t src_v = src_tri_ptr[i / 4];
      src_v         = (src_v >> (2 * (i & 0b11))) & 0b11;

      uint8_t dst_v = 0;
      if (src_v == OPTIX_OPACITY_MICROMAP_STATE_UNKNOWN_OPAQUE) {
        for (uint32_t j = 0; j < 4; j++) {
          const uint8_t opacity = micromap_get_opacity(tri, src_level + 1, 4 * i + j);

          if (opacity == OPTIX_OPACITY_MICROMAP_STATE_UNKNOWN_OPAQUE)
            unknowns_left = true;

          dst_v = dst_v | (opacity << (2 * j));
        }
      }
      else {
        dst_v = src_v | (src_v << 2) | (src_v << 4) | (src_v << 6);
      }

      dst_tri_ptr[i] = dst_v;
    }

    if (!unknowns_left)
      level_record[id] = src_level + 1;

    id += blockDim.x * gridDim.x;
  }
}
