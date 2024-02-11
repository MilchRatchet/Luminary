#include "micromap_utils.cuh"
#include "utils.cuh"

//
// This kernel computes a level 0 format 4 base micromap array.
//
__global__ void omm_level_0_format_4(uint8_t* dst, uint8_t* level_record) {
  int id                        = THREAD_ID;
  const uint32_t triangle_count = device.scene.triangle_data.triangle_count;

  while (id < triangle_count) {
    OMMTextureTriangle tri = micromap_get_ommtexturetriangle(id);

    const int opacity = micromap_get_opacity(tri, 0, 0);

    const uint8_t v = opacity;

    if (opacity != OPTIX_OPACITY_MICROMAP_STATE_UNKNOWN_OPAQUE)
      level_record[id] = 0;

    dst[id] = v;

    id += blockDim.x * gridDim.x;
  }
}
