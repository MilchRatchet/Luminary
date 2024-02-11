#include "math.cuh"
#include "micromap_utils.cuh"
#include "utils.cuh"

LUM_DEVICE_FUNC const uint8_t dmm_umajor_id_level_3_format_64[45] = {0,  15, 6, 21, 3, 39, 12, 42, 2,  17, 16, 23, 22, 41, 40,
                                                                     44, 43, 8, 18, 7, 24, 14, 36, 13, 20, 19, 26, 25, 38, 37,
                                                                     5,  27, 9, 33, 4, 29, 28, 35, 34, 11, 30, 10, 32, 31, 1};

//
// This function is taken from the optixDisplacementMicromesh example code provided in the OptiX 7.7 installation.
//
LUM_DEVICE_FUNC void dmm_set_displacement(uint8_t* dst, const uint32_t u_vert_id, const float disp) {
  const uint32_t vert_id = dmm_umajor_id_level_3_format_64[u_vert_id];

  const uint16_t v = (uint16_t) (disp * ((uint16_t) 0x7FF));

  uint32_t bit_offset = 11 * vert_id;
  uint32_t v_offset   = 0;

  while (v_offset < 11) {
    uint32_t num = (~bit_offset & 7) + 1;

    if (11 - v_offset < num) {
      num = 11 - v_offset;
    }

    const uint16_t mask  = (1u << num) - 1u;
    const uint32_t id    = bit_offset >> 3;
    const uint32_t shift = bit_offset & 7;

    const uint8_t bits = (uint8_t) ((v >> v_offset) & mask);
    dst[id] &= ~(mask << shift);
    dst[id] |= bits << shift;

    bit_offset += num;
    v_offset += num;
  }
}

__global__ void dmm_build_level_3_format_64(uint8_t* dst, const uint32_t* mapping, const uint32_t count) {
  int id = THREAD_ID;

  while (id < count) {
    const uint32_t tri_id = mapping[id];

    if (tri_id == DMM_NONE) {
      uint4* ptr    = (uint4*) (dst + 64 * id);
      const uint4 v = make_uint4(0, 0, 0, 0);

      __stcs(ptr + 0, v);
      __stcs(ptr + 1, v);
      __stcs(ptr + 2, v);
      __stcs(ptr + 3, v);
    }
    else {
      const DMMTextureTriangle tri = micromap_get_dmmtexturetriangle(tri_id);

      float2 bary0;
      float2 bary1;
      float2 bary2;
      optixMicromapIndexToBaseBarycentrics(0, 0, bary0, bary1, bary2);

      const UV uv0 = lerp_uv(tri.vertex, tri.edge1, tri.edge2, bary0);
      const UV uv1 = lerp_uv(tri.vertex, tri.edge1, tri.edge2, bary1);
      const UV uv2 = lerp_uv(tri.vertex, tri.edge1, tri.edge2, bary2);

      uint32_t u_vert_id          = 0;
      const uint32_t num_segments = 1 << 3;

      for (uint32_t i = 0; i < num_segments + 1; i++) {
        for (uint32_t j = 0; j < num_segments + 1 - i; j++) {
          const float2 micro_vertex_bary = make_float2(((float) i) / num_segments, ((float) j) / num_segments);

          UV uv;
          uv.u = (1.0f - micro_vertex_bary.x - micro_vertex_bary.y) * uv0.u + micro_vertex_bary.x * uv1.u + micro_vertex_bary.y * uv2.u;
          uv.v = (1.0f - micro_vertex_bary.x - micro_vertex_bary.y) * uv0.v + micro_vertex_bary.x * uv1.v + micro_vertex_bary.y * uv2.v;

          const float disp = tex2D<float4>(tri.tex.tex, uv.u, 1.0f - uv.v).w;

          dmm_set_displacement(dst + 64 * id, u_vert_id, disp);
          u_vert_id++;
        }
      }
    }

    id += blockDim.x * gridDim.x;
  }
}
