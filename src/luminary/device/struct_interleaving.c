#include "struct_interleaving.h"

#include "internal_error.h"

LuminaryResult struct_triangles_interleave(DeviceTriangle* dst, const DeviceTriangle* src, uint32_t count) {
  __CHECK_NULL_ARGUMENT(dst);
  __CHECK_NULL_ARGUMENT(src);

  for (uint32_t i = 0; i < count; i++) {
    const DeviceTriangle* triangle = src + i;

    float* dst_float   = (float*) dst;
    uint32_t* dst_uint = (uint32_t*) dst;

    uint32_t offset = 0;

    dst_float[offset + i * 4 + 0] = triangle->vertex.x;
    dst_float[offset + i * 4 + 1] = triangle->vertex.y;
    dst_float[offset + i * 4 + 2] = triangle->vertex.z;
    dst_float[offset + i * 4 + 3] = triangle->edge1.x;

    offset += count * 4;

    dst_float[offset + i * 4 + 0] = triangle->edge1.y;
    dst_float[offset + i * 4 + 1] = triangle->edge1.z;
    dst_float[offset + i * 4 + 2] = triangle->edge2.x;
    dst_float[offset + i * 4 + 3] = triangle->edge2.y;

    offset += count * 4;

    dst_float[offset + i * 4 + 0] = triangle->edge2.z;
    dst_uint[offset + i * 4 + 1]  = triangle->vertex_texture;
    dst_uint[offset + i * 4 + 2]  = triangle->vertex1_texture;
    dst_uint[offset + i * 4 + 3]  = triangle->vertex2_texture;

    offset += count * 4;

    dst_uint[offset + i * 4 + 0] = triangle->vertex_normal;
    dst_uint[offset + i * 4 + 1] = triangle->vertex1_normal;
    dst_uint[offset + i * 4 + 2] = triangle->vertex2_normal;
    dst_uint[offset + i * 4 + 3] = ((uint32_t) triangle->material_id) | (0 << 16);
  }

  return LUMINARY_SUCCESS;
}

LuminaryResult struct_triangles_deinterleave(DeviceTriangle* dst, const DeviceTriangle* src, uint32_t count) {
  __CHECK_NULL_ARGUMENT(dst);
  __CHECK_NULL_ARGUMENT(src);

  for (uint32_t i = 0; i < count; i++) {
    float* src_float   = (float*) src;
    uint32_t* src_uint = (uint32_t*) src;

    DeviceTriangle* triangle = dst + i;

    uint32_t offset = 0;

    triangle->vertex.x = src_float[offset + i * 4 + 0];
    triangle->vertex.y = src_float[offset + i * 4 + 1];
    triangle->vertex.z = src_float[offset + i * 4 + 2];
    triangle->edge1.x  = src_float[offset + i * 4 + 3];

    offset += count * 4;

    triangle->edge1.y = src_float[offset + i * 4 + 0];
    triangle->edge1.z = src_float[offset + i * 4 + 1];
    triangle->edge2.x = src_float[offset + i * 4 + 2];
    triangle->edge2.y = src_float[offset + i * 4 + 3];

    offset += count * 4;

    triangle->edge2.z         = src_float[offset + i * 4 + 0];
    triangle->vertex_texture  = src_uint[offset + i * 4 + 1];
    triangle->vertex1_texture = src_uint[offset + i * 4 + 2];
    triangle->vertex2_texture = src_uint[offset + i * 4 + 3];

    offset += count * 4;

    triangle->vertex_normal  = src_uint[offset + i * 4 + 0];
    triangle->vertex1_normal = src_uint[offset + i * 4 + 1];
    triangle->vertex2_normal = src_uint[offset + i * 4 + 2];
    triangle->material_id    = src_uint[offset + i * 4 + 3] & 0xFFFF;
  }

  return LUMINARY_SUCCESS;
}
