#include "struct_interleaving.h"

void struct_triangles_interleave(Triangle* dst, Triangle* src, uint32_t count) {
  for (uint32_t i = 0; i < count; i++) {
    Triangle triangle = src[i];

    float* dst_float   = (float*) dst;
    uint32_t* dst_uint = (uint32_t*) dst;

    uint32_t offset = 0;

    dst_float[offset + i * 4 + 0] = triangle.vertex.x;
    dst_float[offset + i * 4 + 1] = triangle.vertex.y;
    dst_float[offset + i * 4 + 2] = triangle.vertex.z;
    dst_float[offset + i * 4 + 3] = triangle.edge1.x;

    offset += count * 4;

    dst_float[offset + i * 4 + 0] = triangle.edge1.y;
    dst_float[offset + i * 4 + 1] = triangle.edge1.z;
    dst_float[offset + i * 4 + 2] = triangle.edge2.x;
    dst_float[offset + i * 4 + 3] = triangle.edge2.y;

    offset += count * 4;

    dst_float[offset + i * 4 + 0] = triangle.edge2.z;
    dst_float[offset + i * 4 + 1] = triangle.vertex_normal.x;
    dst_float[offset + i * 4 + 2] = triangle.vertex_normal.y;
    dst_float[offset + i * 4 + 3] = triangle.vertex_normal.z;

    offset += count * 4;

    dst_float[offset + i * 4 + 0] = triangle.edge1_normal.x;
    dst_float[offset + i * 4 + 1] = triangle.edge1_normal.y;
    dst_float[offset + i * 4 + 2] = triangle.edge1_normal.z;
    dst_float[offset + i * 4 + 3] = triangle.edge2_normal.x;

    offset += count * 4;

    dst_float[offset + i * 4 + 0] = triangle.edge2_normal.y;
    dst_float[offset + i * 4 + 1] = triangle.edge2_normal.z;
    dst_float[offset + i * 4 + 2] = triangle.vertex_texture.u;
    dst_float[offset + i * 4 + 3] = triangle.vertex_texture.v;

    offset += count * 4;

    dst_float[offset + i * 4 + 0] = triangle.edge1_texture.u;
    dst_float[offset + i * 4 + 1] = triangle.edge1_texture.v;
    dst_float[offset + i * 4 + 2] = triangle.edge2_texture.u;
    dst_float[offset + i * 4 + 3] = triangle.edge2_texture.v;

    offset += count * 4;

    dst_uint[offset + i] = triangle.material_id;

    offset += count;

    dst_uint[offset + i] = triangle.light_id;
  }
}

void struct_triangles_deinterleave(Triangle* dst, Triangle* src, uint32_t count) {
  for (uint32_t i = 0; i < count; i++) {
    float* src_float   = (float*) src;
    uint32_t* src_uint = (uint32_t*) src;

    Triangle triangle;

    uint32_t offset = 0;

    triangle.vertex.x = src_float[offset + i * 4 + 0];
    triangle.vertex.y = src_float[offset + i * 4 + 1];
    triangle.vertex.z = src_float[offset + i * 4 + 2];
    triangle.edge1.x  = src_float[offset + i * 4 + 3];

    offset += count * 4;

    triangle.edge1.y = src_float[offset + i * 4 + 0];
    triangle.edge1.z = src_float[offset + i * 4 + 1];
    triangle.edge2.x = src_float[offset + i * 4 + 2];
    triangle.edge2.y = src_float[offset + i * 4 + 3];

    offset += count * 4;

    triangle.edge2.z         = src_float[offset + i * 4 + 0];
    triangle.vertex_normal.x = src_float[offset + i * 4 + 1];
    triangle.vertex_normal.y = src_float[offset + i * 4 + 2];
    triangle.vertex_normal.z = src_float[offset + i * 4 + 3];

    offset += count * 4;

    triangle.edge1_normal.x = src_float[offset + i * 4 + 0];
    triangle.edge1_normal.y = src_float[offset + i * 4 + 1];
    triangle.edge1_normal.z = src_float[offset + i * 4 + 2];
    triangle.edge2_normal.x = src_float[offset + i * 4 + 3];

    offset += count * 4;

    triangle.edge2_normal.y   = src_float[offset + i * 4 + 0];
    triangle.edge2_normal.z   = src_float[offset + i * 4 + 1];
    triangle.vertex_texture.u = src_float[offset + i * 4 + 2];
    triangle.vertex_texture.v = src_float[offset + i * 4 + 3];

    offset += count * 4;

    triangle.edge1_texture.u = src_float[offset + i * 4 + 0];
    triangle.edge1_texture.v = src_float[offset + i * 4 + 1];
    triangle.edge2_texture.u = src_float[offset + i * 4 + 2];
    triangle.edge2_texture.v = src_float[offset + i * 4 + 3];

    offset += count * 4;

    triangle.material_id = src_uint[offset + i];

    offset += count;

    triangle.light_id = src_uint[offset + i];

    dst[i] = triangle;
  }
}
