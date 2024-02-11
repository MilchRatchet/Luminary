#include "math.cuh"
#include "memory.cuh"
#include "utils.cuh"

__global__ void dmm_setup_vertex_directions(half* dst) {
  int id = THREAD_ID;

  while (id < device.scene.triangle_data.triangle_count) {
    const float4 data0 = __ldg((float4*) triangle_get_entry_address(2, 0, id));
    const float4 data1 = __ldg((float4*) triangle_get_entry_address(3, 0, id));
    const float2 data2 = __ldg((float2*) triangle_get_entry_address(4, 0, id));

    const vec3 vertex_normal = get_vector(data0.y, data0.z, data0.w);
    const vec3 edge1_normal  = get_vector(data1.x, data1.y, data1.z);
    const vec3 edge2_normal  = get_vector(data1.w, data2.x, data2.y);

    vec3 v0 = vertex_normal;
    vec3 v1 = add_vector(vertex_normal, edge1_normal);
    vec3 v2 = add_vector(vertex_normal, edge2_normal);

    v0 = scale_vector(v0, 0.1f);
    v1 = scale_vector(v1, 0.1f);
    v2 = scale_vector(v2, 0.1f);

    dst[9 * id + 0] = (half) v0.x;
    dst[9 * id + 1] = (half) v0.y;
    dst[9 * id + 2] = (half) v0.z;
    dst[9 * id + 3] = (half) v1.x;
    dst[9 * id + 4] = (half) v1.y;
    dst[9 * id + 5] = (half) v1.z;
    dst[9 * id + 6] = (half) v2.x;
    dst[9 * id + 7] = (half) v2.y;
    dst[9 * id + 8] = (half) v2.z;

    id += blockDim.x * gridDim.x;
  }
}
