#include "bvh_utils.cuh"
#include "math.cuh"
#include "memory.cuh"
#include "ocean_utils.cuh"
#include "toy_utils.cuh"
#include "utils.cuh"

__global__ __launch_bounds__(THREADS_PER_BLOCK, 12) void preprocess_trace_tasks() {
  const int task_count = device.trace_count[THREAD_ID];

  for (int i = 0; i < task_count; i++) {
    const int offset = get_task_address(i);
    TraceTask task   = load_trace_task(device.trace_tasks + offset);

    const uint32_t pixel = get_pixel_id(task.index.x, task.index.y);

    float depth     = device.scene.camera.far_clip_distance;
    uint32_t hit_id = HIT_TYPE_SKY;

    uint32_t light_id;

    if (device.iteration_type == TYPE_LIGHT) {
      light_id = device.ptrs.light_sample_history[pixel];
    }

    if (device.shading_mode != SHADING_HEAT) {
      if (is_first_ray() || (device.iteration_type == TYPE_LIGHT && light_id <= LIGHT_ID_TRIANGLE_ID_LIMIT)) {
        uint32_t t_id;
        TraversalTriangle tt;
        uint32_t material_id;
        if (device.iteration_type == TYPE_LIGHT) {
          const TriangleLight tri_light = load_triangle_light(device.scene.triangle_lights, light_id);
          t_id                          = tri_light.triangle_id;
          material_id                   = tri_light.material_id;
          tt.vertex                     = tri_light.vertex;
          tt.edge1                      = tri_light.edge1;
          tt.edge2                      = tri_light.edge2;
          tt.albedo_tex                 = device.scene.materials[material_id].albedo_map;
          tt.id                         = t_id;
        }
        else {
          t_id = device.ptrs.trace_result_buffer[get_pixel_id(task.index.x, task.index.y)].hit_id;
          if (t_id <= HIT_TYPE_TRIANGLE_ID_LIMIT) {
            material_id = load_triangle_material_id(t_id);

            const float4 data0 = __ldg((float4*) triangle_get_entry_address(0, 0, t_id));
            const float4 data1 = __ldg((float4*) triangle_get_entry_address(1, 0, t_id));
            const float data2  = __ldg((float*) triangle_get_entry_address(2, 0, t_id));

            tt.vertex     = get_vector(data0.x, data0.y, data0.z);
            tt.edge1      = get_vector(data0.w, data1.x, data1.y);
            tt.edge2      = get_vector(data1.z, data1.w, data2);
            tt.id         = t_id;
            tt.albedo_tex = device.scene.materials[material_id].albedo_map;
          }
        }

        if (t_id <= HIT_TYPE_TRIANGLE_ID_LIMIT) {
          // This feature does not work for displacement triangles
          if (device.scene.materials[material_id].normal_map == TEXTURE_NONE) {
            float2 coords;
            const float dist = bvh_triangle_intersection_uv(tt, task.origin, task.ray, coords);

            if (dist < depth) {
              const int alpha_result = bvh_triangle_intersection_alpha_test(tt, t_id, coords);

              if (alpha_result != 2) {
                depth  = dist;
                hit_id = t_id;
              }
              else if (device.iteration_type == TYPE_LIGHT) {
                depth  = -1.0f;
                hit_id = HIT_TYPE_REJECT;
              }
            }
            else if (device.iteration_type == TYPE_LIGHT && dist == FLT_MAX) {
              depth  = -1.0f;
              hit_id = HIT_TYPE_REJECT;
            }
          }
          else if (device.iteration_type == TYPE_LIGHT) {
            depth  = -1.0f;
            hit_id = HIT_TYPE_REJECT;
          }
        }
      }
    }

    if (
      device.scene.toy.active && (!device.scene.toy.flashlight_mode || (device.iteration_type == TYPE_LIGHT && light_id == LIGHT_ID_TOY))) {
      const float toy_dist = get_toy_distance(task.origin, task.ray);

      if (toy_dist < depth) {
        depth  = toy_dist;
        hit_id = (device.iteration_type == TYPE_LIGHT && light_id != LIGHT_ID_TOY && device.scene.toy.albedo.a == 1.0f) ? HIT_TYPE_REJECT
                                                                                                                        : HIT_TYPE_TOY;
      }
    }

    if (device.scene.ocean.active && device.iteration_type != TYPE_LIGHT) {
      if (task.origin.y < OCEAN_MIN_HEIGHT && task.origin.y > OCEAN_MAX_HEIGHT) {
        const float far_distance = ocean_far_distance(task.origin, task.ray);

        if (far_distance < depth) {
          depth  = far_distance;
          hit_id = HIT_TYPE_REJECT;
        }
      }
    }

    float2 result;
    result.x = depth;
    result.y = __uint_as_float(hit_id);

    __stcs((float2*) (device.ptrs.trace_results + offset), result);
  }
}
