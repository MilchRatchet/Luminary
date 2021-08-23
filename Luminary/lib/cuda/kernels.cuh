#ifndef CU_KERNELS_H
#define CU_KERNELS_H

#include <cuda_runtime_api.h>
#include <float.h>
#include <immintrin.h>
#include <math.h>
#include <optix.h>
#include <optix_stubs.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <chrono>
#include <thread>

#include "SDL/SDL.h"
#include "brdf.cuh"
#include "bvh.cuh"
#include "directives.cuh"
#include "image.h"
#include "math.cuh"
#include "memory.cuh"
#include "mesh.h"
#include "ocean.cuh"
#include "primitives.h"
#include "random.cuh"
#include "raytrace.h"
#include "scene.h"
#include "sky.cuh"
#include "toy.cuh"
#include "utils.cuh"

__device__ TraceTask get_starting_ray(TraceTask task) {
  vec3 default_ray;

  const float random_offset = sample_blue_noise(task.index.x, task.index.y, task.state, 8);

  default_ray.x =
    device_scene.camera.focal_length * (-device_scene.camera.fov + device_step * task.index.x + device_offset_x * random_offset * 2.0f);
  default_ray.y =
    device_scene.camera.focal_length * (device_vfov - device_step * task.index.y - device_offset_y * fractf(random_offset * 10.0f) * 2.0f);
  default_ray.z = -device_scene.camera.focal_length;

  const float alpha =
    (device_scene.camera.aperture_size == 0.0f) ? 0.0f : sample_blue_noise(task.index.x, task.index.y, task.state, 0) * 2.0f * PI;
  const float beta = (device_scene.camera.aperture_size == 0.0f)
                       ? 0.0f
                       : sqrtf(sample_blue_noise(task.index.x, task.index.y, task.state, 1)) * device_scene.camera.aperture_size;

  vec3 point_on_aperture = get_vector(cosf(alpha) * beta, sinf(alpha) * beta, 0.0f);

  default_ray       = sub_vector(default_ray, point_on_aperture);
  point_on_aperture = rotate_vector_by_quaternion(point_on_aperture, device_camera_rotation);

  task.ray    = normalize_vector(rotate_vector_by_quaternion(default_ray, device_camera_rotation));
  task.origin = add_vector(device_scene.camera.pos, point_on_aperture);

  return task;
}

__global__ void initialize_randoms() {
  unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;

  curandStateXORWOW_t state;
  curand_init(id, 0, 0, &state);
  device_sample_randoms[id] = state;
}

__global__ __launch_bounds__(THREADS_PER_BLOCK, 12) void generate_trace_tasks() {
  int offset = 0;

  for (int pixel = threadIdx.x + blockIdx.x * blockDim.x; pixel < device_amount; pixel += blockDim.x * gridDim.x) {
    TraceTask task;

    task.index.x = (uint16_t) (pixel % device_width);
    task.index.y = (uint16_t) (pixel / device_width);

    task.state = (1 << 31) | (device_max_ray_depth << 16) | (device_temporal_frames & RANDOM_INDEX);

    task = get_starting_ray(task);

    device_records[pixel]              = get_color(1.0f, 1.0f, 1.0f);
    device_frame_buffer[pixel]         = get_color(0.0f, 0.0f, 0.0f);
    device_light_sample_history[pixel] = ANY_LIGHT;

    if (device_denoiser)
      device_albedo_buffer[pixel] = get_color(0.0f, 0.0f, 0.0f);

    store_trace_task(device_tasks + get_task_address(offset++), task);
  }

  device_task_counts[(threadIdx.x + blockIdx.x * blockDim.x) * 4] = offset;
}

__global__ __launch_bounds__(THREADS_PER_BLOCK, 10) void balance_trace_tasks() {
  const int warp = (threadIdx.x + blockIdx.x * blockDim.x) >> 5;

  if (warp >= (THREADS_PER_BLOCK * BLOCKS_PER_GRID) >> 5)
    return;

  __shared__ uint16_t counts[THREADS_PER_BLOCK][32];
  uint16_t average = 0;

  for (int i = 0; i < 32; i++) {
    const uint16_t c       = device_task_counts[4 * (32 * warp + i)];
    counts[threadIdx.x][i] = c;
    average += c;
  }

  average = average >> 5;

  for (int i = 0; i < 32; i++) {
    uint16_t count        = counts[threadIdx.x][i];
    int source_index      = -1;
    uint16_t source_count = 0;

    if (count >= average)
      continue;

    for (int j = 0; j < 32; j++) {
      uint16_t c = counts[threadIdx.x][j];
      if (c > average && c > count + 1 && c > source_count) {
        source_count = c;
        source_index = j;
      }
    }

    if (source_index != -1) {
      const int swaps = (source_count - count) >> 1;
      for (int j = 0; j < swaps; j++) {
        source_count--;
        float4* source_ptr =
          (float4*) (device_tasks + get_task_address_of_thread(((warp & 0b11) << 5) + source_index, warp >> 2, source_count));
        float4* sink_ptr = (float4*) (device_tasks + get_task_address_of_thread(((warp & 0b11) << 5) + i, warp >> 2, count));
        count++;

        __stwb(sink_ptr, __ldca(source_ptr));
        __stwb(sink_ptr + 1, __ldca(source_ptr + 1));
      }
      counts[threadIdx.x][i]            = count;
      counts[threadIdx.x][source_index] = source_count;
    }
  }

  for (int i = 0; i < 32; i++) {
    device_task_counts[4 * (32 * warp + i)] = counts[threadIdx.x][i];
  }
}

__global__ __launch_bounds__(THREADS_PER_BLOCK, 12) void preprocess_trace_tasks() {
  const int task_count = device_task_counts[(threadIdx.x + blockIdx.x * blockDim.x) * 4];

  for (int i = 0; i < task_count; i++) {
    const int offset = get_task_address(i);
    TraceTask task   = load_trace_task_essentials(device_tasks + offset);

    float depth     = device_scene.camera.far_clip_distance;
    uint32_t hit_id = SKY_HIT;

    if (device_scene.ocean.active) {
      const float ocean_dist = get_intersection_ocean(task.origin, task.ray, depth);

      if (ocean_dist < depth) {
        depth  = ocean_dist;
        hit_id = OCEAN_HIT;
      }
    }

    if (device_scene.toy.active) {
      const float toy_dist = get_toy_distance(task.origin, task.ray);

      if (toy_dist < depth) {
        depth  = toy_dist;
        hit_id = TOY_HIT;
      }
    }

    float2 result;
    result.x = depth;
    result.y = uint_as_float(hit_id);

    __stcs((float2*) (device_trace_results + offset), result);
  }
}

__global__ __launch_bounds__(THREADS_PER_BLOCK, 12) void postprocess_trace_tasks() {
  const int task_count         = device_task_counts[(threadIdx.x + blockIdx.x * blockDim.x) * 4];
  uint16_t geometry_task_count = 0;
  uint16_t sky_task_count      = 0;
  uint16_t ocean_task_count    = 0;
  uint16_t toy_task_count      = 0;

  // count data
  for (int i = 0; i < task_count; i++) {
    const int offset      = get_task_address(i);
    const uint32_t hit_id = __ldca((uint32_t*) (device_trace_results + offset) + 1);

    if (hit_id == SKY_HIT) {
      sky_task_count++;
    }
    else if (hit_id == OCEAN_HIT) {
      ocean_task_count++;
    }
    else if (hit_id == TOY_HIT) {
      toy_task_count++;
    }
    else {
      geometry_task_count++;
    }
  }

  int geometry_offset = 0;
  int ocean_offset    = geometry_task_count;
  int sky_offset      = geometry_task_count + ocean_task_count;
  int toy_offset      = geometry_task_count + ocean_task_count + sky_task_count;
  int k               = 0;

  ushort4 task_offsets;
  task_offsets.x = geometry_offset;
  task_offsets.y = ocean_offset;
  task_offsets.z = sky_offset;
  task_offsets.w = toy_offset;
  __stcs((ushort4*) (device_task_offsets + (threadIdx.x + blockIdx.x * blockDim.x) * 4), task_offsets);

  // order data
  while (k < task_count) {
    const int offset      = get_task_address(k);
    const uint32_t hit_id = __ldca((uint32_t*) (device_trace_results + offset) + 1);

    int index;
    int needs_swapping;

    if (hit_id == SKY_HIT) {
      index          = sky_offset++;
      needs_swapping = (k < geometry_task_count + ocean_task_count) || (k >= geometry_task_count + ocean_task_count + sky_task_count);
    }
    else if (hit_id == OCEAN_HIT) {
      index          = ocean_offset++;
      needs_swapping = (k < geometry_task_count) || (k >= geometry_task_count + ocean_task_count);
    }
    else if (hit_id == TOY_HIT) {
      index          = toy_offset++;
      needs_swapping = (k < geometry_task_count + ocean_task_count + sky_task_count);
    }
    else {
      index          = geometry_offset++;
      needs_swapping = (k >= geometry_task_count);
    }

    if (needs_swapping) {
      swap_trace_data(k, index);
    }
    else {
      k++;
    }
  }

  geometry_task_count = 0;
  sky_task_count      = 0;
  ocean_task_count    = 0;
  toy_task_count      = 0;

  // process data
  for (int i = 0; i < task_count; i++) {
    const int offset    = get_task_address(i);
    TraceTask task      = load_trace_task(device_tasks + offset);
    const float2 result = __ldcs((float2*) (device_trace_results + offset));

    const float depth     = result.x;
    const uint32_t hit_id = float_as_uint(result.y);

    float4* ptr = (float4*) (device_tasks + offset);
    float4 data0;
    float4 data1;

    if (hit_id == SKY_HIT) {
      sky_task_count++;
      data0.x = task.ray.x;
      data0.y = task.ray.y;
      data0.z = task.ray.z;
      data0.w = *((float*) (&task.index));
    }
    else {
      task.origin = add_vector(task.origin, scale_vector(task.ray, depth));
      data0.x     = task.origin.x;
      data0.y     = task.origin.y;
      data0.z     = task.origin.z;
      if (hit_id == TOY_HIT) {
        toy_task_count++;
        data0.w = task.ray.x;
        data1.x = task.ray.y;
        data1.y = task.ray.z;
      }
      else {
        data0.w = asinf(task.ray.y);
        data1.x = atan2f(task.ray.z, task.ray.x);

        if (hit_id == OCEAN_HIT) {
          ocean_task_count++;
          data1.y = depth;
        }
        else {
          geometry_task_count++;
          data1.y = uint_as_float(hit_id);
        }
      }

      data1.z = *((float*) &task.index);
      data1.w = *((float*) &task.state);

      __stcs(ptr + 1, data1);
    }

    __stcs(ptr, data0);
  }

  ushort4 task_counts;
  task_counts.x = geometry_task_count;
  task_counts.y = ocean_task_count;
  task_counts.z = sky_task_count;
  task_counts.w = toy_task_count;

  __stcs((ushort4*) (device_task_counts + (threadIdx.x + blockIdx.x * blockDim.x) * 4), task_counts);
}

__global__ __launch_bounds__(THREADS_PER_BLOCK, 9) void process_geometry_tasks() {
  int trace_count      = 0;
  const int task_count = device_task_counts[(threadIdx.x + blockIdx.x * blockDim.x) * 4];

  for (int i = 0; i < task_count; i++) {
    GeometryTask task = load_geometry_task(device_tasks + get_task_address(i));
    const int pixel   = task.index.y * device_width + task.index.x;

    vec3 ray;
    ray.x = cosf(task.ray_xz) * cosf(task.ray_y);
    ray.y = sinf(task.ray_y);
    ray.z = sinf(task.ray_xz) * cosf(task.ray_y);

    task.state = (task.state & ~DEPTH_LEFT) | (((task.state & DEPTH_LEFT) - 1) & DEPTH_LEFT);

    const float4* hit_address = (float4*) (device_scene.triangles + task.hit_id);

    const float4 t1 = __ldg(hit_address);
    const float4 t2 = __ldg(hit_address + 1);
    const float4 t3 = __ldg(hit_address + 2);
    const float4 t4 = __ldg(hit_address + 3);
    const float4 t5 = __ldg(hit_address + 4);
    const float4 t6 = __ldg(hit_address + 5);
    const float2 t7 = __ldg((float2*) (hit_address + 6));

    vec3 vertex = get_vector(t1.x, t1.y, t1.z);
    vec3 edge1  = get_vector(t1.w, t2.x, t2.y);
    vec3 edge2  = get_vector(t2.z, t2.w, t3.x);

    vec3 face_normal = normalize_vector(cross_product(edge1, edge2));

    vec3 normal = get_coordinates_in_triangle(vertex, edge1, edge2, task.position);

    const float lambda = normal.x;
    const float mu     = normal.y;

    vec3 vertex_normal = get_vector(t3.y, t3.z, t3.w);
    vec3 edge1_normal  = get_vector(t4.x, t4.y, t4.z);
    vec3 edge2_normal  = get_vector(t4.w, t5.x, t5.y);

    normal = lerp_normals(vertex_normal, edge1_normal, edge2_normal, lambda, mu, face_normal);

    UV vertex_texture = get_UV(t5.z, t5.w);
    UV edge1_texture  = get_UV(t6.x, t6.y);
    UV edge2_texture  = get_UV(t6.z, t6.w);

    const UV tex_coords = lerp_uv(vertex_texture, edge1_texture, edge2_texture, lambda, mu);

    if (dot_product(normal, face_normal) < 0.0f) {
      face_normal = scale_vector(face_normal, -1.0f);
    }

    if (dot_product(face_normal, scale_vector(ray, -1.0f)) < 0.0f) {
      normal = scale_vector(normal, -1.0f);
    }

    const int texture_object         = __float_as_int(t7.x);
    const uint32_t triangle_light_id = __float_as_uint(t7.y);

    const ushort4 maps = __ldg((ushort4*) (device_texture_assignments + texture_object));

    float roughness;
    float metallic;
    float intensity;

    if (maps.z) {
      const float4 material_f = tex2D<float4>(device_material_atlas[maps.z], tex_coords.u, 1.0f - tex_coords.v);

      roughness = (1.0f - material_f.x) * (1.0f - material_f.x);
      metallic  = material_f.y;
      intensity = material_f.z * 255.0f;
    }
    else {
      roughness = (1.0f - device_default_material.r) * (1.0f - device_default_material.r);
      metallic  = device_default_material.g;
      intensity = device_default_material.b;
    }

    RGBAF albedo;

    if (maps.x) {
      const float4 albedo_f = tex2D<float4>(device_albedo_atlas[maps.x], tex_coords.u, 1.0f - tex_coords.v);
      albedo.r              = albedo_f.x;
      albedo.g              = albedo_f.y;
      albedo.b              = albedo_f.z;
      albedo.a              = albedo_f.w;

      albedo = saturate_albedo(albedo, 0.0f);
    }
    else {
      albedo.r = 0.9f;
      albedo.g = 0.9f;
      albedo.b = 0.9f;
      albedo.a = 1.0f;
    }

    RGBF emission = get_color(0.0f, 0.0f, 0.0f);

    if (maps.y && device_lights_active) {
      const float4 illuminance_f = tex2D<float4>(device_illuminance_atlas[maps.y], tex_coords.u, 1.0f - tex_coords.v);

      emission = get_color(illuminance_f.x, illuminance_f.y, illuminance_f.z);
    }

    if (albedo.a < device_scene.camera.alpha_cutoff)
      albedo.a = 0.0f;

    RGBF record = device_records[pixel];

    if (albedo.a > 0.0f && (emission.r > 0.0f || emission.g > 0.0f || emission.b > 0.0f)) {
      if (device_denoiser && task.state & ALBEDO_BUFFER_STATE) {
        device_albedo_buffer[pixel] = emission;
      }

      if (!isnan(record.r) && !isinf(record.r) && !isnan(record.g) && !isinf(record.g) && !isnan(record.b) && !isinf(record.b)) {
        emission.r *= intensity * record.r;
        emission.g *= intensity * record.g;
        emission.b *= intensity * record.b;

        const uint32_t light = device_light_sample_history[pixel];

        if (light == ANY_LIGHT || light == triangle_light_id) {
          device_frame_buffer[pixel] = emission;
        }
      }

      atomicSub(&device_pixels_left, 1);
    }
    else if (task.state & DEPTH_LEFT) {
      if (device_denoiser && task.state & ALBEDO_BUFFER_STATE) {
        device_albedo_buffer[pixel] = get_color(albedo.r, albedo.g, albedo.b);
        task.state ^= ALBEDO_BUFFER_STATE;
      }

      uint32_t light_sample_id;

      if (sample_blue_noise(task.index.x, task.index.y, task.state, 40) > albedo.a) {
        task.position = add_vector(task.position, scale_vector(ray, 2.0f * eps));

        record.r *= (albedo.r * albedo.a + 1.0f - albedo.a);
        record.g *= (albedo.g * albedo.a + 1.0f - albedo.a);
        record.b *= (albedo.b * albedo.a + 1.0f - albedo.a);

        light_sample_id = device_light_sample_history[pixel];
      }
      else {
        const float specular_probability = lerp(0.5f, 1.0f, metallic);

        const vec3 V = scale_vector(ray, -1.0f);

        task.position = add_vector(task.position, scale_vector(normal, 8.0f * eps));

        int light_count;
        Light light;
        light = sample_light(task.position, light_count, light_sample_id, sample_blue_noise(task.index.x, task.index.y, task.state, 51));

        const float light_sample = sample_blue_noise(task.index.x, task.index.y, task.state, 50);

        const float light_sample_depth_bias = powf(0.1f, device_max_ray_depth - ((task.state & DEPTH_LEFT) >> 16));
        const float light_sample_probability =
          fmaxf(1.0f - 1.0f / (light_count + 1), 1.0f - (1 + device_temporal_frames) * light_sample_depth_bias);

        const float gamma = 2.0f * PI * sample_blue_noise(task.index.x, task.index.y, task.state, 3);
        const float beta  = sample_blue_noise(task.index.x, task.index.y, task.state, 2);

        if (sample_blue_noise(task.index.x, task.index.y, task.state, 10) < specular_probability) {
          ray = specular_BRDF(
            record, light_sample_id, normal, V, light, light_sample, light_sample_probability, light_count, albedo, roughness, metallic,
            beta, gamma, specular_probability);
        }
        else {
          ray = diffuse_BRDF(
            record, light_sample_id, normal, V, light, light_sample, light_sample_probability, light_count, albedo, roughness, metallic,
            beta, gamma, specular_probability);
        }
      }

      task.state = (task.state & ~RANDOM_INDEX) | (((task.state & RANDOM_INDEX) + 1) & RANDOM_INDEX);

      int remains_active = 1;

#ifdef WEIGHT_BASED_EXIT
      const float max_record = fmaxf(record.r, fmaxf(record.g, record.b));
      if (
        max_record < CUTOFF
        || (max_record < PROBABILISTIC_CUTOFF && sample_blue_noise(task.index.x, task.index.y, task.state, 20) > (max_record - CUTOFF) / (CUTOFF - PROBABILISTIC_CUTOFF))) {
        remains_active = 0;
      }
#endif

#ifdef LOW_QUALITY_LONG_BOUNCES
      if (
        albedo.a > 0.0f && ((task.state & DEPTH_LEFT) >> 16) <= (device_max_ray_depth - MIN_BOUNCES)
        && sample_blue_noise(task.index.x, task.index.y, task.state, 21) < 1.0f / device_max_ray_depth) {
        remains_active = 0;
      }
#endif

      if (remains_active) {
        TraceTask next_task;
        next_task.origin = task.position;
        next_task.ray    = ray;
        next_task.index  = task.index;
        next_task.state  = task.state;

        store_trace_task(device_tasks + get_task_address(trace_count++), next_task);

        device_records[pixel]              = record;
        device_light_sample_history[pixel] = light_sample_id;
      }
      else {
        atomicSub(&device_pixels_left, 1);
      }
    }
    else {
      atomicSub(&device_pixels_left, 1);
    }
  }

  device_task_counts[(threadIdx.x + blockIdx.x * blockDim.x) * 4] = trace_count;
}

__global__ __launch_bounds__(THREADS_PER_BLOCK, 9) void process_debug_geometry_tasks() {
  const int task_count = device_task_counts[(threadIdx.x + blockIdx.x * blockDim.x) * 4];

  for (int i = 0; i < task_count; i++) {
    GeometryTask task = load_geometry_task(device_tasks + get_task_address(i));
    const int pixel   = task.index.y * device_width + task.index.x;

    if (device_shading_mode == SHADING_ALBEDO) {
      const float4* hit_address = (float4*) (device_scene.triangles + task.hit_id);

      const float4 t1 = __ldg(hit_address);
      const float4 t2 = __ldg(hit_address + 1);
      const float4 t3 = __ldg(hit_address + 2);
      const float4 t4 = __ldg(hit_address + 3);
      const float4 t5 = __ldg(hit_address + 4);
      const float4 t6 = __ldg(hit_address + 5);
      const float2 t7 = __ldg((float2*) (hit_address + 6));

      vec3 vertex = get_vector(t1.x, t1.y, t1.z);
      vec3 edge1  = get_vector(t1.w, t2.x, t2.y);
      vec3 edge2  = get_vector(t2.z, t2.w, t3.x);

      vec3 normal = get_coordinates_in_triangle(vertex, edge1, edge2, task.position);

      const float lambda = normal.x;
      const float mu     = normal.y;

      UV vertex_texture = get_UV(t5.z, t5.w);
      UV edge1_texture  = get_UV(t6.x, t6.y);
      UV edge2_texture  = get_UV(t6.z, t6.w);

      const UV tex_coords = lerp_uv(vertex_texture, edge1_texture, edge2_texture, lambda, mu);

      const int texture_object = __float_as_int(t7.x);

      const ushort4 maps = __ldg((ushort4*) (device_texture_assignments + texture_object));

      RGBF color = get_color(0.0f, 0.0f, 0.0f);

      if (maps.x) {
        const float4 albedo_f = tex2D<float4>(device_albedo_atlas[maps.x], tex_coords.u, 1.0f - tex_coords.v);
        color                 = add_color(color, get_color(albedo_f.x, albedo_f.y, albedo_f.z));
      }
      else {
        color = add_color(color, get_color(0.9f, 0.9f, 0.9f));
      }

      if (maps.y && device_lights_active) {
        const float4 illuminance_f = tex2D<float4>(device_illuminance_atlas[maps.y], tex_coords.u, 1.0f - tex_coords.v);

        color = add_color(color, get_color(illuminance_f.x, illuminance_f.y, illuminance_f.z));
      }

      device_frame_buffer[pixel] = color;
    }
    else if (device_shading_mode == SHADING_DEPTH) {
      const float dist           = get_length(sub_vector(device_scene.camera.pos, task.position));
      const float value          = __saturatef((1.0f / dist) * 2.0f);
      device_frame_buffer[pixel] = get_color(value, value, value);
    }
    else if (device_shading_mode == SHADING_NORMAL) {
      const float4* hit_address = (float4*) (device_scene.triangles + task.hit_id);

      const float4 t1 = __ldg(hit_address);
      const float4 t2 = __ldg(hit_address + 1);
      const float4 t3 = __ldg(hit_address + 2);
      const float4 t4 = __ldg(hit_address + 3);
      const float4 t5 = __ldg(hit_address + 4);

      vec3 vertex = get_vector(t1.x, t1.y, t1.z);
      vec3 edge1  = get_vector(t1.w, t2.x, t2.y);
      vec3 edge2  = get_vector(t2.z, t2.w, t3.x);

      vec3 face_normal = normalize_vector(cross_product(edge1, edge2));

      vec3 normal = get_coordinates_in_triangle(vertex, edge1, edge2, task.position);

      const float lambda = normal.x;
      const float mu     = normal.y;

      vec3 vertex_normal = get_vector(t3.y, t3.z, t3.w);
      vec3 edge1_normal  = get_vector(t4.x, t4.y, t4.z);
      vec3 edge2_normal  = get_vector(t4.w, t5.x, t5.y);

      normal = lerp_normals(vertex_normal, edge1_normal, edge2_normal, lambda, mu, face_normal);

      device_frame_buffer[pixel] = get_color(__saturatef(normal.x), __saturatef(normal.y), __saturatef(normal.z));
    }
    else if (device_shading_mode == SHADING_HEAT) {
      const float cost  = uint_as_float(task.hit_id);
      const float value = 1.0f - 1.0f / (powf(cost, 0.25f));
      const float red   = __saturatef(2.0f * value);
      const float green = __saturatef(2.0f * (value - 0.5f));
      const float blue  = __saturatef((value > 0.5f) ? 4.0f * (0.25f - fabsf(value - 1.0f)) : 4.0f * (0.25f - fabsf(value - 0.25f)));
      device_frame_buffer[pixel] = get_color(red, green, blue);
    }

    atomicSub(&device_pixels_left, 1);
  }
}

__global__ __launch_bounds__(THREADS_PER_BLOCK, 10) void process_ocean_tasks() {
  const int id = threadIdx.x + blockIdx.x * blockDim.x;

  int trace_count       = device_task_counts[id * 4];
  const int task_count  = device_task_counts[id * 4 + 1];
  const int task_offset = device_task_offsets[id * 4 + 1];

  for (int i = 0; i < task_count; i++) {
    OceanTask task  = load_ocean_task(device_tasks + get_task_address(task_offset + i));
    const int pixel = task.index.y * device_width + task.index.x;

    vec3 ray;
    ray.x = cosf(task.ray_xz) * cosf(task.ray_y);
    ray.y = sinf(task.ray_y);
    ray.z = sinf(task.ray_xz) * cosf(task.ray_y);

    task.state = (task.state & ~DEPTH_LEFT) | (((task.state & DEPTH_LEFT) - 1) & DEPTH_LEFT);

    const vec3 normal = get_ocean_normal(task.position, fmaxf(0.1f * eps, task.distance * 0.1f / device_width));

    RGBAF albedo = device_scene.ocean.albedo;
    RGBF record  = device_records[pixel];

    if (device_scene.ocean.emissive) {
      RGBF emission = get_color(albedo.r, albedo.g, albedo.b);

      if (device_denoiser && task.state & ALBEDO_BUFFER_STATE) {
        device_albedo_buffer[pixel] = emission;
      }

      if (!isnan(record.r) && !isinf(record.r) && !isnan(record.g) && !isinf(record.g) && !isnan(record.b) && !isinf(record.b)) {
        emission.r *= 2.0f * record.r;
        emission.g *= 2.0f * record.g;
        emission.b *= 2.0f * record.b;

        device_frame_buffer[pixel] = emission;
      }

      atomicSub(&device_pixels_left, 1);
    }
    else if (task.state & DEPTH_LEFT) {
      if (device_denoiser && task.state & ALBEDO_BUFFER_STATE) {
        device_albedo_buffer[pixel] = get_color(albedo.r, albedo.g, albedo.b);
        task.state ^= ALBEDO_BUFFER_STATE;
      }

      if (sample_blue_noise(task.index.x, task.index.y, task.state, 40) > albedo.a) {
        task.position = add_vector(task.position, scale_vector(ray, 2.0f * eps));

        record.r *= (albedo.r * albedo.a + 1.0f - albedo.a);
        record.g *= (albedo.g * albedo.a + 1.0f - albedo.a);
        record.b *= (albedo.b * albedo.a + 1.0f - albedo.a);

        const float refraction_index = 1.0f / device_scene.ocean.refractive_index;

        ray = refraction_BRDF(record, normal, ray, 0.0f, refraction_index, 0.0f, 0.0f);
      }
      else {
        const vec3 V = scale_vector(ray, -1.0f);

        task.position = add_vector(task.position, scale_vector(normal, 8.0f * eps));

        const float gamma = 2.0f * PI * sample_blue_noise(task.index.x, task.index.y, task.state, 3);
        const float beta  = sample_blue_noise(task.index.x, task.index.y, task.state, 2);

        Light light = {};
        uint32_t light_sample_id;
        if (sample_blue_noise(task.index.x, task.index.y, task.state, 10) < 0.5f) {
          ray = specular_BRDF(record, light_sample_id, normal, V, light, 1.0f, 0.0f, 0, albedo, 0.0f, 0.0f, beta, gamma, 0.5f);
        }
        else {
          ray = diffuse_BRDF(record, light_sample_id, normal, V, light, 1.0f, 0.0f, 0, albedo, 0.0f, 0.0f, beta, gamma, 0.5f);
        }
      }

      task.state = (task.state & ~RANDOM_INDEX) | (((task.state & RANDOM_INDEX) + 1) & RANDOM_INDEX);

      TraceTask next_task;
      next_task.origin = task.position;
      next_task.ray    = ray;
      next_task.index  = task.index;
      next_task.state  = task.state;

      store_trace_task(device_tasks + get_task_address(trace_count++), next_task);

      device_records[pixel]              = record;
      device_light_sample_history[pixel] = ANY_LIGHT;
    }
    else {
      atomicSub(&device_pixels_left, 1);
    }
  }

  device_task_counts[id * 4] = trace_count;
}

__global__ __launch_bounds__(THREADS_PER_BLOCK, 10) void process_debug_ocean_tasks() {
  const int id = threadIdx.x + blockIdx.x * blockDim.x;

  const int task_count  = device_task_counts[id * 4 + 1];
  const int task_offset = device_task_offsets[id * 4 + 1];

  for (int i = 0; i < task_count; i++) {
    OceanTask task  = load_ocean_task(device_tasks + get_task_address(task_offset + i));
    const int pixel = task.index.y * device_width + task.index.x;

    if (device_shading_mode == SHADING_ALBEDO) {
      RGBAF albedo               = device_scene.ocean.albedo;
      device_frame_buffer[pixel] = get_color(albedo.r, albedo.g, albedo.b);
    }
    else if (device_shading_mode == SHADING_DEPTH) {
      const float value          = __saturatef((1.0f / task.distance) * 2.0f);
      device_frame_buffer[pixel] = get_color(value, value, value);
    }
    else if (device_shading_mode == SHADING_NORMAL) {
      const vec3 normal = get_ocean_normal(task.position, fmaxf(0.1f * eps, task.distance * 0.1f / device_width));

      device_frame_buffer[pixel] = get_color(__saturatef(normal.x), __saturatef(normal.y), __saturatef(normal.z));
    }

    atomicSub(&device_pixels_left, 1);
  }
}

__global__ __launch_bounds__(THREADS_PER_BLOCK, 10) void process_sky_tasks() {
  const int id = threadIdx.x + blockIdx.x * blockDim.x;

  const int task_count  = device_task_counts[id * 4 + 2];
  const int task_offset = device_task_offsets[id * 4 + 2];

  for (int i = 0; i < task_count; i++) {
    const SkyTask task = load_sky_task(device_tasks + get_task_address(task_offset + i));
    const int pixel    = task.index.y * device_width + task.index.x;

    const RGBF record = device_records[pixel];
    RGBF sky          = get_sky_color(task.ray);
    sky               = mul_color(sky, record);

    const uint32_t light = device_light_sample_history[pixel];

    if (light == ANY_LIGHT || light == 0) {
      device_frame_buffer[pixel] = sky;
    }

    atomicSub(&device_pixels_left, 1);
  }
}

__global__ __launch_bounds__(THREADS_PER_BLOCK, 10) void process_debug_sky_tasks() {
  const int id = threadIdx.x + blockIdx.x * blockDim.x;

  const int task_count  = device_task_counts[id * 4 + 2];
  const int task_offset = device_task_offsets[id * 4 + 2];

  for (int i = 0; i < task_count; i++) {
    const SkyTask task = load_sky_task(device_tasks + get_task_address(task_offset + i));
    const int pixel    = task.index.y * device_width + task.index.x;

    if (device_shading_mode == SHADING_ALBEDO) {
      device_frame_buffer[pixel] = get_sky_color(task.ray);
    }
    else if (device_shading_mode == SHADING_DEPTH) {
      const float value          = __saturatef((1.0f / device_scene.camera.far_clip_distance) * 2.0f);
      device_frame_buffer[pixel] = get_color(value, value, value);
    }
    else if (device_shading_mode == SHADING_NORMAL) {
      device_frame_buffer[pixel] = get_color(0.0f, 0.0f, 0.0f);
    }

    atomicSub(&device_pixels_left, 1);
  }
}

__global__ __launch_bounds__(THREADS_PER_BLOCK, 9) void process_toy_tasks() {
  const int id = threadIdx.x + blockIdx.x * blockDim.x;

  int trace_count       = device_task_counts[id * 4];
  const int task_count  = device_task_counts[id * 4 + 3];
  const int task_offset = device_task_offsets[id * 4 + 3];

  for (int i = 0; i < task_count; i++) {
    ToyTask task    = load_toy_task(device_tasks + get_task_address(task_offset + i));
    const int pixel = task.index.y * device_width + task.index.x;

    task.state = (task.state & ~DEPTH_LEFT) | (((task.state & DEPTH_LEFT) - 1) & DEPTH_LEFT);

    vec3 normal     = get_toy_normal(task.position);
    int from_inside = 0;

    if (dot_product(normal, task.ray) > 0.0f) {
      normal      = scale_vector(normal, -1.0f);
      from_inside = 1;
    }

    RGBAF albedo          = device_scene.toy.albedo;
    const float roughness = (1.0f - device_scene.toy.material.r) * (1.0f - device_scene.toy.material.r);
    const float metallic  = device_scene.toy.material.g;
    const float intensity = device_scene.toy.material.b;
    RGBF emission         = get_color(device_scene.toy.emission.r, device_scene.toy.emission.g, device_scene.toy.emission.b);

    if (albedo.a < device_scene.camera.alpha_cutoff)
      albedo.a = 0.0f;

    RGBF record = device_records[pixel];

    if (albedo.a > 0.0f && device_scene.toy.emissive) {
      if (device_denoiser && task.state & ALBEDO_BUFFER_STATE) {
        device_albedo_buffer[pixel] = emission;
      }

      if (!isnan(record.r) && !isinf(record.r) && !isnan(record.g) && !isinf(record.g) && !isnan(record.b) && !isinf(record.b)) {
        emission.r *= intensity * record.r;
        emission.g *= intensity * record.g;
        emission.b *= intensity * record.b;

        const uint32_t light = device_light_sample_history[pixel];

        if (light == ANY_LIGHT || light == TOY_LIGHT) {
          device_frame_buffer[pixel] = emission;
        }
      }

      atomicSub(&device_pixels_left, 1);
    }
    else if (task.state & DEPTH_LEFT) {
      if (device_denoiser && task.state & ALBEDO_BUFFER_STATE) {
        device_albedo_buffer[pixel] = get_color(albedo.r, albedo.g, albedo.b);
        task.state ^= ALBEDO_BUFFER_STATE;
      }

      uint32_t light_sample_id;
      const float gamma = 2.0f * PI * sample_blue_noise(task.index.x, task.index.y, task.state, 3);
      const float beta  = sample_blue_noise(task.index.x, task.index.y, task.state, 2);

      if (sample_blue_noise(task.index.x, task.index.y, task.state, 40) > albedo.a) {
        task.position = add_vector(task.position, scale_vector(task.ray, eps * 8.0f));

        record.r *= (albedo.r * albedo.a + 1.0f - albedo.a);
        record.g *= (albedo.g * albedo.a + 1.0f - albedo.a);
        record.b *= (albedo.b * albedo.a + 1.0f - albedo.a);

        light_sample_id = device_light_sample_history[pixel];

        const float refraction_index = (from_inside) ? device_scene.toy.refractive_index : 1.0f / device_scene.toy.refractive_index;

        task.ray = refraction_BRDF(record, normal, task.ray, roughness, refraction_index, beta, gamma);
      }
      else {
        const float specular_probability = lerp(0.5f, 1.0f, metallic);

        const vec3 V = scale_vector(task.ray, -1.0f);

        task.position = add_vector(task.position, scale_vector(normal, 8.0f * eps));

        int light_count;
        Light light;
        light = sample_light(task.position, light_count, light_sample_id, sample_blue_noise(task.index.x, task.index.y, task.state, 51));

        const float light_sample = sample_blue_noise(task.index.x, task.index.y, task.state, 50);

        const float light_sample_depth_bias = powf(0.1f, device_max_ray_depth - ((task.state & DEPTH_LEFT) >> 16));
        const float light_sample_probability =
          fmaxf(1.0f - 1.0f / (light_count + 1), 1.0f - (1 + device_temporal_frames) * light_sample_depth_bias);

        if (sample_blue_noise(task.index.x, task.index.y, task.state, 10) < specular_probability) {
          task.ray = specular_BRDF(
            record, light_sample_id, normal, V, light, light_sample, light_sample_probability, light_count, albedo, roughness, metallic,
            beta, gamma, specular_probability);
        }
        else {
          task.ray = diffuse_BRDF(
            record, light_sample_id, normal, V, light, light_sample, light_sample_probability, light_count, albedo, roughness, metallic,
            beta, gamma, specular_probability);
        }
      }

      task.state = (task.state & ~RANDOM_INDEX) | (((task.state & RANDOM_INDEX) + 1) & RANDOM_INDEX);

      TraceTask next_task;
      next_task.origin = task.position;
      next_task.ray    = task.ray;
      next_task.index  = task.index;
      next_task.state  = task.state;

      store_trace_task(device_tasks + get_task_address(trace_count++), next_task);

      device_records[pixel]              = record;
      device_light_sample_history[pixel] = light_sample_id;
    }
    else {
      atomicSub(&device_pixels_left, 1);
    }
  }

  device_task_counts[id * 4] = trace_count;
}

__global__ __launch_bounds__(THREADS_PER_BLOCK, 12) void process_debug_toy_tasks() {
  const int id = threadIdx.x + blockIdx.x * blockDim.x;

  const int task_count  = device_task_counts[id * 4 + 3];
  const int task_offset = device_task_offsets[id * 4 + 3];

  for (int i = 0; i < task_count; i++) {
    const ToyTask task = load_toy_task(device_tasks + get_task_address(task_offset + i));
    const int pixel    = task.index.y * device_width + task.index.x;

    if (device_shading_mode == SHADING_ALBEDO) {
      device_frame_buffer[pixel] = get_color(device_scene.toy.albedo.r, device_scene.toy.albedo.g, device_scene.toy.albedo.b);
    }
    else if (device_shading_mode == SHADING_DEPTH) {
      const float dist           = get_length(sub_vector(device_scene.camera.pos, task.position));
      const float value          = __saturatef((1.0f / dist) * 2.0f);
      device_frame_buffer[pixel] = get_color(value, value, value);
    }
    else if (device_shading_mode == SHADING_NORMAL) {
      vec3 normal = get_toy_normal(task.position);

      if (dot_product(normal, task.ray) > 0.0f) {
        normal = scale_vector(normal, -1.0f);
      }

      device_frame_buffer[pixel] = get_color(normal.x, normal.y, normal.z);
    }

    atomicSub(&device_pixels_left, 1);
  }
}

__global__ void finalize_samples() {
  int offset = 4 * (threadIdx.x + blockIdx.x * blockDim.x);

  for (; offset < 3 * device_amount - 4; offset += 4 * blockDim.x * gridDim.x) {
    float4 buffer = __ldcs((float4*) ((float*) device_frame_buffer + offset));
    float4 output;
    float4 variance;
    float4 bias_cache;

    if (device_temporal_frames == 0) {
      output.x   = buffer.x;
      output.y   = buffer.y;
      output.z   = buffer.z;
      output.w   = buffer.w;
      variance.x = 1.0f;
      variance.y = 1.0f;
      variance.z = 1.0f;
      variance.w = 1.0f;
      __stcs((float4*) ((float*) device_frame_variance + offset), variance);
      bias_cache.x = 0.0f;
      bias_cache.y = 0.0f;
      bias_cache.z = 0.0f;
      bias_cache.w = 0.0f;
    }
    else {
      output     = __ldcs((float4*) ((float*) device_frame_output + offset));
      variance   = __ldcs((float4*) ((float*) device_frame_variance + offset));
      bias_cache = __ldcs((float4*) ((float*) device_frame_bias_cache + offset));
    }

    float4 deviation;
    deviation.x = sqrtf(fmaxf(eps, variance.x));
    deviation.y = sqrtf(fmaxf(eps, variance.y));
    deviation.z = sqrtf(fmaxf(eps, variance.z));
    deviation.w = sqrtf(fmaxf(eps, variance.w));

    if (device_temporal_frames) {
      variance.x = ((buffer.x - output.x) * (buffer.x - output.x) + variance.x * (device_temporal_frames - 1)) / device_temporal_frames;
      variance.y = ((buffer.y - output.y) * (buffer.y - output.y) + variance.y * (device_temporal_frames - 1)) / device_temporal_frames;
      variance.z = ((buffer.z - output.z) * (buffer.z - output.z) + variance.z * (device_temporal_frames - 1)) / device_temporal_frames;
      variance.w = ((buffer.w - output.w) * (buffer.w - output.w) + variance.w * (device_temporal_frames - 1)) / device_temporal_frames;
      __stcs((float4*) ((float*) device_frame_variance + offset), variance);
    }

    float4 firefly_rejection;
    firefly_rejection.x = 0.1f + output.x + deviation.x * 4.0f;
    firefly_rejection.y = 0.1f + output.y + deviation.y * 4.0f;
    firefly_rejection.z = 0.1f + output.z + deviation.z * 4.0f;
    firefly_rejection.w = 0.1f + output.w + deviation.w * 4.0f;

    firefly_rejection.x = fmaxf(0.0f, buffer.x - firefly_rejection.x);
    firefly_rejection.y = fmaxf(0.0f, buffer.y - firefly_rejection.y);
    firefly_rejection.z = fmaxf(0.0f, buffer.z - firefly_rejection.z);
    firefly_rejection.w = fmaxf(0.0f, buffer.w - firefly_rejection.w);

    bias_cache.x += firefly_rejection.x;
    bias_cache.y += firefly_rejection.y;
    bias_cache.z += firefly_rejection.z;
    bias_cache.w += firefly_rejection.w;

    buffer.x -= firefly_rejection.x;
    buffer.y -= firefly_rejection.y;
    buffer.z -= firefly_rejection.z;
    buffer.w -= firefly_rejection.w;

    float4 debias;
    debias.x = fmaxf(0.0f, fminf(bias_cache.x, output.x - deviation.x * 2.0f - buffer.x));
    debias.y = fmaxf(0.0f, fminf(bias_cache.y, output.y - deviation.y * 2.0f - buffer.y));
    debias.z = fmaxf(0.0f, fminf(bias_cache.z, output.z - deviation.z * 2.0f - buffer.z));
    debias.w = fmaxf(0.0f, fminf(bias_cache.w, output.w - deviation.w * 2.0f - buffer.w));

    buffer.x += debias.x;
    buffer.y += debias.y;
    buffer.z += debias.z;
    buffer.w += debias.w;

    bias_cache.x -= debias.x;
    bias_cache.y -= debias.y;
    bias_cache.z -= debias.z;
    bias_cache.w -= debias.w;
    __stcs((float4*) ((float*) device_frame_bias_cache + offset), bias_cache);

    output.x = (buffer.x + output.x * device_temporal_frames) / (device_temporal_frames + 1);
    output.y = (buffer.y + output.y * device_temporal_frames) / (device_temporal_frames + 1);
    output.z = (buffer.z + output.z * device_temporal_frames) / (device_temporal_frames + 1);
    output.w = (buffer.w + output.w * device_temporal_frames) / (device_temporal_frames + 1);
    __stcs((float4*) ((float*) device_frame_output + offset), output);
  }

  for (; offset < 3 * device_amount; offset++) {
    float buffer     = __ldcs((float*) device_frame_buffer + offset);
    float output     = __ldcs((float*) device_frame_output + offset);
    float variance   = __ldcs((float*) device_frame_variance + offset);
    float bias_cache = __ldcs(((float*) device_frame_bias_cache + offset));
    if (device_temporal_frames == 0) {
      bias_cache = 0.0f;
    }
    float deviation = sqrtf(fmaxf(eps, variance));
    variance        = ((buffer - output) * (buffer - output) + variance * device_temporal_frames) / (device_temporal_frames + 1);
    __stcs((float*) device_frame_variance + offset, variance);
    float firefly_rejection = 0.1f + output + deviation * 8.0f;
    firefly_rejection       = fmaxf(0.0f, buffer - firefly_rejection);
    bias_cache += firefly_rejection;
    buffer -= firefly_rejection;
    float debias = fmaxf(0.0f, fminf(bias_cache, output - deviation * 2.0f - buffer));
    buffer += debias;
    bias_cache -= debias;
    __stcs(((float*) device_frame_bias_cache + offset), bias_cache);
    output = (buffer + output * device_temporal_frames) / (device_temporal_frames + 1);
    __stcs((float*) device_frame_output + offset, output);
  }
}

__global__ void convert_RGBF_to_XRGB8(const int width, const int height, const RGBF* source) {
  unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;

  const int amount    = width * height;
  const float scale_x = 1.0f / width;
  const float scale_y = 1.0f / height;

  while (id < amount) {
    const int x = id % width;
    const int y = id / width;

    const float sx = x * scale_x;
    const float sy = y * scale_y;

    RGBF pixel = sample_pixel(source, sx, sy, device_width, device_height);

    pixel.r *= device_scene.camera.exposure;
    pixel.g *= device_scene.camera.exposure;
    pixel.b *= device_scene.camera.exposure;

    switch (device_scene.camera.tonemap) {
      case TONEMAP_NONE:
        break;
      case TONEMAP_ACES:
        pixel = aces_tonemap(pixel);
        break;
      case TONEMAP_REINHARD:
        pixel = reinhard_tonemap(pixel);
        break;
      case TONEMAP_UNCHARTED2:
        pixel = uncharted2_tonemap(pixel);
        break;
    }

    const float dither = (device_scene.camera.dithering) ? get_dithering(x, y) : 0.0f;

    pixel.r = fminf(255.9f, 0.5f + dither + 255.9f * linearRGB_to_SRGB(pixel.r));
    pixel.g = fminf(255.9f, 0.5f + dither + 255.9f * linearRGB_to_SRGB(pixel.g));
    pixel.b = fminf(255.9f, 0.5f + dither + 255.9f * linearRGB_to_SRGB(pixel.b));

    XRGB8 converted_pixel;
    converted_pixel.ignore = 0;
    converted_pixel.r      = (uint8_t) pixel.r;
    converted_pixel.g      = (uint8_t) pixel.g;
    converted_pixel.b      = (uint8_t) pixel.b;

    device_frame_8bit[x + y * width] = converted_pixel;

    id += blockDim.x * gridDim.x;
  }
}

#endif /* CU_KERNELS_H */
