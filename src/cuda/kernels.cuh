#ifndef CU_KERNELS_H
#define CU_KERNELS_H

#include <cuda_runtime_api.h>

#include "bvh.cuh"
#include "cloud.cuh"
#include "fog.cuh"
#include "geometry.cuh"
#include "ocean.cuh"
#include "purkinje.cuh"
#include "sky.cuh"
#include "sky_utils.cuh"
#include "temporal.cuh"
#include "toy.cuh"
#include "utils.cuh"

__device__ TraceTask get_starting_ray(TraceTask task) {
  vec3 default_ray;

  default_ray.x =
    device.scene.camera.focal_length * (-device.scene.camera.fov + device.emitter.step * (task.index.x + device.emitter.jitter.x));
  default_ray.y = device.scene.camera.focal_length * (device.emitter.vfov - device.emitter.step * (task.index.y + device.emitter.jitter.y));
  default_ray.z = -device.scene.camera.focal_length;

  const float alpha = (device.scene.camera.aperture_size == 0.0f) ? 0.0f : white_noise() * 2.0f * PI;
  const float beta  = (device.scene.camera.aperture_size == 0.0f) ? 0.0f : sqrtf(white_noise()) * device.scene.camera.aperture_size;

  vec3 point_on_aperture = get_vector(cosf(alpha) * beta, sinf(alpha) * beta, 0.0f);

  default_ray       = sub_vector(default_ray, point_on_aperture);
  point_on_aperture = rotate_vector_by_quaternion(point_on_aperture, device.emitter.camera_rotation);

  task.ray    = normalize_vector(rotate_vector_by_quaternion(default_ray, device.emitter.camera_rotation));
  task.origin = add_vector(device.scene.camera.pos, point_on_aperture);

  return task;
}

__global__ __launch_bounds__(THREADS_PER_BLOCK, 12) void generate_trace_tasks() {
  int offset       = 0;
  const int amount = device.width * device.height;

  for (int pixel = threadIdx.x + blockIdx.x * blockDim.x; pixel < amount; pixel += blockDim.x * gridDim.x) {
    TraceTask task;

    task.index.x = (uint16_t) (pixel % device.width);
    task.index.y = (uint16_t) (pixel / device.width);

    task = get_starting_ray(task);

    device.ptrs.light_records[pixel]  = get_RGBAhalf(1.0f, 1.0f, 1.0f, 0.0f);
    device.ptrs.bounce_records[pixel] = get_RGBAhalf(1.0f, 1.0f, 1.0f, 0.0f);
    device.ptrs.frame_buffer[pixel]   = get_RGBAhalf(0.0f, 0.0f, 0.0f, 0.0f);
    device.ptrs.state_buffer[pixel]   = 0;

    if (device.denoiser && !device.temporal_frames) {
      device.ptrs.albedo_buffer[pixel] = get_RGBAhalf(0.0f, 0.0f, 0.0f, 0.0f);
      device.ptrs.normal_buffer[pixel] = get_RGBAhalf(0.0f, 0.0f, 0.0f, 0.0f);
    }

    store_trace_task(device.ptrs.bounce_trace + get_task_address(offset++), task);
  }

  device.ptrs.light_trace_count[threadIdx.x + blockIdx.x * blockDim.x]  = 0;
  device.ptrs.bounce_trace_count[threadIdx.x + blockIdx.x * blockDim.x] = offset;
}

__global__ __launch_bounds__(THREADS_PER_BLOCK, 10) void balance_trace_tasks() {
  const int warp = (threadIdx.x + blockIdx.x * blockDim.x) >> 5;

  if (warp >= (THREADS_PER_BLOCK * BLOCKS_PER_GRID) >> 5)
    return;

  __shared__ uint16_t counts[THREADS_PER_BLOCK][32];
  uint16_t average = 0;

  for (int i = 0; i < 32; i += 4) {
    ushort4 c                  = __ldcs((ushort4*) (device.trace_count + 32 * warp + i));
    counts[threadIdx.x][i + 0] = c.x;
    counts[threadIdx.x][i + 1] = c.y;
    counts[threadIdx.x][i + 2] = c.z;
    counts[threadIdx.x][i + 3] = c.w;
    average += c.x;
    average += c.y;
    average += c.z;
    average += c.w;
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
          (float4*) (device.trace_tasks + get_task_address_of_thread(((warp & 0b11) << 5) + source_index, warp >> 2, source_count));
        float4* sink_ptr = (float4*) (device.trace_tasks + get_task_address_of_thread(((warp & 0b11) << 5) + i, warp >> 2, count));
        count++;

        __stwb(sink_ptr, __ldca(source_ptr));
        __stwb(sink_ptr + 1, __ldca(source_ptr + 1));
      }
      counts[threadIdx.x][i]            = count;
      counts[threadIdx.x][source_index] = source_count;
    }
  }

  for (int i = 0; i < 32; i += 4) {
    ushort4 vals = make_ushort4(counts[threadIdx.x][i], counts[threadIdx.x][i + 1], counts[threadIdx.x][i + 2], counts[threadIdx.x][i + 3]);
    __stcs((ushort4*) (device.trace_count + 32 * warp + i), vals);
  }
}

__global__ __launch_bounds__(THREADS_PER_BLOCK, 12) void preprocess_trace_tasks() {
  const int task_count = device.trace_count[threadIdx.x + blockIdx.x * blockDim.x];

  for (int i = 0; i < task_count; i++) {
    const int offset = get_task_address(i);
    TraceTask task   = load_trace_task(device.trace_tasks + offset);

    const uint32_t pixel = get_pixel_id(task.index.x, task.index.y);

    float depth     = device.scene.camera.far_clip_distance;
    uint32_t hit_id = SKY_HIT;

    uint32_t light_id;

    if (device.iteration_type == TYPE_LIGHT) {
      light_id = device.ptrs.light_sample_history[pixel];
    }

    if (is_first_ray() || (device.iteration_type == TYPE_LIGHT && light_id <= TRIANGLE_ID_LIMIT)) {
      uint32_t t_id;
      if (device.iteration_type == TYPE_LIGHT) {
        const TriangleLight tri_light = load_triangle_light(light_id);
        t_id                          = tri_light.triangle_id;
      }
      else {
        t_id = device.ptrs.trace_result_buffer[get_pixel_id(task.index.x, task.index.y)].hit_id;
      }

      if (t_id <= TRIANGLE_ID_LIMIT) {
        TraversalTriangle t = load_traversal_triangle(t_id);

        UV coords;
        const float dist = bvh_triangle_intersection_uv(t, task.origin, task.ray, coords);

        if (dist < depth) {
          const int alpha_result = bvh_triangle_intersection_alpha_test(t, t_id, coords);

          if (alpha_result != 2) {
            depth  = dist;
            hit_id = t_id;
          }
          else if (device.iteration_type == TYPE_LIGHT) {
            depth  = -1.0f;
            hit_id = REJECT_HIT;
          }
        }
      }
    }

    if (
      device.scene.toy.active && (!device.scene.toy.flashlight_mode || (device.iteration_type == TYPE_LIGHT && light_id == LIGHT_ID_TOY))) {
      const float toy_dist = get_toy_distance(task.origin, task.ray);

      if (toy_dist < depth) {
        depth = toy_dist;
        hit_id =
          (device.iteration_type == TYPE_LIGHT && light_id != LIGHT_ID_TOY && device.scene.toy.albedo.a == 1.0f) ? REJECT_HIT : TOY_HIT;
      }
    }

    if (device.scene.ocean.active && device.iteration_type != TYPE_LIGHT) {
      const float ocean_dist = ocean_far_distance(task.origin, task.ray);

      if (ocean_dist < depth) {
        depth  = ocean_dist;
        hit_id = OCEAN_HIT;
      }
    }

    float2 result;
    result.x = depth;
    result.y = __uint_as_float(hit_id);

    __stcs((float2*) (device.ptrs.trace_results + offset), result);
  }
}

__global__ __launch_bounds__(THREADS_PER_BLOCK, 12) void ocean_depth_trace_tasks() {
  const int task_count = device.trace_count[threadIdx.x + blockIdx.x * blockDim.x];

  for (int i = 0; i < task_count; i++) {
    const int offset    = get_task_address(i);
    TraceTask task      = load_trace_task(device.trace_tasks + offset);
    const float2 result = __ldcs((float2*) (device.ptrs.trace_results + offset));

    float depth     = result.x;
    uint32_t hit_id = __float_as_uint(result.y);

    const float far_distance   = ocean_far_distance(task.origin, task.ray);
    const float short_distance = ocean_short_distance(task.origin, task.ray);

    if (depth <= far_distance && depth > short_distance) {
      const float ocean_depth = ocean_intersection_distance(task.origin, task.ray, depth);

      if (ocean_depth < depth) {
        float2 result;
        result.x = ocean_depth;
        result.y = __uint_as_float(OCEAN_HIT);
        __stcs((float2*) (device.ptrs.trace_results + offset), result);
      }
    }
  }
}

__global__ __launch_bounds__(THREADS_PER_BLOCK, 6) void process_sky_inscattering_tasks() {
  const int task_count = device.trace_count[threadIdx.x + blockIdx.x * blockDim.x];

  for (int i = 0; i < task_count; i++) {
    const int offset    = get_task_address(i);
    TraceTask task      = load_trace_task(device.trace_tasks + offset);
    const float2 result = __ldcs((float2*) (device.ptrs.trace_results + offset));

    const float depth     = result.x;
    const uint32_t hit_id = __float_as_uint(result.y);

    if (hit_id == FOG_HIT || depth == device.scene.camera.far_clip_distance) {
      continue;
    }

    const int pixel = task.index.y * device.width + task.index.x;

    const vec3 sky_origin = world_to_sky_transform(task.origin);

    const float inscattering_limit = world_to_sky_scale(depth);

    RGBF record = RGBAhalf_to_RGBF(load_RGBAhalf(device.records + pixel));

    RGBF inscattering = sky_trace_inscattering(sky_origin, task.ray, inscattering_limit, record);

    RGBF color = RGBAhalf_to_RGBF(load_RGBAhalf(device.ptrs.frame_buffer + pixel));
    color      = add_color(color, inscattering);

    store_RGBAhalf(device.records + pixel, RGBF_to_RGBAhalf(record));
    store_RGBAhalf(device.ptrs.frame_buffer + pixel, RGBF_to_RGBAhalf(color));
  }
}

__global__ __launch_bounds__(THREADS_PER_BLOCK, 12) void postprocess_trace_tasks() {
  const int task_count         = device.trace_count[threadIdx.x + blockIdx.x * blockDim.x];
  uint16_t geometry_task_count = 0;
  uint16_t sky_task_count      = 0;
  uint16_t ocean_task_count    = 0;
  uint16_t toy_task_count      = 0;
  uint16_t fog_task_count      = 0;

  // count data
  for (int i = 0; i < task_count; i++) {
    const int offset      = get_task_address(i);
    const uint32_t hit_id = __ldca((uint32_t*) (device.ptrs.trace_results + offset) + 1);

    if (hit_id == SKY_HIT) {
      sky_task_count++;
    }
    else if (hit_id == OCEAN_HIT) {
      ocean_task_count++;
    }
    else if (hit_id == TOY_HIT) {
      toy_task_count++;
    }
    else if (hit_id == FOG_HIT) {
      fog_task_count++;
    }
    else if (hit_id <= TRIANGLE_ID_LIMIT) {
      geometry_task_count++;
    }
  }

  int geometry_offset = 0;
  int ocean_offset    = geometry_offset + geometry_task_count;
  int sky_offset      = ocean_offset + ocean_task_count;
  int toy_offset      = sky_offset + sky_task_count;
  int fog_offset      = toy_offset + toy_task_count;
  int rejects_offset  = fog_offset + fog_task_count;
  int k               = 0;

  device.ptrs.task_offsets[(threadIdx.x + blockIdx.x * blockDim.x) * 5 + 0] = geometry_offset;
  device.ptrs.task_offsets[(threadIdx.x + blockIdx.x * blockDim.x) * 5 + 1] = ocean_offset;
  device.ptrs.task_offsets[(threadIdx.x + blockIdx.x * blockDim.x) * 5 + 2] = sky_offset;
  device.ptrs.task_offsets[(threadIdx.x + blockIdx.x * blockDim.x) * 5 + 3] = toy_offset;
  device.ptrs.task_offsets[(threadIdx.x + blockIdx.x * blockDim.x) * 5 + 4] = fog_offset;

  const int num_tasks = rejects_offset;

  // order data
  while (k < task_count) {
    const int offset      = get_task_address(k);
    const uint32_t hit_id = __ldca((uint32_t*) (device.ptrs.trace_results + offset) + 1);

    int index;
    int needs_swapping;

    if (hit_id <= TRIANGLE_ID_LIMIT) {
      index          = geometry_offset++;
      needs_swapping = (k >= geometry_task_count);
    }
    else if (hit_id == OCEAN_HIT) {
      index          = ocean_offset++;
      needs_swapping = (k < geometry_task_count) || (k >= geometry_task_count + ocean_task_count);
    }
    else if (hit_id == SKY_HIT) {
      index          = sky_offset++;
      needs_swapping = (k < geometry_task_count + ocean_task_count) || (k >= geometry_task_count + ocean_task_count + sky_task_count);
    }
    else if (hit_id == TOY_HIT) {
      index          = toy_offset++;
      needs_swapping = (k < geometry_task_count + ocean_task_count + sky_task_count)
                       || (k >= geometry_task_count + ocean_task_count + sky_task_count + toy_task_count);
    }
    else if (hit_id == FOG_HIT) {
      index          = fog_offset++;
      needs_swapping = (k < geometry_task_count + ocean_task_count + sky_task_count + toy_task_count)
                       || (k >= geometry_task_count + ocean_task_count + sky_task_count + toy_task_count + fog_task_count);
    }
    else {
      index          = rejects_offset++;
      needs_swapping = (k < geometry_task_count + ocean_task_count + sky_task_count + toy_task_count + fog_task_count);
    }

    if (needs_swapping) {
      swap_trace_data(k, index);
    }
    else {
      k++;
    }
  }

  if (device.iteration_type == TYPE_LIGHT) {
    for (int i = 0; i < task_count; i++) {
      const int offset     = get_task_address(i);
      TraceTask task       = load_trace_task(device.trace_tasks + offset);
      const uint32_t pixel = get_pixel_id(task.index.x, task.index.y);
      device.ptrs.state_buffer[pixel] &= ~STATE_LIGHT_OCCUPIED;
    }
  }

  // process data
  for (int i = 0; i < num_tasks; i++) {
    const int offset    = get_task_address(i);
    TraceTask task      = load_trace_task(device.trace_tasks + offset);
    const float2 result = __ldcs((float2*) (device.ptrs.trace_results + offset));

    const float depth     = result.x;
    const uint32_t hit_id = __float_as_uint(result.y);
    const uint32_t pixel  = get_pixel_id(task.index.x, task.index.y);

    if (is_first_ray()) {
      device.ptrs.raydir_buffer[pixel] = task.ray;

      TraceResult trace_result;
      trace_result.depth  = depth;
      trace_result.hit_id = hit_id;

      device.ptrs.trace_result_buffer[pixel] = trace_result;
    }

    if (hit_id != SKY_HIT)
      task.origin = add_vector(task.origin, scale_vector(task.ray, depth));

    if (device.iteration_type != TYPE_LIGHT) {
      if (device.light_resampling) {
        LightEvalData light_data;
        light_data.position = task.origin;
        light_data.flags    = 0;

        switch (hit_id) {
          case SKY_HIT:
          case FOG_HIT:
            break;
          case OCEAN_HIT:
          case TOY_HIT:
          default:
            light_data.flags = 1;
            break;
        }

        store_light_eval_data(light_data, pixel);
      }
      else {
        if (hit_id == OCEAN_HIT || hit_id == TOY_HIT || hit_id <= TRIANGLE_ID_LIMIT) {
          const vec3 sky_pos    = world_to_sky_transform(task.origin);
          const int sun_visible = !sph_ray_hit_p0(normalize_vector(sub_vector(device.sun_pos, sky_pos)), sky_pos, SKY_EARTH_RADIUS);

          LightSample selected;
          selected.id = LIGHT_ID_NONE;
          selected.M  = 0;

          if (sun_visible) {
            selected.id = LIGHT_ID_SUN;
            selected.M  = 1;
          }

          selected.solid_angle = brdf_light_sample_solid_angle(selected, task.origin);
          selected.weight      = brdf_light_sample_target_weight(selected);

          store_light_sample(device.ptrs.light_samples, selected, pixel);
        }
      }
    }

    float4* ptr = (float4*) (device.trace_tasks + offset);
    float4 data0;
    float4 data1;

    data0.x = __uint_as_float(((uint32_t) task.index.x & 0xffff) | ((uint32_t) task.index.y << 16));
    data0.y = task.origin.x;
    data0.z = task.origin.y;
    data0.w = task.origin.z;

    __stcs(ptr, data0);

    if (hit_id == TOY_HIT || hit_id == SKY_HIT) {
      data1.x = task.ray.x;
      data1.y = task.ray.y;
      data1.z = task.ray.z;
    }
    else {
      data1.x = asinf(task.ray.y);
      data1.y = atan2f(task.ray.z, task.ray.x);

      if (hit_id == OCEAN_HIT || hit_id == FOG_HIT) {
        data1.z = depth;
      }
      else {
        data1.z = __uint_as_float(hit_id);
      }
    }

    __stcs(ptr + 1, data1);
  }

  device.ptrs.task_counts[(threadIdx.x + blockIdx.x * blockDim.x) * 6 + 0] = geometry_task_count;
  device.ptrs.task_counts[(threadIdx.x + blockIdx.x * blockDim.x) * 6 + 1] = ocean_task_count;
  device.ptrs.task_counts[(threadIdx.x + blockIdx.x * blockDim.x) * 6 + 2] = sky_task_count;
  device.ptrs.task_counts[(threadIdx.x + blockIdx.x * blockDim.x) * 6 + 3] = toy_task_count;
  device.ptrs.task_counts[(threadIdx.x + blockIdx.x * blockDim.x) * 6 + 4] = fog_task_count;
  device.ptrs.task_counts[(threadIdx.x + blockIdx.x * blockDim.x) * 6 + 5] = num_tasks;
  device.trace_count[threadIdx.x + blockIdx.x * blockDim.x]                = 0;
}

__global__ void convert_RGBhalf_to_XRGB8(const RGBAhalf* source, XRGB8* dest, const int width, const int height, const int ld) {
  unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;

  const int amount    = width * height;
  const float scale_x = 1.0f / (width - 1);
  const float scale_y = 1.0f / (height - 1);

  while (id < amount) {
    const int x = id % width;
    const int y = id / width;

    const float sx = x * scale_x;
    const float sy = y * scale_y;

    RGBF pixel = RGBAhalf_to_RGBF(sample_pixel(source, sx, sy, device.output_width, device.output_height));

    if (device.scene.camera.purkinje) {
      pixel = purkinje_shift(pixel);
    }

    pixel.r *= device.scene.camera.exposure;
    pixel.g *= device.scene.camera.exposure;
    pixel.b *= device.scene.camera.exposure;

    switch (device.scene.camera.tonemap) {
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

    switch (device.scene.camera.filter) {
      case FILTER_NONE:
        break;
      case FILTER_GRAY:
        pixel = filter_gray(pixel);
        break;
      case FILTER_SEPIA:
        pixel = filter_sepia(pixel);
        break;
      case FILTER_GAMEBOY:
        pixel = filter_gameboy(pixel, x, y);
        break;
      case FILTER_2BITGRAY:
        pixel = filter_2bitgray(pixel, x, y);
        break;
      case FILTER_CRT:
        pixel = filter_crt(pixel, x, y);
        break;
      case FILTER_BLACKWHITE:
        pixel = filter_blackwhite(pixel, x, y);
        break;
    }

    const float dither = (device.scene.camera.dithering) ? get_dithering(x, y) : 0.0f;

    pixel.r = fminf(255.9f, 0.5f - dither + 255.9f * linearRGB_to_SRGB(pixel.r));
    pixel.g = fminf(255.9f, 0.5f - dither + 255.9f * linearRGB_to_SRGB(pixel.g));
    pixel.b = fminf(255.9f, 0.5f - dither + 255.9f * linearRGB_to_SRGB(pixel.b));

    XRGB8 converted_pixel;
    converted_pixel.ignore = 0;
    converted_pixel.r      = (uint8_t) pixel.r;
    converted_pixel.g      = (uint8_t) pixel.g;
    converted_pixel.b      = (uint8_t) pixel.b;

    dest[x + y * ld] = converted_pixel;

    id += blockDim.x * gridDim.x;
  }
}

#endif /* CU_KERNELS_H */
