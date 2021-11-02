#ifndef CU_KERNELS_H
#define CU_KERNELS_H

#include <cuda_runtime_api.h>

#include "bvh.cuh"
#include "fog.cuh"
#include "geometry.cuh"
#include "ocean.cuh"
#include "sky.cuh"
#include "toy.cuh"
#include "utils.cuh"

__device__ TraceTask get_starting_ray(TraceTask task) {
  vec3 default_ray;

  default_ray.x = device_scene.camera.focal_length * (-device_scene.camera.fov + device_step * (task.index.x + device_jitter.x));
  default_ray.y = device_scene.camera.focal_length * (device_vfov - device_step * (task.index.y + device_jitter.y));
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

    task.state = (device_max_ray_depth << 16) | (device_temporal_frames & RANDOM_INDEX);

    task = get_starting_ray(task);

    device_light_records[pixel]  = get_color(1.0f, 1.0f, 1.0f);
    device_bounce_records[pixel] = get_color(1.0f, 1.0f, 1.0f);
    device_frame_buffer[pixel]   = get_color(0.0f, 0.0f, 0.0f);

    if (device_denoiser)
      device_albedo_buffer[pixel] = get_color(0.0f, 0.0f, 0.0f);

    store_trace_task(device_bounce_trace + get_task_address(offset++), task);
  }

  device_light_trace_count[threadIdx.x + blockIdx.x * blockDim.x]  = 0;
  device_bounce_trace_count[threadIdx.x + blockIdx.x * blockDim.x] = offset;
}

__global__ __launch_bounds__(THREADS_PER_BLOCK, 10) void balance_trace_tasks() {
  const int warp = (threadIdx.x + blockIdx.x * blockDim.x) >> 5;

  if (warp >= (THREADS_PER_BLOCK * BLOCKS_PER_GRID) >> 5)
    return;

  __shared__ uint16_t counts[THREADS_PER_BLOCK][32];
  uint16_t average = 0;

  for (int i = 0; i < 32; i++) {
    const uint16_t c       = device_trace_count[32 * warp + i];
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
          (float4*) (device_trace_tasks + get_task_address_of_thread(((warp & 0b11) << 5) + source_index, warp >> 2, source_count));
        float4* sink_ptr = (float4*) (device_trace_tasks + get_task_address_of_thread(((warp & 0b11) << 5) + i, warp >> 2, count));
        count++;

        __stwb(sink_ptr, __ldca(source_ptr));
        __stwb(sink_ptr + 1, __ldca(source_ptr + 1));
      }
      counts[threadIdx.x][i]            = count;
      counts[threadIdx.x][source_index] = source_count;
    }
  }

  for (int i = 0; i < 32; i++) {
    device_trace_count[32 * warp + i] = counts[threadIdx.x][i];
  }
}

__global__ __launch_bounds__(THREADS_PER_BLOCK, 12) void preprocess_trace_tasks() {
  const int task_count = device_trace_count[threadIdx.x + blockIdx.x * blockDim.x];

  for (int i = 0; i < task_count; i++) {
    const int offset = get_task_address(i);
    TraceTask task   = load_trace_task(device_trace_tasks + offset);

    float depth     = device_scene.camera.far_clip_distance;
    uint32_t hit_id = SKY_HIT;

    if (device_scene.fog.active && is_first_ray(task.state)) {
      const float fog_dist = get_intersection_fog(task.origin, task.ray, sample_blue_noise(task.index.x, task.index.y, 256, 169));

      if (fog_dist < depth) {
        depth  = fog_dist;
        hit_id = FOG_HIT;
      }
    }

    if (device_scene.toy.active) {
      const float toy_dist = get_toy_distance(task.origin, task.ray);

      if (toy_dist < depth) {
        depth  = toy_dist;
        hit_id = TOY_HIT;
      }
    }

    if (device_scene.ocean.active) {
      const float ocean_dist = get_intersection_ocean(task.origin, task.ray, depth);

      if (ocean_dist < depth) {
        depth  = ocean_dist;
        hit_id = OCEAN_HIT;
      }
    }

    float2 result;
    result.x = depth;
    result.y = uint_as_float(hit_id);

    __stcs((float2*) (device_trace_results + offset), result);
  }
}

__global__ __launch_bounds__(THREADS_PER_BLOCK, 12) void postprocess_trace_tasks() {
  const int task_count         = device_trace_count[threadIdx.x + blockIdx.x * blockDim.x];
  uint16_t geometry_task_count = 0;
  uint16_t sky_task_count      = 0;
  uint16_t ocean_task_count    = 0;
  uint16_t toy_task_count      = 0;
  uint16_t fog_task_count      = 0;

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
    else if (hit_id == FOG_HIT) {
      fog_task_count++;
    }
    else {
      geometry_task_count++;
    }
  }

  int geometry_offset = 0;
  int ocean_offset    = geometry_offset + geometry_task_count;
  int sky_offset      = ocean_offset + ocean_task_count;
  int toy_offset      = sky_offset + sky_task_count;
  int fog_offset      = toy_offset + toy_task_count;
  int k               = 0;

  device_task_offsets[(threadIdx.x + blockIdx.x * blockDim.x) * 5 + 0] = geometry_offset;
  device_task_offsets[(threadIdx.x + blockIdx.x * blockDim.x) * 5 + 1] = ocean_offset;
  device_task_offsets[(threadIdx.x + blockIdx.x * blockDim.x) * 5 + 2] = sky_offset;
  device_task_offsets[(threadIdx.x + blockIdx.x * blockDim.x) * 5 + 3] = toy_offset;
  device_task_offsets[(threadIdx.x + blockIdx.x * blockDim.x) * 5 + 4] = fog_offset;

  // order data
  while (k < task_count) {
    const int offset      = get_task_address(k);
    const uint32_t hit_id = __ldca((uint32_t*) (device_trace_results + offset) + 1);

    int index;
    int needs_swapping;

    if (hit_id == OCEAN_HIT) {
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
      needs_swapping = (k < geometry_task_count + ocean_task_count + sky_task_count + toy_task_count);
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
  fog_task_count      = 0;

  // process data
  for (int i = 0; i < task_count; i++) {
    const int offset    = get_task_address(i);
    TraceTask task      = load_trace_task(device_trace_tasks + offset);
    const float2 result = __ldcs((float2*) (device_trace_results + offset));

    const float depth     = result.x;
    const uint32_t hit_id = float_as_uint(result.y);

    if (device_scene.fog.active) {
      float t      = get_fog_depth(task.origin.y, task.ray.y, depth);
      float weight = expf(-t * device_scene.fog.absorption * 0.001f);

      RGBF record = device_records[task.index.x + task.index.y * device_width];

      record = scale_color(record, weight);

      device_records[task.index.x + task.index.y * device_width] = record;
    }

    float4* ptr = (float4*) (device_trace_tasks + offset);
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
        else if (hit_id == FOG_HIT) {
          fog_task_count++;
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

  device_task_counts[(threadIdx.x + blockIdx.x * blockDim.x) * 5 + 0] = geometry_task_count;
  device_task_counts[(threadIdx.x + blockIdx.x * blockDim.x) * 5 + 1] = ocean_task_count;
  device_task_counts[(threadIdx.x + blockIdx.x * blockDim.x) * 5 + 2] = sky_task_count;
  device_task_counts[(threadIdx.x + blockIdx.x * blockDim.x) * 5 + 3] = toy_task_count;
  device_task_counts[(threadIdx.x + blockIdx.x * blockDim.x) * 5 + 4] = fog_task_count;
  device_trace_count[threadIdx.x + blockIdx.x * blockDim.x]           = 0;
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
  const float scale_x = 1.0f / (width - 1);
  const float scale_y = 1.0f / (height - 1);

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

    switch (device_scene.camera.filter) {
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

    const float dither = (device_scene.camera.dithering) ? get_dithering(x, y) : 0.0f;

    pixel.r = fminf(255.9f, 0.5f - dither + 255.9f * linearRGB_to_SRGB(pixel.r));
    pixel.g = fminf(255.9f, 0.5f - dither + 255.9f * linearRGB_to_SRGB(pixel.g));
    pixel.b = fminf(255.9f, 0.5f - dither + 255.9f * linearRGB_to_SRGB(pixel.b));

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
