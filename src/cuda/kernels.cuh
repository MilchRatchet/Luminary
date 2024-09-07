#ifndef CU_KERNELS_H
#define CU_KERNELS_H

#include <cuda_runtime_api.h>

#include "bsdf_utils.cuh"
#include "bvh.cuh"
#include "camera.cuh"
#include "cloud.cuh"
#include "ior_stack.cuh"
#include "purkinje.cuh"
#include "sky.cuh"
#include "sky_utils.cuh"
#include "state.cuh"
#include "temporal.cuh"
#include "tonemap.cuh"
#include "utils.cuh"
#include "volume.cuh"

//
// Undersampling pattern:
// The lowest undersampling pass is used 4 times
// while the others are done 3 times.
// Once all undersampling passes have been performed,
// each pixel has exactly one sample and we continue
// without undersampling. Example for 3 passes with
// a 8x8 resolution.
//
//  3 | 1 | 2 | 1 | 3 | 1 | 2 | 1
// ---+---+---+---+---+---+---+---
//  1 | 1 | 1 | 1 | 1 | 1 | 1 | 1
// ---+---+---+---+---+---+---+---
//  2 | 1 | 2 | 1 | 2 | 1 | 2 | 1
// ---+---+---+---+---+---+---+---
//  1 | 1 | 1 | 1 | 1 | 1 | 1 | 1
// ---+---+---+---+---+---+---+---
//  3 | 1 | 2 | 1 | 3 | 1 | 2 | 1
// ---+---+---+---+---+---+---+---
//  1 | 1 | 1 | 1 | 1 | 1 | 1 | 1
// ---+---+---+---+---+---+---+---
//  2 | 1 | 2 | 1 | 2 | 1 | 2 | 1
// ---+---+---+---+---+---+---+---
//  1 | 1 | 1 | 1 | 1 | 1 | 1 | 1
//

// TODO: Turns this into a function and call this in the trace kernel if primary ray.
LUMINARY_KERNEL void generate_trace_tasks() {
  uint32_t task_count = 0;

  const uint32_t undersampling_scale = 1 << device.undersampling;

  uint32_t amount = device.internal_width * device.internal_height;
  amount          = amount >> (2 * device.undersampling);

  const uint32_t undersampling_width  = device.internal_width >> device.undersampling;
  const uint32_t undersampling_height = device.internal_height >> device.undersampling;

  const uint32_t undersampling_index = roundf(device.temporal_frames * undersampling_scale * undersampling_scale);

  for (uint32_t undersampling_pixel = THREAD_ID; undersampling_pixel < amount; undersampling_pixel += blockDim.x * gridDim.x) {
    uint16_t undersampling_y = (uint16_t) (undersampling_pixel / undersampling_width);
    uint16_t undersampling_x = (uint16_t) (undersampling_pixel - undersampling_y * undersampling_width);

    if (undersampling_scale > 1) {
      undersampling_x *= undersampling_scale;
      undersampling_y *= undersampling_scale;

      undersampling_x += (undersampling_scale >> 1) * ((undersampling_index & 0b01) ? 1.0f : 0.0f);
      undersampling_y += (undersampling_scale >> 1) * ((undersampling_index & 0b10) ? 1.0f : 0.0f);
    }

    TraceTask task;
    task.index.x = undersampling_x;
    task.index.y = undersampling_y;

    task = camera_get_ray(task);

    const uint32_t pixel = get_pixel_id(task.index);

    device.ptrs.records[pixel]      = get_color(1.0f, 1.0f, 1.0f);
    device.ptrs.state_buffer[pixel] = STATE_FLAG_DELTA_PATH | STATE_FLAG_CAMERA_DIRECTION;

    if ((device.denoiser || device.aov_mode) && device.temporal_frames == 0.0f) {
      device.ptrs.albedo_buffer[pixel] = get_color(0.0f, 0.0f, 0.0f);
      device.ptrs.normal_buffer[pixel] = get_color(0.0f, 0.0f, 0.0f);
    }

    device.ptrs.frame_direct_buffer[pixel]   = get_color(0.0f, 0.0f, 0.0f);
    device.ptrs.frame_indirect_buffer[pixel] = get_color(0.0f, 0.0f, 0.0f);

    const float ambient_ior = bsdf_refraction_index_ambient(task.origin, task.ray);
    ior_stack_interact(ambient_ior, pixel, IOR_STACK_METHOD_RESET);

    store_trace_task(device.ptrs.trace_tasks + get_task_address(task_count++), task);
  }

  device.ptrs.trace_counts[THREAD_ID] = task_count;
}

LUMINARY_KERNEL void balance_trace_tasks() {
  const int warp = THREAD_ID;

  if (warp >= (THREADS_PER_BLOCK * BLOCKS_PER_GRID) >> 5)
    return;

  __shared__ uint16_t counts[THREADS_PER_BLOCK][32];
  uint32_t sum = 0;

  for (int i = 0; i < 32; i += 4) {
    ushort4 c                  = __ldcs((ushort4*) (device.ptrs.trace_counts + 32 * warp + i));
    counts[threadIdx.x][i + 0] = c.x;
    counts[threadIdx.x][i + 1] = c.y;
    counts[threadIdx.x][i + 2] = c.z;
    counts[threadIdx.x][i + 3] = c.w;
    sum += c.x;
    sum += c.y;
    sum += c.z;
    sum += c.w;
  }

  const uint16_t average = 1 + (sum >> 5);

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

      static_assert(THREADS_PER_BLOCK == 128, "The following code assumes that we have 4 warps per block.");
      const int thread_id_base = ((warp & 0b11) << 5);
      const int block_id       = warp >> 2;

      for (int j = 0; j < swaps; j++) {
        TraceTask* source_ptr =
          device.ptrs.trace_tasks + get_task_address_of_thread(thread_id_base + source_index, block_id, source_count - 1);
        TraceTask* sink_ptr = device.ptrs.trace_tasks + get_task_address_of_thread(thread_id_base + i, block_id, count);

        __stwb((float4*) sink_ptr, __ldca((float4*) source_ptr));
        __stwb((float4*) (sink_ptr) + 1, __ldca((float4*) (source_ptr) + 1));

        sink_ptr++;
        count++;

        source_ptr--;
        source_count--;
      }
      counts[threadIdx.x][i]            = count;
      counts[threadIdx.x][source_index] = source_count;
    }
  }

  for (int i = 0; i < 32; i += 4) {
    ushort4 vals = make_ushort4(counts[threadIdx.x][i], counts[threadIdx.x][i + 1], counts[threadIdx.x][i + 2], counts[threadIdx.x][i + 3]);
    __stcs((ushort4*) (device.ptrs.trace_counts + 32 * warp + i), vals);
  }
}

LUMINARY_KERNEL void process_sky_inscattering_tasks() {
  const int task_count = device.ptrs.trace_counts[THREAD_ID];

  for (int i = 0; i < task_count; i++) {
    const int offset    = get_task_address(i);
    TraceTask task      = load_trace_task(device.ptrs.trace_tasks + offset);
    const float2 result = __ldcs((float2*) (device.ptrs.trace_results + offset));

    const float depth     = result.x;
    const uint32_t hit_id = __float_as_uint(result.y);

    if (hit_id == HIT_TYPE_SKY)
      continue;

    const uint32_t pixel = get_pixel_id(task.index);

    const vec3 sky_origin = world_to_sky_transform(task.origin);

    const float inscattering_limit = world_to_sky_scale(depth);

    RGBF record = load_RGBF(device.ptrs.records + pixel);

    const RGBF inscattering = sky_trace_inscattering(sky_origin, task.ray, inscattering_limit, record, task.index);

    store_RGBF(device.ptrs.records + pixel, record);
    write_beauty_buffer(inscattering, pixel);
  }
}

LUMINARY_KERNEL void postprocess_trace_tasks() {
  const int task_count         = device.ptrs.trace_counts[THREAD_ID];
  uint16_t geometry_task_count = 0;
  uint16_t particle_task_count = 0;
  uint16_t sky_task_count      = 0;
  uint16_t ocean_task_count    = 0;
  uint16_t toy_task_count      = 0;
  uint16_t volume_task_count   = 0;

  // count data
  for (int i = 0; i < task_count; i++) {
    const int offset      = get_task_address(i);
    const uint32_t hit_id = __ldca(&device.ptrs.trace_results[offset].hit_id);

    if (hit_id == HIT_TYPE_SKY) {
      sky_task_count++;
    }
    else if (hit_id == HIT_TYPE_OCEAN) {
      ocean_task_count++;
    }
    else if (hit_id == HIT_TYPE_TOY) {
      toy_task_count++;
    }
    else if (VOLUME_HIT_CHECK(hit_id)) {
      volume_task_count++;
    }
    else if (hit_id <= HIT_TYPE_PARTICLE_MAX && hit_id >= HIT_TYPE_PARTICLE_MIN) {
      particle_task_count++;
    }
    else if (hit_id <= HIT_TYPE_TRIANGLE_ID_LIMIT) {
      geometry_task_count++;
    }
  }

  int geometry_offset = 0;
  int toy_offset      = geometry_offset + geometry_task_count;
  int ocean_offset    = toy_offset + toy_task_count;
  int volume_offset   = ocean_offset + ocean_task_count;
  int particle_offset = volume_offset + volume_task_count;
  int sky_offset      = particle_offset + particle_task_count;
  int rejects_offset  = sky_offset + sky_task_count;
  int k               = 0;

  device.ptrs.task_offsets[THREAD_ID * TASK_ADDRESS_OFFSET_STRIDE + TASK_ADDRESS_OFFSET_GEOMETRY] = geometry_offset;
  device.ptrs.task_offsets[THREAD_ID * TASK_ADDRESS_OFFSET_STRIDE + TASK_ADDRESS_OFFSET_VOLUME]   = volume_offset;
  device.ptrs.task_offsets[THREAD_ID * TASK_ADDRESS_OFFSET_STRIDE + TASK_ADDRESS_OFFSET_PARTICLE] = particle_offset;
  device.ptrs.task_offsets[THREAD_ID * TASK_ADDRESS_OFFSET_STRIDE + TASK_ADDRESS_OFFSET_SKY]      = sky_offset;

  const int num_tasks               = rejects_offset;
  const int initial_geometry_offset = 0;
  const int initial_toy_offset      = initial_geometry_offset + geometry_task_count;
  const int initial_ocean_offset    = initial_toy_offset + toy_task_count;
  const int initial_volume_offset   = initial_ocean_offset + ocean_task_count;
  const int initial_particle_offset = initial_volume_offset + volume_task_count;
  const int initial_sky_offset      = initial_particle_offset + particle_task_count;
  const int initial_rejects_offset  = initial_sky_offset + sky_task_count;

  // order data
  while (k < task_count) {
    const int offset      = get_task_address(k);
    const uint32_t hit_id = __ldca(&device.ptrs.trace_results[offset].hit_id);

    int index;
    int needs_swapping;

    if (hit_id <= HIT_TYPE_TRIANGLE_ID_LIMIT) {
      index          = geometry_offset;
      needs_swapping = (k < initial_geometry_offset) || (k >= geometry_task_count + initial_geometry_offset);
      if (needs_swapping || k >= geometry_offset) {
        geometry_offset++;
      }
    }
    else if (hit_id == HIT_TYPE_TOY) {
      index          = toy_offset;
      needs_swapping = (k < initial_toy_offset) || (k >= toy_task_count + initial_toy_offset);
      if (needs_swapping || k >= toy_offset) {
        toy_offset++;
      }
    }
    else if (hit_id == HIT_TYPE_OCEAN) {
      index          = ocean_offset;
      needs_swapping = (k < initial_ocean_offset) || (k >= ocean_task_count + initial_ocean_offset);
      if (needs_swapping || k >= ocean_offset) {
        ocean_offset++;
      }
    }
    else if (VOLUME_HIT_CHECK(hit_id)) {
      index          = volume_offset;
      needs_swapping = (k < initial_volume_offset) || (k >= volume_task_count + initial_volume_offset);
      if (needs_swapping || k >= volume_offset) {
        volume_offset++;
      }
    }
    else if (hit_id <= HIT_TYPE_PARTICLE_MAX) {
      index          = particle_offset;
      needs_swapping = (k < initial_particle_offset) || (k >= particle_task_count + initial_particle_offset);
      if (needs_swapping || k >= particle_offset) {
        particle_offset++;
      }
    }
    else if (hit_id == HIT_TYPE_SKY) {
      index          = sky_offset;
      needs_swapping = (k < initial_sky_offset) || (k >= sky_task_count + initial_sky_offset);
      if (needs_swapping || k >= sky_offset) {
        sky_offset++;
      }
    }
    else {
      index          = rejects_offset;
      needs_swapping = (k < initial_rejects_offset);
      if (needs_swapping || k >= rejects_offset) {
        rejects_offset++;
      }
    }

    if (needs_swapping) {
      swap_trace_data(k, index);
    }
    else {
      k++;
    }
  }

  // process data
  for (int i = 0; i < num_tasks; i++) {
    const int offset    = get_task_address(i);
    TraceTask task      = load_trace_task(device.ptrs.trace_tasks + offset);
    const float2 result = __ldcs((float2*) (device.ptrs.trace_results + offset));

    const float depth     = result.x;
    const uint32_t hit_id = __float_as_uint(result.y);
    const uint32_t pixel  = get_pixel_id(task.index);

    if (IS_PRIMARY_RAY) {
      TraceResult trace_result;
      trace_result.depth  = depth;
      trace_result.hit_id = hit_id;

      device.ptrs.trace_results_history[pixel] = trace_result;
    }

    if (hit_id != HIT_TYPE_SKY)
      task.origin = add_vector(task.origin, scale_vector(task.ray, depth));

    float4* ptr = (float4*) (device.ptrs.trace_tasks + offset);
    float4 data0;
    float4 data1;

    data0.x = __uint_as_float(hit_id);
    data0.y = __uint_as_float(((uint32_t) task.index.x & 0xffff) | ((uint32_t) task.index.y << 16));
    data0.z = task.origin.x;
    data0.w = task.origin.y;

    __stcs(ptr, data0);

    data1.x = task.origin.z;
    data1.y = task.ray.x;
    data1.z = task.ray.y;
    data1.w = task.ray.z;

    __stcs(ptr + 1, data1);
  }

  const uint16_t geometry_kernel_task_count = geometry_task_count + toy_task_count + ocean_task_count;

  device.ptrs.task_counts[THREAD_ID * TASK_ADDRESS_COUNT_STRIDE + TASK_ADDRESS_OFFSET_GEOMETRY]   = geometry_kernel_task_count;
  device.ptrs.task_counts[THREAD_ID * TASK_ADDRESS_COUNT_STRIDE + TASK_ADDRESS_OFFSET_VOLUME]     = volume_task_count;
  device.ptrs.task_counts[THREAD_ID * TASK_ADDRESS_COUNT_STRIDE + TASK_ADDRESS_OFFSET_PARTICLE]   = particle_task_count;
  device.ptrs.task_counts[THREAD_ID * TASK_ADDRESS_COUNT_STRIDE + TASK_ADDRESS_OFFSET_SKY]        = sky_task_count;
  device.ptrs.task_counts[THREAD_ID * TASK_ADDRESS_COUNT_STRIDE + TASK_ADDRESS_OFFSET_TOTALCOUNT] = num_tasks;

  device.ptrs.trace_counts[THREAD_ID] = 0;
}

LUMINARY_KERNEL void generate_final_image(const RGBF* src) {
  const uint32_t undersampling       = max(device.undersampling, 1);
  const uint32_t undersampling_scale = 1 << undersampling;

  uint32_t amount = device.internal_width * device.internal_height;
  amount          = amount >> (2 * undersampling);

  const uint32_t undersampling_width  = device.internal_width >> undersampling;
  const uint32_t undersampling_height = device.internal_height >> undersampling;

  const float color_scale = 1.0f / (undersampling_scale * undersampling_scale * __saturatef(device.temporal_frames + temporal_increment()));

  for (uint32_t undersampling_pixel = THREAD_ID; undersampling_pixel < amount; undersampling_pixel += blockDim.x * gridDim.x) {
    const uint16_t y = (uint16_t) (undersampling_pixel / undersampling_width);
    const uint16_t x = (uint16_t) (undersampling_pixel - y * undersampling_width);

    const uint16_t source_x = x * undersampling_scale;
    const uint16_t source_y = y * undersampling_scale;

    RGBF accumulated_color = get_color(0.0f, 0.0f, 0.0f);

    for (uint32_t yi = 0; yi < undersampling_scale; yi++) {
      for (uint32_t xi = 0; xi < undersampling_scale; xi++) {
        const uint32_t index = (source_x + xi) + (source_y + yi) * device.internal_width;

        RGBF pixel = load_RGBF(src + index);
        pixel      = tonemap_apply(pixel);

        accumulated_color = add_color(accumulated_color, pixel);
      }
    }

    accumulated_color = scale_color(accumulated_color, color_scale);

    const uint16_t dst_scale = 1 << (undersampling - 1);

    const uint16_t dst_x = x * dst_scale;
    const uint16_t dst_y = y * dst_scale;

    for (uint32_t yi = 0; yi < dst_scale; yi++) {
      for (uint32_t xi = 0; xi < dst_scale; xi++) {
        const uint32_t index = (dst_x + xi) + (dst_y + yi) * device.width;

        store_RGBF(device.ptrs.frame_final + index, accumulated_color);
      }
    }
  }
}

LUMINARY_KERNEL void convert_RGBF_to_XRGB8(
  XRGB8* dest, const int width, const int height, const int ld, const OutputVariable output_variable) {
  unsigned int id = THREAD_ID;

  const int amount    = width * height;
  const float scale_x = 1.0f / (width - 1);
  const float scale_y = 1.0f / (height - 1);

  const int src_width  = (output_variable == OUTPUT_VARIABLE_BEAUTY) ? device.output_width : device.width;
  const int src_height = (output_variable == OUTPUT_VARIABLE_BEAUTY) ? device.output_height : device.height;

  while (id < amount) {
    const int y = id / width;
    const int x = id - y * width;

    const float sx = x * scale_x;
    const float sy = y * scale_y;

    RGBF pixel = sample_pixel_clamp(device.ptrs.frame_final, sx, sy, src_width, src_height);

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

    const float dither = (device.scene.camera.dithering) ? random_dither_mask(x, y) : 0.5f;

    pixel.r = fmaxf(0.0f, fminf(255.9999f, dither + 255.0f * linearRGB_to_SRGB(pixel.r)));
    pixel.g = fmaxf(0.0f, fminf(255.9999f, dither + 255.0f * linearRGB_to_SRGB(pixel.g)));
    pixel.b = fmaxf(0.0f, fminf(255.9999f, dither + 255.0f * linearRGB_to_SRGB(pixel.b)));

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
