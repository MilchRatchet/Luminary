#ifndef CU_KERNELS_H
#define CU_KERNELS_H

#include "bsdf_utils.cuh"
#include "bvh.cuh"
#include "camera.cuh"
#include "camera_post_common.cuh"
#include "cloud.cuh"
#include "ior_stack.cuh"
#include "purkinje.cuh"
#include "sky.cuh"
#include "sky_utils.cuh"
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

// TODO: Turn this into a function and call this in the trace kernel if primary ray.
LUMINARY_KERNEL void generate_trace_tasks() {
  HANDLE_DEVICE_ABORT();

  uint32_t task_count = 0;

  const uint32_t undersampling_stage     = (device.state.undersampling & UNDERSAMPLING_STAGE_MASK) >> UNDERSAMPLING_STAGE_SHIFT;
  const uint32_t undersampling_iteration = device.state.undersampling & UNDERSAMPLING_ITERATION_MASK;

  const uint32_t undersampling_scale = 1 << undersampling_stage;

  const uint32_t undersampling_width  = (device.settings.width + (1 << undersampling_stage) - 1) >> undersampling_stage;
  const uint32_t undersampling_height = (device.settings.height + (1 << undersampling_stage) - 1) >> undersampling_stage;

  const uint32_t amount = undersampling_width * undersampling_height;

  for (uint32_t undersampling_pixel = THREAD_ID; undersampling_pixel < amount; undersampling_pixel += blockDim.x * gridDim.x) {
    uint16_t undersampling_y = (uint16_t) (undersampling_pixel / undersampling_width);
    uint16_t undersampling_x = (uint16_t) (undersampling_pixel - undersampling_y * undersampling_width);

    if (undersampling_scale > 1) {
      undersampling_x *= undersampling_scale;
      undersampling_y *= undersampling_scale;

      undersampling_x += (undersampling_iteration & 0b01) ? 0 : undersampling_scale >> 1;
      undersampling_y += (undersampling_iteration & 0b10) ? 0 : undersampling_scale >> 1;
    }

    if (undersampling_x >= device.settings.width || undersampling_y >= device.settings.height)
      continue;

    if (undersampling_x < device.settings.window_x || undersampling_x >= device.settings.window_x + device.settings.window_width)
      continue;

    if (undersampling_y < device.settings.window_y || undersampling_y >= device.settings.window_y + device.settings.window_height)
      continue;

    DeviceTask task;
    task.state   = STATE_FLAG_DELTA_PATH | STATE_FLAG_CAMERA_DIRECTION | STATE_FLAG_ALLOW_EMISSION;
    task.index.x = undersampling_x;
    task.index.y = undersampling_y;

    task = camera_get_ray(task);

    const uint32_t pixel = get_pixel_id(task.index);

    store_RGBF(device.ptrs.records, pixel, splat_color(1.0f));

    device.ptrs.frame_direct_buffer[pixel]   = {};
    device.ptrs.frame_indirect_buffer[pixel] = {};

    const float ambient_ior = bsdf_refraction_index_ambient(task.origin, task.ray);
    ior_stack_interact(ambient_ior, pixel, IOR_STACK_METHOD_RESET);

    task_store(task, get_task_address(task_count++));
  }

  device.ptrs.trace_counts[THREAD_ID] = task_count;
}

LUMINARY_KERNEL void balance_trace_tasks() {
  HANDLE_DEVICE_ABORT();

  const uint32_t warp = THREAD_ID;

  if (warp >= (THREADS_PER_BLOCK * BLOCKS_PER_GRID) >> 5)
    return;

  __shared__ uint16_t counts[32 * THREADS_PER_BLOCK];
  uint32_t sum = 0;

  for (uint32_t i = 0; i < 32; i += 4) {
    ushort4 c                                         = __ldcs((ushort4*) (device.ptrs.trace_counts + 32 * warp + i));
    counts[threadIdx.x + (i + 0) * THREADS_PER_BLOCK] = c.x;
    counts[threadIdx.x + (i + 1) * THREADS_PER_BLOCK] = c.y;
    counts[threadIdx.x + (i + 2) * THREADS_PER_BLOCK] = c.z;
    counts[threadIdx.x + (i + 3) * THREADS_PER_BLOCK] = c.w;
    sum += c.x;
    sum += c.y;
    sum += c.z;
    sum += c.w;
  }

  const uint16_t average = 1 + (sum >> 5);

  for (uint32_t i = 0; i < 32; i++) {
    uint16_t count        = counts[threadIdx.x + i * THREADS_PER_BLOCK];
    int source_index      = -1;
    uint16_t source_count = 0;

    if (count >= average)
      continue;

    for (uint32_t j = 0; j < 32; j++) {
      uint16_t c = counts[threadIdx.x + j * THREADS_PER_BLOCK];
      if (c > average && c > count + 1 && c > source_count) {
        source_count = c;
        source_index = j;
      }
    }

    if (source_index != -1) {
      const uint32_t swaps = (source_count - count) >> 1;

      static_assert(THREADS_PER_BLOCK == 128, "The following code assumes that we have 4 warps per block.");
      const uint32_t thread_id_base = ((warp & 0b11) << 5);
      const uint32_t block_id       = warp >> 2;

      for (uint32_t j = 0; j < swaps; j++) {
        // TODO: Write a function for this
        DeviceTask* source_ptr = device.ptrs.tasks + get_task_address_of_thread(thread_id_base + source_index, block_id, source_count - 1);
        DeviceTask* sink_ptr   = device.ptrs.tasks + get_task_address_of_thread(thread_id_base + i, block_id, count);

        __stwb((float4*) sink_ptr, __ldca((float4*) source_ptr));
        __stwb((float4*) (sink_ptr) + 1, __ldca((float4*) (source_ptr) + 1));

        sink_ptr++;
        count++;

        source_ptr--;
        source_count--;
      }
      counts[threadIdx.x + i * THREADS_PER_BLOCK]            = count;
      counts[threadIdx.x + source_index * THREADS_PER_BLOCK] = source_count;
    }
  }

  for (uint32_t i = 0; i < 32; i += 4) {
    const ushort4 vals = make_ushort4(
      counts[threadIdx.x + (i + 0) * THREADS_PER_BLOCK], counts[threadIdx.x + (i + 1) * THREADS_PER_BLOCK],
      counts[threadIdx.x + (i + 2) * THREADS_PER_BLOCK], counts[threadIdx.x + (i + 3) * THREADS_PER_BLOCK]);
    __stcs((ushort4*) (device.ptrs.trace_counts + 32 * warp + i), vals);
  }
}

LUMINARY_KERNEL void sky_process_inscattering_events() {
  HANDLE_DEVICE_ABORT();

  const int task_count = device.ptrs.trace_counts[THREAD_ID];

  for (int i = 0; i < task_count; i++) {
    const int offset            = get_task_address(i);
    DeviceTask task             = task_load(offset);
    const TriangleHandle handle = triangle_handle_load(offset);

    if (handle.instance_id == HIT_TYPE_SKY)
      continue;

    const float depth = trace_depth_load(offset);

    const uint32_t pixel = get_pixel_id(task.index);

    const vec3 sky_origin = world_to_sky_transform(task.origin);

    const float inscattering_limit = world_to_sky_scale(depth);

    RGBF record = load_RGBF(device.ptrs.records + pixel);

    const RGBF inscattering = sky_trace_inscattering(sky_origin, task.ray, inscattering_limit, record, task.index);

    store_RGBF(device.ptrs.records, pixel, record);
    write_beauty_buffer(inscattering, pixel, task.state);
  }
}

LUMINARY_KERNEL void postprocess_trace_tasks() {
  HANDLE_DEVICE_ABORT();

  const uint32_t task_count    = device.ptrs.trace_counts[THREAD_ID];
  uint16_t geometry_task_count = 0;
  uint16_t ocean_task_count    = 0;
  uint16_t volume_task_count   = 0;
  uint16_t particle_task_count = 0;
  uint16_t sky_task_count      = 0;

  // count data
  for (uint32_t i = 0; i < task_count; i++) {
    const uint32_t offset      = get_task_address(i);
    const uint32_t instance_id = triangle_handle_load(offset).instance_id;

    if (instance_id == HIT_TYPE_SKY) {
      sky_task_count++;
    }
    else if (instance_id == HIT_TYPE_OCEAN) {
      ocean_task_count++;
    }
    else if (VOLUME_HIT_CHECK(instance_id)) {
      volume_task_count++;
    }
    else if (instance_id <= HIT_TYPE_PARTICLE_MAX && instance_id >= HIT_TYPE_PARTICLE_MIN) {
      particle_task_count++;
    }
    else if (instance_id <= HIT_TYPE_TRIANGLE_ID_LIMIT) {
      geometry_task_count++;
    }
  }

  uint32_t geometry_offset = 0;
  uint32_t ocean_offset    = geometry_offset + geometry_task_count;
  uint32_t volume_offset   = ocean_offset + ocean_task_count;
  uint32_t particle_offset = volume_offset + volume_task_count;
  uint32_t sky_offset      = particle_offset + particle_task_count;
  uint32_t rejects_offset  = sky_offset + sky_task_count;
  uint32_t k               = 0;

  device.ptrs.task_offsets[TASK_ADDRESS_OFFSET_GEOMETRY] = geometry_offset;
  device.ptrs.task_offsets[TASK_ADDRESS_OFFSET_OCEAN]    = ocean_offset;
  device.ptrs.task_offsets[TASK_ADDRESS_OFFSET_VOLUME]   = volume_offset;
  device.ptrs.task_offsets[TASK_ADDRESS_OFFSET_PARTICLE] = particle_offset;
  device.ptrs.task_offsets[TASK_ADDRESS_OFFSET_SKY]      = sky_offset;

  const uint32_t num_tasks               = rejects_offset;
  const uint32_t initial_geometry_offset = 0;
  const uint32_t initial_ocean_offset    = initial_geometry_offset + geometry_task_count;
  const uint32_t initial_volume_offset   = initial_ocean_offset + ocean_task_count;
  const uint32_t initial_particle_offset = initial_volume_offset + volume_task_count;
  const uint32_t initial_sky_offset      = initial_particle_offset + particle_task_count;
  const uint32_t initial_rejects_offset  = initial_sky_offset + sky_task_count;

  // order data
  while (k < task_count) {
    const uint32_t offset      = get_task_address(k);
    const uint32_t instance_id = triangle_handle_load(offset).instance_id;

    uint32_t index;
    bool needs_swapping;

    if (instance_id <= HIT_TYPE_TRIANGLE_ID_LIMIT) {
      index          = geometry_offset;
      needs_swapping = k >= geometry_task_count + initial_geometry_offset;
      if (needs_swapping || k >= geometry_offset) {
        geometry_offset++;
      }
    }
    else if (instance_id == HIT_TYPE_OCEAN) {
      index          = ocean_offset;
      needs_swapping = (k < initial_ocean_offset) || (k >= ocean_task_count + initial_ocean_offset);
      if (needs_swapping || k >= ocean_offset) {
        ocean_offset++;
      }
    }
    else if (VOLUME_HIT_CHECK(instance_id)) {
      index          = volume_offset;
      needs_swapping = (k < initial_volume_offset) || (k >= volume_task_count + initial_volume_offset);
      if (needs_swapping || k >= volume_offset) {
        volume_offset++;
      }
    }
    else if (instance_id <= HIT_TYPE_PARTICLE_MAX) {
      index          = particle_offset;
      needs_swapping = (k < initial_particle_offset) || (k >= particle_task_count + initial_particle_offset);
      if (needs_swapping || k >= particle_offset) {
        particle_offset++;
      }
    }
    else if (instance_id == HIT_TYPE_SKY) {
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

  device.ptrs.task_counts[TASK_ADDRESS_OFFSET_GEOMETRY] = geometry_task_count;
  device.ptrs.task_counts[TASK_ADDRESS_OFFSET_OCEAN]    = ocean_task_count;
  device.ptrs.task_counts[TASK_ADDRESS_OFFSET_VOLUME]   = volume_task_count;
  device.ptrs.task_counts[TASK_ADDRESS_OFFSET_PARTICLE] = particle_task_count;
  device.ptrs.task_counts[TASK_ADDRESS_OFFSET_SKY]      = sky_task_count;
  device.ptrs.task_counts[TASK_ADDRESS_OFFSET_TOTAL]    = num_tasks;

  device.ptrs.trace_counts[THREAD_ID] = 0;
}

__device__ RGBF final_image_get_undersampling_sample(
  const KernelArgsGenerateFinalImage args, const uint16_t source_x, const uint16_t source_y, const uint32_t output_scale,
  const uint32_t undersampling_iteration, const uint32_t undersampling_stage) {
  const uint32_t offset_x = (undersampling_iteration & 0b01) ? 0 : output_scale >> 1;
  const uint32_t offset_y = (undersampling_iteration & 0b10) ? 0 : output_scale >> 1;

  const uint32_t pixel_x = min(source_x + offset_x, device.settings.width - 1);
  const uint32_t pixel_y = min(source_y + offset_y, device.settings.height - 1);

  const uint32_t index = pixel_x + pixel_y * device.settings.width;

  RGBF pixel = load_RGBF(args.src + index);

  if (undersampling_stage == 0) {
    pixel = tonemap_apply(pixel, pixel_x, pixel_y, args.color_correction, args.agx_params);
  }

  return pixel;
}

LUMINARY_KERNEL void generate_final_image(const KernelArgsGenerateFinalImage args) {
  HANDLE_DEVICE_ABORT();

  const uint32_t undersampling_stage     = (device.state.undersampling & UNDERSAMPLING_STAGE_MASK) >> UNDERSAMPLING_STAGE_SHIFT;
  const uint32_t undersampling_iteration = device.state.undersampling & UNDERSAMPLING_ITERATION_MASK;

  const uint32_t undersampling_output = max(undersampling_stage, device.settings.supersampling);

  const uint32_t output_scale = 1 << undersampling_output;

  const uint32_t output_width  = device.settings.width >> undersampling_output;
  const uint32_t output_height = device.settings.height >> undersampling_output;

  const uint32_t supersampling_scale = max(device.settings.supersampling, undersampling_stage) - undersampling_stage;

  const uint32_t output_amount = output_width * output_height;

  const uint32_t num_samples = (undersampling_stage) ? 4 : (1 << device.settings.supersampling);

  const float color_scale = (undersampling_stage) ? (1.0f / (num_samples - undersampling_iteration)) : 1.0f / (output_scale * output_scale);

  for (uint32_t output_pixel = THREAD_ID; output_pixel < output_amount; output_pixel += blockDim.x * gridDim.x) {
    const uint16_t y = (uint16_t) (output_pixel / output_width);
    const uint16_t x = (uint16_t) (output_pixel - y * output_width);

    const uint16_t source_x = x * output_scale;
    const uint16_t source_y = y * output_scale;

    RGBF color = get_color(0.0f, 0.0f, 0.0f);

    if (undersampling_stage) {
      for (uint32_t sample_id = undersampling_iteration; sample_id < num_samples; sample_id++) {
        const RGBF pixel = final_image_get_undersampling_sample(args, source_x, source_y, output_scale, sample_id, undersampling_stage);
        color            = add_color(color, pixel);
      }
    }
    else {
      for (uint32_t yi = 0; yi < output_scale; yi++) {
        for (uint32_t xi = 0; xi < output_scale; xi++) {
          const uint32_t pixel_x = min(source_x + xi, device.settings.width - 1);
          const uint32_t pixel_y = min(source_y + yi, device.settings.height - 1);

          const uint32_t index = pixel_x + pixel_y * device.settings.width;

          RGBF pixel = load_RGBF(args.src + index);

          if (undersampling_stage == 0) {
            pixel = tonemap_apply(pixel, pixel_x, pixel_y, args.color_correction, args.agx_params);
          }

          color = add_color(color, pixel);
        }
      }
    }

    color = scale_color(color, color_scale);

    if (undersampling_stage != 0) {
      color = tonemap_apply(color, x, y, args.color_correction, args.agx_params);
    }

    const uint32_t dst_index = x + y * output_width;
    store_RGBF(device.ptrs.frame_final, dst_index, color);
  }
}

LUMINARY_KERNEL void convert_RGBF_to_ARGB8(const KernelArgsConvertRGBFToARGB8 args) {
  HANDLE_DEVICE_ABORT();

  uint32_t id = THREAD_ID;

  const uint32_t amount = args.width * args.height;
  const float scale_x   = 1.0f / (args.width - 1);
  const float scale_y   = 1.0f / (args.height - 1);

  const uint32_t undersampling_stage = (device.state.undersampling & UNDERSAMPLING_STAGE_MASK) >> UNDERSAMPLING_STAGE_SHIFT;

  const uint32_t undersampling_output = max(undersampling_stage, device.settings.supersampling);

  const uint32_t final_image_width  = device.settings.width >> device.settings.supersampling;
  const uint32_t final_image_height = device.settings.height >> device.settings.supersampling;

  const uint32_t output_width = device.settings.width >> undersampling_output;

  const uint32_t undersampling_mem = undersampling_output - device.settings.supersampling;

  const float mem_scale    = 1.0f / (1 << undersampling_mem);
  const bool scaled_output = (args.width != final_image_width) || (args.height != final_image_height);

  while (id < amount) {
    const uint32_t y = id / args.width;
    const uint32_t x = id - y * args.width;

    RGBF pixel;
    if (scaled_output) {
      const float sx = x * scale_x;
      const float sy = y * scale_y;

      pixel = sample_pixel_clamp(device.ptrs.frame_final, sx, sy, final_image_width, final_image_height, mem_scale);
    }
    else {
      const uint32_t src_x = x >> undersampling_mem;
      const uint32_t src_y = y >> undersampling_mem;

      pixel = load_RGBF(device.ptrs.frame_final + src_x + src_y * output_width);
    }

    switch (args.filter) {
      case LUMINARY_FILTER_NONE:
        break;
      case LUMINARY_FILTER_GRAY:
        pixel = filter_gray(pixel);
        break;
      case LUMINARY_FILTER_SEPIA:
        pixel = filter_sepia(pixel);
        break;
      case LUMINARY_FILTER_GAMEBOY:
        pixel = filter_gameboy(pixel, x, y);
        break;
      case LUMINARY_FILTER_2BITGRAY:
        pixel = filter_2bitgray(pixel, x, y);
        break;
      case LUMINARY_FILTER_CRT:
        pixel = filter_crt(pixel, x, y);
        break;
      case LUMINARY_FILTER_BLACKWHITE:
        pixel = filter_blackwhite(pixel, x, y);
        break;
    }

    const float dither = (device.camera.dithering) ? random_dither_mask(x, y) : 0.5f;

    pixel.r = fmaxf(0.0f, fminf(255.9999f, dither + 255.0f * linearRGB_to_SRGB(pixel.r)));
    pixel.g = fmaxf(0.0f, fminf(255.9999f, dither + 255.0f * linearRGB_to_SRGB(pixel.g)));
    pixel.b = fmaxf(0.0f, fminf(255.9999f, dither + 255.0f * linearRGB_to_SRGB(pixel.b)));

    ARGB8 converted_pixel;
    converted_pixel.a = 0xFF;
    converted_pixel.r = (uint8_t) pixel.r;
    converted_pixel.g = (uint8_t) pixel.g;
    converted_pixel.b = (uint8_t) pixel.b;

    args.dst[x + y * args.width] = converted_pixel;

    id += blockDim.x * gridDim.x;
  }
}

#endif /* CU_KERNELS_H */
