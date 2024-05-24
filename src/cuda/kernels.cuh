#ifndef CU_KERNELS_H
#define CU_KERNELS_H

#include <cuda_runtime_api.h>

#include "bsdf_utils.cuh"
#include "bvh.cuh"
#include "camera.cuh"
#include "cloud.cuh"
#include "ior_stack.cuh"
#include "ocean.cuh"
#include "purkinje.cuh"
#include "sky.cuh"
#include "sky_utils.cuh"
#include "state.cuh"
#include "temporal.cuh"
#include "tonemap.cuh"
#include "utils.cuh"
#include "volume.cuh"

LUMINARY_KERNEL void generate_trace_tasks() {
  int offset       = 0;
  const int amount = device.width * device.height;

  for (int pixel = THREAD_ID; pixel < amount; pixel += blockDim.x * gridDim.x) {
    TraceTask task;

    task.index.y = (uint16_t) (pixel / device.width);
    task.index.x = (uint16_t) (pixel - task.index.y * device.width);

    task = camera_get_ray(task, pixel);

    device.ptrs.records[pixel]      = get_color(1.0f, 1.0f, 1.0f);
    device.ptrs.frame_buffer[pixel] = get_color(0.0f, 0.0f, 0.0f);
    device.ptrs.state_buffer[pixel] = STATE_FLAG_BOUNCE_LIGHTING;

    if ((device.denoiser || device.aov_mode) && !device.temporal_frames) {
      device.ptrs.albedo_buffer[pixel] = get_color(0.0f, 0.0f, 0.0f);
      device.ptrs.normal_buffer[pixel] = get_color(0.0f, 0.0f, 0.0f);
    }

    device.ptrs.frame_direct_buffer[pixel]   = get_color(0.0f, 0.0f, 0.0f);
    device.ptrs.frame_indirect_buffer[pixel] = get_color(0.0f, 0.0f, 0.0f);

    const float ambient_ior = bsdf_refraction_index_ambient(task.origin, task.ray);
    ior_stack_interact(ambient_ior, pixel, IOR_STACK_METHOD_RESET);

    store_trace_task(device.ptrs.trace_tasks + get_task_address(offset++), task);
  }

  device.ptrs.trace_counts[THREAD_ID] = offset;
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

LUMINARY_KERNEL void preprocess_trace_tasks() {
  const int task_count = device.ptrs.trace_counts[THREAD_ID];

  for (int i = 0; i < task_count; i++) {
    const int offset = get_task_address(i);
    TraceTask task   = load_trace_task(device.ptrs.trace_tasks + offset);

    const uint32_t pixel = get_pixel_id(task.index.x, task.index.y);

    float depth     = device.scene.camera.far_clip_distance;
    uint32_t hit_id = HIT_TYPE_SKY;

    if (device.shading_mode != SHADING_HEAT && IS_PRIMARY_RAY) {
      uint32_t t_id;
      TraversalTriangle tt;
      uint32_t material_id;

      t_id = device.ptrs.trace_result_buffer[get_pixel_id(task.index.x, task.index.y)].hit_id;
      if (t_id <= HIT_TYPE_TRIANGLE_ID_LIMIT) {
        material_id = load_triangle_material_id(t_id);

        const float4 data0 = __ldg((float4*) triangle_get_entry_address(0, 0, t_id));
        const float4 data1 = __ldg((float4*) triangle_get_entry_address(1, 0, t_id));
        const float data2  = __ldg((float*) triangle_get_entry_address(2, 0, t_id));

        tt.vertex = get_vector(data0.x, data0.y, data0.z);
        tt.edge1  = get_vector(data0.w, data1.x, data1.y);
        tt.edge2  = get_vector(data1.z, data1.w, data2);
        tt.id     = t_id;

        const Material mat = load_material(device.scene.materials, material_id);

        tt.albedo_tex = mat.albedo_map;

        // This optimization does not work with displacement.
        if (mat.normal_map == TEXTURE_NONE) {
          float2 coords;
          const float dist = bvh_triangle_intersection_uv(tt, task.origin, task.ray, coords);

          if (dist < depth) {
            const BVHAlphaResult alpha_result = bvh_triangle_intersection_alpha_test(tt, t_id, coords);

            if (alpha_result != BVH_ALPHA_RESULT_TRANSPARENT) {
              depth  = dist;
              hit_id = t_id;
            }
          }
        }
      }
    }

    if (device.scene.toy.active) {
      const float toy_dist = get_toy_distance(task.origin, task.ray);

      if (toy_dist < depth) {
        depth  = toy_dist;
        hit_id = HIT_TYPE_TOY;
      }
    }

    if (device.scene.ocean.active) {
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

LUMINARY_KERNEL void process_sky_inscattering_tasks() {
  const int task_count = device.ptrs.trace_counts[THREAD_ID];

  for (int i = 0; i < task_count; i++) {
    const int offset    = get_task_address(i);
    TraceTask task      = load_trace_task(device.ptrs.trace_tasks + offset);
    const float2 result = __ldcs((float2*) (device.ptrs.trace_results + offset));

    const float depth     = result.x;
    const uint32_t hit_id = __float_as_uint(result.y);

    if (hit_id == HIT_TYPE_SKY) {
      continue;
    }

    const int pixel = task.index.y * device.width + task.index.x;

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
  int volume_offset   = toy_offset + toy_task_count;
  int particle_offset = volume_offset + volume_task_count;
  int ocean_offset    = particle_offset + particle_task_count;
  int sky_offset      = ocean_offset + ocean_task_count;
  int rejects_offset  = sky_offset + sky_task_count;
  int k               = 0;

  device.ptrs.task_offsets[THREAD_ID * TASK_ADDRESS_OFFSET_STRIDE + TASK_ADDRESS_OFFSET_GEOMETRY] = geometry_offset;
  device.ptrs.task_offsets[THREAD_ID * TASK_ADDRESS_OFFSET_STRIDE + TASK_ADDRESS_OFFSET_VOLUME]   = volume_offset;
  device.ptrs.task_offsets[THREAD_ID * TASK_ADDRESS_OFFSET_STRIDE + TASK_ADDRESS_OFFSET_OCEAN]    = ocean_offset;
  device.ptrs.task_offsets[THREAD_ID * TASK_ADDRESS_OFFSET_STRIDE + TASK_ADDRESS_OFFSET_SKY]      = sky_offset;

  const int num_tasks               = rejects_offset;
  const int initial_geometry_offset = 0;
  const int initial_toy_offset      = initial_geometry_offset + geometry_task_count;
  const int initial_volume_offset   = initial_toy_offset + toy_task_count;
  const int initial_particle_offset = initial_volume_offset + volume_task_count;
  const int initial_ocean_offset    = initial_particle_offset + particle_task_count;
  const int initial_sky_offset      = initial_ocean_offset + ocean_task_count;
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
    else if (hit_id == HIT_TYPE_OCEAN) {
      index          = ocean_offset;
      needs_swapping = (k < initial_ocean_offset) || (k >= ocean_task_count + initial_ocean_offset);
      if (needs_swapping || k >= ocean_offset) {
        ocean_offset++;
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
    const uint32_t pixel  = get_pixel_id(task.index.x, task.index.y);

    if (IS_PRIMARY_RAY) {
      device.ptrs.raydir_buffer[pixel] = task.ray;

      TraceResult trace_result;
      trace_result.depth  = depth;
      trace_result.hit_id = hit_id;

      device.ptrs.trace_result_buffer[pixel] = trace_result;
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

  device.ptrs.task_counts[THREAD_ID * TASK_ADDRESS_COUNT_STRIDE + TASK_ADDRESS_OFFSET_GEOMETRY]   = geometry_task_count + toy_task_count;
  device.ptrs.task_counts[THREAD_ID * TASK_ADDRESS_COUNT_STRIDE + TASK_ADDRESS_OFFSET_VOLUME]     = volume_task_count + particle_task_count;
  device.ptrs.task_counts[THREAD_ID * TASK_ADDRESS_COUNT_STRIDE + TASK_ADDRESS_OFFSET_OCEAN]      = ocean_task_count;
  device.ptrs.task_counts[THREAD_ID * TASK_ADDRESS_COUNT_STRIDE + TASK_ADDRESS_OFFSET_SKY]        = sky_task_count;
  device.ptrs.task_counts[THREAD_ID * TASK_ADDRESS_COUNT_STRIDE + TASK_ADDRESS_OFFSET_TOTALCOUNT] = num_tasks;

  device.ptrs.trace_counts[THREAD_ID] = 0;
}

LUMINARY_KERNEL void convert_RGBF_to_XRGB8(
  const RGBF* source, XRGB8* dest, const int width, const int height, const int ld, const OutputVariable output_variable) {
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

    RGBF pixel = sample_pixel_clamp(source, sx, sy, src_width, src_height);

    if (output_variable != OUTPUT_VARIABLE_ALBEDO_GUIDANCE && output_variable != OUTPUT_VARIABLE_NORMAL_GUIDANCE) {
      if (device.scene.camera.purkinje) {
        pixel = purkinje_shift(pixel);
      }

      if (device.scene.camera.use_color_correction) {
        RGBF hsv = rgb_to_hsv(pixel);

        hsv = add_color(hsv, device.scene.camera.color_correction);

        if (hsv.r < 0.0f)
          hsv.r += 1.0f;
        if (hsv.r > 1.0f)
          hsv.r -= 1.0f;
        hsv.g = __saturatef(hsv.g);
        if (hsv.b < 0.0f)
          hsv.b = 0.0f;

        pixel = hsv_to_rgb(hsv);
      }

      pixel.r *= device.scene.camera.exposure;
      pixel.g *= device.scene.camera.exposure;
      pixel.b *= device.scene.camera.exposure;

      switch (device.scene.camera.tonemap) {
        case TONEMAP_NONE:
          break;
        case TONEMAP_ACES:
          pixel = tonemap_aces(pixel);
          break;
        case TONEMAP_REINHARD:
          pixel = tonemap_reinhard(pixel);
          break;
        case TONEMAP_UNCHARTED2:
          pixel = tonemap_uncharted2(pixel);
          break;
        case TONEMAP_AGX:
          pixel = tonemap_agx(pixel);
          break;
        case TONEMAP_AGX_PUNCHY:
          pixel = tonemap_agx_punchy(pixel);
          break;
        case TONEMAP_AGX_CUSTOM:
          pixel = tonemap_agx_custom(pixel);
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
    }

    const float dither = (device.scene.camera.dithering) ? random_dither_mask(x, y) : 0.0f;

    pixel.r = fminf(255.9f, dither + 255.9f * linearRGB_to_SRGB(pixel.r));
    pixel.g = fminf(255.9f, dither + 255.9f * linearRGB_to_SRGB(pixel.g));
    pixel.b = fminf(255.9f, dither + 255.9f * linearRGB_to_SRGB(pixel.b));

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
