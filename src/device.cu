#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuda/kernels.cuh"
#include "utils.h"

#define UTILS_NO_DEVICE_TABLE
__constant__ DeviceConstantMemory device;

#include "bench.h"
#include "buffer.h"
#include "config.h"
#include "cuda/camera_post.cuh"
#include "cuda/micromap.cuh"
#include "cuda/utils.cuh"
#include "device.h"
#include "log.h"
#include "optixrt.h"
#include "raytrace.h"
#include "structs.h"
#include "texture.h"

extern "C" void device_init() {
  gpuErrchk(cudaSetDeviceFlags(cudaDeviceMapHost));

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);

  device_set_memory_limit(prop.totalGlobalMem);

  print_info("Luminary - %s", prop.name);
  print_info("[%s] %s", LUMINARY_BRANCH_NAME, LUMINARY_VERSION_DATE);
  print_info("Compiled using %s on %s", LUMINARY_COMPILER, LUMINARY_OS);
  print_info("CUDA Version %s OptiX Version %s", LUMINARY_CUDA_VERSION, LUMINARY_OPTIX_VERSION);
  print_info("Copyright (c) 2023 MilchRatchet");
}

#define CLOUD_SHAPE_RES 128
#define CLOUD_DETAIL_RES 32
#define CLOUD_WEATHER_RES 1024

extern "C" void device_cloud_noise_generate(RaytraceInstance* instance) {
  bench_tic((const char*) "Cloud Noise Generation");

  if (instance->scene.sky.cloud.initialized) {
    texture_free_atlas(instance->cloud_noise, 3);
  }

  TextureRGBA noise_tex[3];
  texture_create(noise_tex + 0, CLOUD_SHAPE_RES, CLOUD_SHAPE_RES, CLOUD_SHAPE_RES, CLOUD_SHAPE_RES, (void*) 0, TexDataUINT8, TexStorageGPU);
  texture_create(
    noise_tex + 1, CLOUD_DETAIL_RES, CLOUD_DETAIL_RES, CLOUD_DETAIL_RES, CLOUD_DETAIL_RES, (void*) 0, TexDataUINT8, TexStorageGPU);
  texture_create(noise_tex + 2, CLOUD_WEATHER_RES, CLOUD_WEATHER_RES, 1, CLOUD_WEATHER_RES, (void*) 0, TexDataUINT8, TexStorageGPU);

  noise_tex[0].mipmap = TexMipmapGenerate;
  noise_tex[1].mipmap = TexMipmapGenerate;
  noise_tex[2].mipmap = TexMipmapNone;

  device_malloc((void**) &noise_tex[0].data, noise_tex[0].depth * noise_tex[0].height * noise_tex[0].pitch * 4 * sizeof(uint8_t));
  cloud_noise_generate_shape<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(noise_tex[0].width, (uint8_t*) noise_tex[0].data);

  device_malloc((void**) &noise_tex[1].data, noise_tex[1].depth * noise_tex[1].height * noise_tex[1].pitch * 4 * sizeof(uint8_t));
  cloud_noise_generate_detail<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(noise_tex[1].width, (uint8_t*) noise_tex[1].data);

  device_malloc((void**) &noise_tex[2].data, noise_tex[2].height * noise_tex[2].pitch * 4 * sizeof(uint8_t));
  cloud_noise_generate_weather<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(
    noise_tex[2].width, (float) instance->scene.sky.cloud.seed, (uint8_t*) noise_tex[2].data);

  texture_create_atlas(&instance->cloud_noise, noise_tex, 3);

  device_free(noise_tex[0].data, noise_tex[0].depth * noise_tex[0].height * noise_tex[0].pitch * 4 * sizeof(uint8_t));
  device_free(noise_tex[1].data, noise_tex[1].depth * noise_tex[1].height * noise_tex[1].pitch * 4 * sizeof(uint8_t));
  device_free(noise_tex[2].data, noise_tex[2].height * noise_tex[2].pitch * 4 * sizeof(uint8_t));

  raytrace_update_device_pointers(instance);

  instance->scene.sky.cloud.initialized = 1;

  bench_toc();
}

void device_handle_accumulation(RaytraceInstance* instance) {
  switch (instance->accum_mode) {
    case NO_ACCUMULATION:
      device_buffer_copy(instance->frame_buffer, instance->frame_accumulate);
      break;
    case TEMPORAL_ACCUMULATION:
      temporal_accumulation<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
      break;
    case TEMPORAL_REPROJECTION:
      device_buffer_copy(instance->frame_accumulate, instance->frame_temporal);
      temporal_reprojection<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
      break;
    default:
      error_message("Invalid accumulation mode %d specified", instance->accum_mode);
      break;
  }

  if (instance->aov_mode) {
    switch (instance->accum_mode) {
      case NO_ACCUMULATION:
      case TEMPORAL_REPROJECTION:
        device_buffer_copy(instance->frame_direct_buffer, instance->frame_direct_accumulate);
        device_buffer_copy(instance->frame_indirect_buffer, instance->frame_indirect_accumulate);
        break;
      case TEMPORAL_ACCUMULATION:
        temporal_accumulation_aov<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(
          (const RGBF*) device_buffer_get_pointer(instance->frame_direct_buffer),
          (RGBF*) device_buffer_get_pointer(instance->frame_direct_accumulate));
        temporal_accumulation_aov<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(
          (const RGBF*) device_buffer_get_pointer(instance->frame_indirect_buffer),
          (RGBF*) device_buffer_get_pointer(instance->frame_indirect_accumulate));
        break;
      default:
        error_message("Invalid accumulation mode %d specified", instance->accum_mode);
        break;
    }
  }
}

static void bind_type(RaytraceInstance* instance, int type, int depth) {
  device_update_symbol(iteration_type, type);
  device_update_symbol(depth, depth);

  switch (type) {
    case TYPE_CAMERA:
    case TYPE_BOUNCE:
      device_update_symbol(trace_tasks, instance->bounce_trace->device_pointer);
      device_update_symbol(trace_count, instance->bounce_trace_count->device_pointer);
      device_update_symbol(records, instance->bounce_records->device_pointer);
      break;
    case TYPE_LIGHT:
      device_update_symbol(trace_tasks, instance->light_trace->device_pointer);
      device_update_symbol(trace_count, instance->light_trace_count->device_pointer);
      device_update_symbol(records, instance->light_records->device_pointer);
      break;
  }
}

extern "C" void device_execute_main_kernels(RaytraceInstance* instance, int type, int depth) {
  bind_type(instance, type, depth);

  if (type != TYPE_CAMERA)
    balance_trace_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();

  preprocess_trace_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();

  if (instance->scene.particles.active && type != TYPE_LIGHT) {
    optixrt_execute(instance->particles_instance.kernel);
  }

  if (instance->bvh_type == BVH_LUMINARY) {
    process_trace_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
  }
  else {
    optixrt_execute(instance->optix_kernel);
  }

  if (instance->scene.fog.active || instance->scene.ocean.active) {
    volume_process_events<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
  }

  if (instance->scene.sky.cloud.active && !instance->scene.sky.hdri_active) {
    clouds_render_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
  }

  if (type != TYPE_LIGHT && instance->scene.sky.aerial_perspective) {
    process_sky_inscattering_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
  }

  postprocess_trace_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();

  if (type != TYPE_LIGHT) {
    restir_candidates_pool_generation<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
  }

  process_geometry_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();

  if (instance->scene.particles.active && type != TYPE_LIGHT) {
    particle_process_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
  }

  if (instance->scene.ocean.active) {
    process_ocean_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
  }

  process_sky_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();

  if (instance->scene.toy.active) {
    process_toy_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
  }

  if (type != TYPE_LIGHT && (instance->scene.fog.active || instance->scene.ocean.active)) {
    volume_process_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
  }
}

extern "C" void device_execute_debug_kernels(RaytraceInstance* instance, int type) {
  bind_type(instance, type, 0);

  preprocess_trace_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();

  if (instance->scene.particles.active && type != TYPE_LIGHT) {
    optixrt_execute(instance->particles_instance.kernel);
  }

  if (instance->bvh_type == BVH_LUMINARY) {
    process_trace_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
  }
  else {
    optixrt_execute(instance->optix_kernel);
  }

  if (instance->scene.fog.active || instance->scene.ocean.active) {
    volume_process_events<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
  }

  postprocess_trace_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();

  process_debug_geometry_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
  if (instance->scene.particles.active && type != TYPE_LIGHT) {
    particle_process_debug_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
  }
  if (instance->scene.ocean.active) {
    process_debug_ocean_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
  }
  process_debug_sky_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
  if (instance->scene.toy.active) {
    process_debug_toy_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
  }
}

extern "C" void device_generate_tasks() {
  generate_trace_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
}

extern "C" void _device_update_symbol(const size_t offset, const void* src, const size_t size) {
  gpuErrchk(cudaMemcpyToSymbol(device, src, size, offset, cudaMemcpyHostToDevice));
}

extern "C" void _device_gather_symbol(void* dst, const size_t offset, size_t size) {
  gpuErrchk(cudaMemcpyFromSymbol(dst, device, size, offset, cudaMemcpyDeviceToHost));
}

extern "C" void device_gather_device_table(void* dst, enum cudaMemcpyKind kind) {
  gpuErrchk(cudaMemcpyFromSymbol(dst, device, sizeof(DeviceConstantMemory), 0, kind));
}

extern "C" void device_initialize_random_generators() {
  initialize_randoms<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
}

extern "C" unsigned int device_get_thread_count() {
  return THREADS_PER_BLOCK * BLOCKS_PER_GRID;
}

extern "C" void device_copy_framebuffer_to_8bit(
  RGBF* gpu_source, XRGB8* gpu_scratch, XRGB8* cpu_dest, const int width, const int height, const int ld,
  const OutputVariable output_variable) {
  convert_RGBF_to_XRGB8<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(gpu_source, gpu_scratch, width, height, ld, output_variable);
  gpuErrchk(cudaMemcpy(cpu_dest, gpu_scratch, sizeof(XRGB8) * ld * height, cudaMemcpyDeviceToHost));
}

extern "C" void device_particle_generate(RaytraceInstance* instance) {
  bench_tic((const char*) "Particles Generation");

  ParticlesInstance particles = instance->particles_instance;

  if (particles.vertex_buffer)
    device_buffer_destroy(&particles.vertex_buffer);
  if (particles.index_buffer)
    device_buffer_destroy(&particles.index_buffer);
  if (particles.quad_buffer)
    device_buffer_destroy(&particles.quad_buffer);

  const uint32_t count     = instance->scene.particles.count;
  particles.triangle_count = 2 * count;
  particles.vertex_count   = 4 * count;
  particles.index_count    = 6 * count;

  device_buffer_init(&particles.vertex_buffer);
  device_buffer_init(&particles.index_buffer);
  device_buffer_init(&particles.quad_buffer);

  device_buffer_malloc(particles.vertex_buffer, 4 * sizeof(float4), count);
  device_buffer_malloc(particles.index_buffer, 6 * sizeof(uint32_t), count);
  device_buffer_malloc(particles.quad_buffer, sizeof(Quad), count);

  void* quads = device_buffer_get_pointer(particles.quad_buffer);
  device_update_symbol(particle_quads, quads);

  particle_kernel_generate<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(
    count, instance->scene.particles.size, instance->scene.particles.size_variation,
    (float4*) device_buffer_get_pointer(particles.vertex_buffer), (uint32_t*) device_buffer_get_pointer(particles.index_buffer),
    (Quad*) device_buffer_get_pointer(particles.quad_buffer));
  gpuErrchk(cudaDeviceSynchronize());

  instance->particles_instance = particles;

  bench_toc();
}

extern "C" void device_mipmap_generate(cudaMipmappedArray_t mipmap_array, TextureRGBA* tex) {
  const unsigned int num_levels = device_mipmap_compute_max_level(tex);

  cudaTextureFilterMode filter_mode = texture_get_filter_mode(tex);
  cudaTextureReadMode read_mode     = texture_get_read_mode(tex);

  for (unsigned int level = 0; level < num_levels; level++) {
    cudaArray_t level_src;
    gpuErrchk(cudaGetMipmappedArrayLevel(&level_src, mipmap_array, level));
    cudaArray_t level_dst;
    gpuErrchk(cudaGetMipmappedArrayLevel(&level_dst, mipmap_array, level + 1));

    cudaExtent dst_size;
    gpuErrchk(cudaArrayGetInfo(NULL, &dst_size, NULL, level_dst));

    cudaTextureObject_t src_tex;
    cudaResourceDesc res_desc;
    memset(&res_desc, 0, sizeof(cudaResourceDesc));

    res_desc.resType         = cudaResourceTypeArray;
    res_desc.res.array.array = level_src;

    cudaTextureDesc tex_desc;
    memset(&tex_desc, 0, sizeof(cudaTextureDesc));

    tex_desc.normalizedCoords = 1;
    tex_desc.filterMode       = filter_mode;

    tex_desc.addressMode[0] = texture_get_address_mode(tex->wrap_mode_S);
    tex_desc.addressMode[1] = texture_get_address_mode(tex->wrap_mode_T);
    tex_desc.addressMode[2] = texture_get_address_mode(tex->wrap_mode_R);

    tex_desc.readMode = read_mode;

    gpuErrchk(cudaCreateTextureObject(&src_tex, &res_desc, &tex_desc, NULL));

    cudaSurfaceObject_t dst_surface;
    cudaResourceDesc res_desc_surface;
    memset(&res_desc_surface, 0, sizeof(cudaResourceDesc));
    res_desc_surface.resType         = cudaResourceTypeArray;
    res_desc_surface.res.array.array = level_dst;

    gpuErrchk(cudaCreateSurfaceObject(&dst_surface, &res_desc_surface));

    switch (tex->dim) {
      case Tex2D:
        switch (tex->type) {
          case TexDataFP32:
            mipmap_generate_level_2D_RGBAF<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(src_tex, dst_surface, dst_size.width, dst_size.height);
            break;
          case TexDataUINT8:
            mipmap_generate_level_2D_RGBA8<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(src_tex, dst_surface, dst_size.width, dst_size.height);
            break;
          case TexDataUINT16:
            mipmap_generate_level_2D_RGBA16<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(src_tex, dst_surface, dst_size.width, dst_size.height);
            break;
          default:
            error_message("Invalid texture data type %d", tex->type);
            break;
        }
        break;
      case Tex3D:
        switch (tex->type) {
          case TexDataFP32:
            mipmap_generate_level_3D_RGBAF<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(
              src_tex, dst_surface, dst_size.width, dst_size.height, dst_size.depth);
            break;
          case TexDataUINT8:
            mipmap_generate_level_3D_RGBA8<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(
              src_tex, dst_surface, dst_size.width, dst_size.height, dst_size.depth);
            break;
          case TexDataUINT16:
            mipmap_generate_level_3D_RGBA16<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(
              src_tex, dst_surface, dst_size.width, dst_size.height, dst_size.depth);
            break;
          default:
            error_message("Invalid texture data type %d", tex->type);
            break;
        }
        break;
      default:
        error_message("Invalid texture dim %d", tex->dim);
        break;
    }

    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaGetLastError());

    gpuErrchk(cudaDestroySurfaceObject(dst_surface));

    gpuErrchk(cudaDestroyTextureObject(src_tex));
  }
}

extern "C" unsigned int device_mipmap_compute_max_level(TextureRGBA* tex) {
  unsigned int max_dim;

  switch (tex->dim) {
    case Tex2D:
      max_dim = max(tex->width, tex->height);
      break;
    case Tex3D:
      max_dim = max(tex->width, max(tex->height, tex->depth));
      break;
    default:
      error_message("Invalid texture dim %d", tex->dim);
      return 0;
  }

  if (max_dim == 0)
    return 0;

  unsigned int i = 0;

  while (max_dim != 1) {
    i++;
    max_dim = max_dim >> 1;
  }

  return i;
}
