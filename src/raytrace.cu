#include <cuda_runtime_api.h>
#include <float.h>
#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <chrono>
#include <thread>

#include "SDL.h"
#include "buffer.h"
#include "config.h"
#include "cuda/bloom.cuh"
#include "cuda/brdf.cuh"
#include "cuda/brdf_unittest.cuh"
#include "cuda/bvh.cuh"
#include "cuda/cloud_noise.cuh"
#include "cuda/denoise.cuh"
#include "cuda/directives.cuh"
#include "cuda/kernels.cuh"
#include "cuda/light.cuh"
#include "cuda/math.cuh"
#include "cuda/random.cuh"
#include "cuda/sky.cuh"
#include "cuda/sky_utils.cuh"
#include "cuda/utils.cuh"
#include "device.h"
#include "log.h"
#include "qoi.h"
#include "raytrace.h"
#include "scene.h"
#include "structs.h"
#include "utils.h"

//---------------------------------
// Path Tracing
//---------------------------------

/*
 * Computes sun position.
 * @param scene Scene which contains the sun properties.
 */
static void update_special_lights(const Scene scene) {
  vec3 sun              = angles_to_direction(scene.sky.altitude, scene.sky.azimuth);
  const float scale_sun = 1.0f / (sqrtf(sun.x * sun.x + sun.y * sun.y + sun.z * sun.z));
  sun.x *= scale_sun * SKY_SUN_DISTANCE;
  sun.y *= scale_sun * SKY_SUN_DISTANCE;
  sun.z *= scale_sun * SKY_SUN_DISTANCE;
  sun.y -= SKY_EARTH_RADIUS;
  sun.x -= scene.sky.geometry_offset.x;
  sun.y -= scene.sky.geometry_offset.y;
  sun.z -= scene.sky.geometry_offset.z;

  device_update_symbol(sun, sun);

  vec3 moon              = angles_to_direction(scene.sky.moon_altitude, scene.sky.moon_azimuth);
  const float scale_moon = 1.0f / (sqrtf(moon.x * moon.x + moon.y * moon.y + moon.z * moon.z));
  moon.x *= scale_moon * SKY_MOON_DISTANCE;
  moon.y *= scale_moon * SKY_MOON_DISTANCE;
  moon.z *= scale_moon * SKY_MOON_DISTANCE;
  moon.y -= SKY_EARTH_RADIUS;
  moon.x -= scene.sky.geometry_offset.x;
  moon.y -= scene.sky.geometry_offset.y;
  moon.z -= scene.sky.geometry_offset.z;

  device_update_symbol(moon, moon);
}

/*
 * Computes matrices used for temporal reprojection.
 * @param instance RaytraceInstance to be used.
 */
extern "C" void update_temporal_matrix(RaytraceInstance* instance) {
  Quaternion q;
  device_gather_symbol(camera_rotation, q);

  Mat4x4 mat;
  memset(&mat, 0, sizeof(Mat4x4));

  mat.f11 = 1.0f - (q.y * q.y * 2.0f + q.z * q.z * 2.0f);
  mat.f12 = q.x * q.y * 2.0f + q.w * q.z * 2.0f;
  mat.f13 = q.x * q.z * 2.0f - q.w * q.y * 2.0f;
  mat.f21 = q.x * q.y * 2.0f - q.w * q.z * 2.0f;
  mat.f22 = 1.0f - (q.x * q.x * 2.0f + q.z * q.z * 2.0f);
  mat.f23 = q.y * q.z * 2.0f + q.w * q.x * 2.0f;
  mat.f31 = q.x * q.z * 2.0f + q.w * q.y * 2.0f;
  mat.f32 = q.y * q.z * 2.0f - q.w * q.x * 2.0f;
  mat.f33 = 1.0f - (q.x * q.x * 2.0f + q.y * q.y * 2.0f);

  const vec3 offset = instance->scene_gpu.camera.pos;

  mat.f14 = -(offset.x * mat.f11 + offset.y * mat.f12 + offset.z * mat.f13);
  mat.f24 = -(offset.x * mat.f21 + offset.y * mat.f22 + offset.z * mat.f23);
  mat.f34 = -(offset.x * mat.f31 + offset.y * mat.f32 + offset.z * mat.f33);
  mat.f44 = 1.0f;

  device_update_symbol(view_space, mat);

  const float step = 2.0f * (instance->scene_gpu.camera.fov / instance->width);
  const float vfov = step * instance->height / 2.0f;

  memset(&mat, 0, sizeof(Mat4x4));

  const float z_far  = instance->scene_gpu.camera.far_clip_distance;
  const float z_near = 1.0f;

  mat.f11 = 1.0f / instance->scene_gpu.camera.fov;
  mat.f22 = 1.0f / vfov;
  mat.f33 = -(z_far + z_near) / (z_far - z_near);
  mat.f43 = -1.0f;
  mat.f34 = -(2.0f * z_far * z_near) / (z_far - z_near);

  device_update_symbol(projection, mat);
}

/*
 * Computes value in halton sequence.
 * @param index Index in halton sequence.
 * @param base Base of halton sequence.
 * @result Value in halton sequence of base at index.
 */
static float halton(int index, int base) {
  float fraction = 1.0f;
  float result   = 0.0f;

  while (index > 0) {
    fraction /= base;
    result += fraction * (index % base);
    index = index / base;
  }

  return result;
}

/*
 * Updates the uniform per pixel jitter. Uploads updated jitter to GPU.
 * @param instance RaytraceInstance to be used.
 */
void update_jitter(RaytraceInstance* instance) {
  Jitter jitter;

  jitter.prev_x = instance->jitter.x;
  jitter.prev_y = instance->jitter.y;
  jitter.x      = halton(instance->temporal_frames, 2);
  jitter.y      = halton(instance->temporal_frames, 3);

  instance->jitter = jitter;
  device_update_symbol(jitter, jitter);
}

/*
 * Computes a rotation quaternion from euler angles.
 * @param rotation Euler angles defining the rotation.
 * @result Rotation quaternion
 */
static Quaternion get_rotation_quaternion(const vec3 rotation) {
  const float alpha = rotation.x;
  const float beta  = rotation.y;
  const float gamma = rotation.z;

  const float cy = cosf(gamma * 0.5f);
  const float sy = sinf(gamma * 0.5f);
  const float cp = cosf(beta * 0.5f);
  const float sp = sinf(beta * 0.5f);
  const float cr = cosf(alpha * 0.5f);
  const float sr = sinf(alpha * 0.5f);

  Quaternion q;
  q.w = cr * cp * cy + sr * sp * sy;
  q.x = sr * cp * cy - cr * sp * sy;
  q.y = cr * sp * cy + sr * cp * sy;
  q.z = cr * cp * sy - sr * sp * cy;

  return q;
}

/*
 * Computes and uploads to GPU camera rotation quaternions and fov step increments.
 * @param scene Scene which contains camera information.
 * @param width Number of pixel in horizontal in internal render buffer.
 * @param height Number of pixel in vertical in internal render buffer.
 */
static void update_camera_pos(const Scene scene, const unsigned int width, const unsigned int height) {
  const Quaternion q = get_rotation_quaternion(scene.camera.rotation);
  device_update_symbol(camera_rotation, q);

  const float step = 2.0f * (scene.camera.fov / width);
  const float vfov = step * height / 2.0f;

  device_update_symbol(step, step);
  device_update_symbol(vfov, vfov);
}

/*
 * Positions toy in front of the camera.
 * @param instance RaytraceInstance to be used.
 */
extern "C" void center_toy_at_camera(RaytraceInstance* instance) {
  const Quaternion q = get_rotation_quaternion(instance->scene_gpu.camera.rotation);

  vec3 offset;
  offset.x = 0.0f;
  offset.y = 0.0f;
  offset.z = -1.0f;

  offset = rotate_vector_by_quaternion(offset, q);

  offset = scale_vector(offset, 3.0f * instance->scene_gpu.toy.scale);

  instance->scene_gpu.toy.position = add_vector(instance->scene_gpu.camera.pos, offset);
}

/*
 * Positions toy behind the camera.
 * @param instance RaytraceInstance to be used.
 */
static void toy_flashlight_set_position(RaytraceInstance* instance) {
  const Quaternion q = get_rotation_quaternion(instance->scene_gpu.camera.rotation);

  vec3 offset;
  offset.x = 0.0f;
  offset.y = -0.5f;
  offset.z = 0.0f;

  offset = rotate_vector_by_quaternion(offset, q);

  instance->scene_gpu.toy.position = add_vector(instance->scene_gpu.camera.pos, offset);
}

/*
 * Allocates all pixel count related buffers.
 * @param instance RaytraceInstance to be used.
 */
extern "C" void allocate_buffers(RaytraceInstance* instance) {
  const unsigned int amount        = instance->width * instance->height;
  const unsigned int output_amount = instance->output_width * instance->output_height;

  device_update_symbol(width, instance->width);
  device_update_symbol(height, instance->height);
  device_update_symbol(output_width, instance->output_width);
  device_update_symbol(output_height, instance->output_height);
  device_update_symbol(max_ray_depth, instance->max_ray_depth);
  device_update_symbol(reservoir_size, instance->reservoir_size);
  device_update_symbol(amount, amount);
  device_update_symbol(denoiser, instance->denoiser);

  device_buffer_malloc(instance->frame_buffer, sizeof(RGBAhalf), amount);
  device_buffer_malloc(instance->frame_temporal, sizeof(RGBAhalf), amount);
  device_buffer_malloc(instance->frame_output, sizeof(RGBAhalf), output_amount);
  device_buffer_malloc(instance->frame_variance, sizeof(RGBAhalf), amount);
  device_buffer_malloc(instance->light_records, sizeof(RGBAhalf), amount);
  device_buffer_malloc(instance->bounce_records, sizeof(RGBAhalf), amount);

  if (instance->denoiser) {
    device_buffer_malloc(instance->albedo_buffer, sizeof(RGBAhalf), amount);
    device_buffer_malloc(instance->normal_buffer, sizeof(RGBAhalf), amount);
  }

  const int thread_count      = THREADS_PER_BLOCK * BLOCKS_PER_GRID;
  const int pixels_per_thread = (amount + thread_count - 1) / thread_count;
  const int max_task_count    = pixels_per_thread * thread_count;

  device_update_symbol(pixels_per_thread, pixels_per_thread);

  device_buffer_malloc(instance->light_trace, sizeof(TraceTask), max_task_count);
  device_buffer_malloc(instance->bounce_trace, sizeof(TraceTask), max_task_count);
  device_buffer_malloc(instance->light_trace_count, sizeof(uint16_t), thread_count);
  device_buffer_malloc(instance->bounce_trace_count, sizeof(uint16_t), thread_count);

  device_buffer_malloc(instance->trace_results, sizeof(TraceResult), max_task_count);
  device_buffer_malloc(instance->task_counts, sizeof(uint16_t), 6 * thread_count);
  device_buffer_malloc(instance->task_offsets, sizeof(uint16_t), 5 * thread_count);
  device_buffer_malloc(instance->randoms, sizeof(curandStateXORWOW_t), thread_count);

  device_buffer_malloc(instance->light_sample_history, sizeof(uint32_t), amount);
  device_buffer_malloc(instance->raydir_buffer, sizeof(vec3), amount);
  device_buffer_malloc(instance->trace_result_buffer, sizeof(TraceResult), amount);
  device_buffer_malloc(instance->state_buffer, sizeof(uint8_t), amount);

  device_buffer_malloc(instance->light_samples_1, sizeof(LightSample), amount);

  if (instance->realtime || instance->light_resampling) {
    device_buffer_malloc(instance->light_samples_2, sizeof(LightSample), amount);
    device_buffer_malloc(instance->light_eval_data, sizeof(LightEvalData), amount);
  }

  cudaMemset(device_buffer_get_pointer(instance->trace_result_buffer), 0, sizeof(TraceResult) * amount);
}

extern "C" void reset_raytracing(RaytraceInstance* instance) {
  device_bloom_free_mips(instance);

  // If denoising was set to on but no denoise setup exists, then we do not create one either now
  const int allocate_denoise = instance->settings.denoiser && !(instance->denoiser && !instance->denoise_setup);

  if (instance->denoise_setup) {
    optix_denoise_free(instance);
  }

  instance->width          = instance->settings.width;
  instance->height         = instance->settings.height;
  instance->output_width   = instance->settings.width;
  instance->output_height  = instance->settings.height;
  instance->max_ray_depth  = instance->settings.max_ray_depth;
  instance->denoiser       = instance->settings.denoiser;
  instance->reservoir_size = instance->settings.reservoir_size;

  if (instance->denoiser == DENOISING_UPSCALING) {
    if (instance->width * instance->height > 18144000) {
      error_message(
        "Internal resolution is too high for denoising with upscaling! The maximum is ~18144000 pixels. Upscaling is turned off!");
      instance->denoiser = DENOISING_ON;
    }
    else {
      instance->output_width *= 2;
      instance->output_height *= 2;
    }
  }

  raytrace_update_light_resampling_active(instance);
  allocate_buffers(instance);
  device_bloom_allocate_mips(instance);
  update_device_pointers(instance);
  initialize_randoms<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
  prepare_trace(instance);
  update_temporal_matrix(instance);

  if (allocate_denoise) {
    optix_denoise_create(instance);
  }

  log_message("Reset raytrace instance.");
}

extern "C" void initialize_device() {
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

extern "C" void update_device_scene(RaytraceInstance* instance) {
  device_update_symbol(scene, instance->scene_gpu);
}

/*
 * Updates and uploades all dynamic data. Must be called at least once before using trace.
 * @param instance RaytraceInstance to be used.
 */
extern "C" void prepare_trace(RaytraceInstance* instance) {
  if (instance->scene_gpu.toy.flashlight_mode) {
    toy_flashlight_set_position(instance);
  }
  update_device_scene(instance);
  device_update_symbol(shading_mode, instance->shading_mode);
  update_special_lights(instance->scene_gpu);
  update_camera_pos(instance->scene_gpu, instance->width, instance->height);
  update_jitter(instance);
  device_update_symbol(accum_mode, instance->accum_mode);
  device_update_symbol(spatial_samples, instance->spatial_samples);
}

static void bind_type(RaytraceInstance* instance, int type) {
  device_update_symbol(iteration_type, type);

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

static void execute_kernels(RaytraceInstance* instance, int type) {
  bind_type(instance, type);

  if (type != TYPE_CAMERA)
    balance_trace_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();

  preprocess_trace_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();

  process_trace_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();

  if (instance->scene_gpu.ocean.active && type != TYPE_LIGHT) {
    ocean_depth_trace_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
  }

  fog_preprocess_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();

  process_sky_inscattering_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();

  if (instance->scene_gpu.sky.cloud.active) {
    clouds_render_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
  }

  postprocess_trace_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();

  if (type != TYPE_LIGHT && instance->light_resampling) {
    generate_light_samples<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
    if (type == TYPE_CAMERA && instance->scene_gpu.material.lights_active) {
      LightSample* light_samples_1 = (LightSample*) device_buffer_get_pointer(instance->light_samples_1);
      LightSample* light_samples_2 = (LightSample*) device_buffer_get_pointer(instance->light_samples_2);
      for (int i = 0; i < instance->spatial_iterations; i++) {
        spatial_resampling<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(light_samples_1, light_samples_2);
        spatial_resampling<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(light_samples_2, light_samples_1);
      }
    }
  }

  process_geometry_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();

  if (instance->scene_gpu.ocean.active) {
    process_ocean_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
  }

  process_sky_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();

  if (instance->scene_gpu.toy.active) {
    process_toy_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
  }

  if (type != TYPE_LIGHT) {
    process_fog_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
  }
}

static void execute_debug_kernels(RaytraceInstance* instance, int type) {
  bind_type(instance, type);

  preprocess_trace_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();

  process_trace_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();

  if (instance->scene_gpu.ocean.active && type != TYPE_LIGHT) {
    ocean_depth_trace_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
  }
  postprocess_trace_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
  process_debug_geometry_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
  if (instance->scene_gpu.ocean.active) {
    process_debug_ocean_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
  }
  process_debug_sky_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
  if (instance->scene_gpu.toy.active) {
    process_debug_toy_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
  }
  process_debug_fog_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
}

extern "C" void trace_scene(RaytraceInstance* instance) {
  device_update_symbol(temporal_frames, instance->temporal_frames);

  raytrace_update_light_resampling_active(instance);

  generate_trace_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();

  if (instance->shading_mode) {
    execute_debug_kernels(instance, TYPE_CAMERA);
  }
  else {
    execute_kernels(instance, TYPE_CAMERA);
    execute_kernels(instance, TYPE_LIGHT);
    for (int i = 0; i < instance->max_ray_depth; i++) {
      execute_kernels(instance, TYPE_BOUNCE);
      execute_kernels(instance, TYPE_LIGHT);
    }
  }

  switch (instance->accum_mode) {
    case NO_ACCUMULATION:
      device_buffer_copy(instance->frame_buffer, instance->frame_output);
      break;
    case TEMPORAL_ACCUMULATION:
      temporal_accumulation<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
      break;
    case TEMPORAL_REPROJECTION:
      device_buffer_copy(instance->frame_output, instance->frame_temporal);
      temporal_reprojection<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
      break;
    default:
      break;
  }

  update_temporal_matrix(instance);

  gpuErrchk(cudaDeviceSynchronize());
}

extern "C" void free_inputs(RaytraceInstance* instance) {
  gpuErrchk(cudaFree(instance->scene_gpu.texture_assignments));
  gpuErrchk(cudaFree(instance->scene_gpu.triangles));
  gpuErrchk(cudaFree(instance->scene_gpu.traversal_triangles));
  gpuErrchk(cudaFree(instance->scene_gpu.nodes));
  gpuErrchk(cudaFree(instance->scene_gpu.triangle_lights));
  device_buffer_free(instance->light_trace);
  device_buffer_free(instance->bounce_trace);
  device_buffer_free(instance->light_trace_count);
  device_buffer_free(instance->bounce_trace_count);
  device_buffer_free(instance->trace_results);
  device_buffer_free(instance->task_counts);
  device_buffer_free(instance->task_offsets);
  device_buffer_free(instance->frame_buffer);
  device_buffer_free(instance->frame_temporal);
  device_buffer_free(instance->frame_variance);
  device_buffer_free(instance->light_records);
  device_buffer_free(instance->bounce_records);
  device_buffer_free(instance->randoms);
  device_buffer_free(instance->light_sample_history);
  device_buffer_free(instance->raydir_buffer);
  device_buffer_free(instance->trace_result_buffer);
  device_buffer_free(instance->state_buffer);
  device_buffer_free(instance->light_samples_1);
  device_buffer_free(instance->light_samples_2);
  device_buffer_free(instance->light_eval_data);

  gpuErrchk(cudaDeviceSynchronize());
}

extern "C" void _device_update_symbol(const size_t offset, const void* src, const size_t size) {
  gpuErrchk(cudaMemcpyToSymbol(device, src, size, offset, cudaMemcpyHostToDevice));
}

extern "C" void _device_gather_symbol(void* dst, const size_t offset, size_t size) {
  gpuErrchk(cudaMemcpyFromSymbol(dst, device, size, offset, cudaMemcpyDeviceToHost));
}

extern "C" void free_outputs(RaytraceInstance* instance) {
  device_buffer_free(instance->frame_output);

  if (instance->denoiser) {
    device_buffer_free(instance->albedo_buffer);
    device_buffer_free(instance->normal_buffer);
  }

  device_bloom_free_mips(instance);

  free(instance);
}

extern "C" void device_initialize_random_generators() {
  initialize_randoms<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
}

extern "C" void initialize_8bit_frame(RaytraceInstance* instance, const unsigned int width, const unsigned int height) {
  device_buffer_malloc(instance->buffer_8bit, sizeof(XRGB8), width * height);
  update_device_pointers(instance);
}

extern "C" void free_8bit_frame(RaytraceInstance* instance) {
  device_buffer_free(instance->buffer_8bit);
}

extern "C" void copy_framebuffer_to_8bit(
  RGBAhalf* gpu_source, XRGB8* gpu_scratch, XRGB8* cpu_dest, const int width, const int height, const int ld) {
  convert_RGBhalf_to_XRGB8<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>((RGBAhalf*) gpu_source, gpu_scratch, width, height, ld);
  gpuErrchk(cudaMemcpy(cpu_dest, gpu_scratch, sizeof(XRGB8) * ld * height, cudaMemcpyDeviceToHost));
}

extern "C" void* memcpy_gpu_to_cpu(void* gpu_ptr, size_t size) {
  void* cpu_ptr;
  gpuErrchk(cudaMallocHost((void**) &(cpu_ptr), size));
  gpuErrchk(cudaMemcpy(cpu_ptr, gpu_ptr, size, cudaMemcpyDeviceToHost));
  return cpu_ptr;
}

extern "C" void* memcpy_texture_to_cpu(void* textures_ptr, uint64_t* count) {
  const uint64_t tex_count           = *count;
  const uint64_t header_element_size = 12;
  const uint64_t header_size         = header_element_size * tex_count;

  cudaTextureObject_t* tex_objects;
  gpuErrchk(cudaMallocHost((void**) &(tex_objects), sizeof(cudaTextureObject_t) * tex_count));
  gpuErrchk(cudaMemcpy(tex_objects, textures_ptr, sizeof(cudaTextureObject_t) * tex_count, cudaMemcpyDeviceToHost));

  cudaResourceDesc resource;
  uint64_t buffer_size = header_size;

  for (int i = 0; i < tex_count; i++) {
    cudaGetTextureObjectResourceDesc(&resource, tex_objects[i]);
    uint64_t pitch  = (uint64_t) resource.res.pitch2D.pitchInBytes;
    uint64_t height = (uint64_t) resource.res.pitch2D.height;
    buffer_size += (pitch + 32) * height;
  }

  uint8_t* cpu_ptr;
  gpuErrchk(cudaMallocHost((void**) &(cpu_ptr), buffer_size));

  uint64_t offset = header_size;

  for (int i = 0; i < tex_count; i++) {
    cudaGetTextureObjectResourceDesc(&resource, tex_objects[i]);
    size_t pitch    = resource.res.pitch2D.pitchInBytes;
    uint32_t width  = (uint32_t) resource.res.pitch2D.width;
    uint32_t height = (uint32_t) resource.res.pitch2D.height;
    void* source    = resource.res.pitch2D.devPtr;
    gpuErrchk(cudaMemcpy(cpu_ptr + offset, source, pitch * height, cudaMemcpyDeviceToHost));

    TextureRGBA tex;
    texture_create(&tex, width, height, 1, pitch / sizeof(RGBA8), (void*) (cpu_ptr + offset), TexDataUINT8, TexStorageCPU);

    uint32_t encoded_size;

    void* encoded_data = qoi_encode_RGBA8(&tex, (int*) &encoded_size);

    memcpy(cpu_ptr + header_element_size * i + 0x00, &encoded_size, sizeof(uint32_t));
    memcpy(cpu_ptr + header_element_size * i + 0x04, &width, sizeof(uint32_t));
    memcpy(cpu_ptr + header_element_size * i + 0x08, &height, sizeof(uint32_t));

    memcpy(cpu_ptr + offset, encoded_data, encoded_size);
    offset += encoded_size;

    free(encoded_data);
  }

  buffer_size = offset;

  gpuErrchk(cudaFreeHost(tex_objects));

  *count = buffer_size;

  return cpu_ptr;
}

extern "C" void update_device_pointers(RaytraceInstance* instance) {
  DevicePointers ptrs;

  ptrs.light_trace          = (TraceTask*) device_buffer_get_pointer(instance->light_trace);
  ptrs.bounce_trace         = (TraceTask*) device_buffer_get_pointer(instance->bounce_trace);
  ptrs.light_trace_count    = (uint16_t*) device_buffer_get_pointer(instance->light_trace_count);
  ptrs.bounce_trace_count   = (uint16_t*) device_buffer_get_pointer(instance->bounce_trace_count);
  ptrs.trace_results        = (TraceResult*) device_buffer_get_pointer(instance->trace_results);
  ptrs.task_counts          = (uint16_t*) device_buffer_get_pointer(instance->task_counts);
  ptrs.task_offsets         = (uint16_t*) device_buffer_get_pointer(instance->task_offsets);
  ptrs.light_sample_history = (uint32_t*) device_buffer_get_pointer(instance->light_sample_history);
  ptrs.frame_output         = (RGBAhalf*) device_buffer_get_pointer(instance->frame_output);
  ptrs.frame_temporal       = (RGBAhalf*) device_buffer_get_pointer(instance->frame_temporal);
  ptrs.frame_buffer         = (RGBAhalf*) device_buffer_get_pointer(instance->frame_buffer);
  ptrs.frame_variance       = (RGBAhalf*) device_buffer_get_pointer(instance->frame_variance);
  ptrs.albedo_buffer        = (RGBAhalf*) device_buffer_get_pointer(instance->albedo_buffer);
  ptrs.normal_buffer        = (RGBAhalf*) device_buffer_get_pointer(instance->normal_buffer);
  ptrs.light_records        = (RGBAhalf*) device_buffer_get_pointer(instance->light_records);
  ptrs.bounce_records       = (RGBAhalf*) device_buffer_get_pointer(instance->bounce_records);
  ptrs.buffer_8bit          = (XRGB8*) device_buffer_get_pointer(instance->buffer_8bit);
  ptrs.albedo_atlas         = (cudaTextureObject_t*) device_buffer_get_pointer(instance->tex_atlas.albedo);
  ptrs.illuminance_atlas    = (cudaTextureObject_t*) device_buffer_get_pointer(instance->tex_atlas.illuminance);
  ptrs.material_atlas       = (cudaTextureObject_t*) device_buffer_get_pointer(instance->tex_atlas.material);
  ptrs.normal_atlas         = (cudaTextureObject_t*) device_buffer_get_pointer(instance->tex_atlas.normal);
  ptrs.cloud_noise          = (cudaTextureObject_t*) device_buffer_get_pointer(instance->cloud_noise);
  ptrs.randoms              = (curandStateXORWOW_t*) device_buffer_get_pointer(instance->randoms);
  ptrs.raydir_buffer        = (vec3*) device_buffer_get_pointer(instance->raydir_buffer);
  ptrs.trace_result_buffer  = (TraceResult*) device_buffer_get_pointer(instance->trace_result_buffer);
  ptrs.state_buffer         = (uint8_t*) device_buffer_get_pointer(instance->state_buffer);
  ptrs.light_samples        = (LightSample*) device_buffer_get_pointer(instance->light_samples_1);
  ptrs.light_eval_data      = (LightEvalData*) device_buffer_get_pointer(instance->light_eval_data);
  ptrs.sky_tm_luts          = (cudaTextureObject_t*) device_buffer_get_pointer(instance->sky_tm_luts);
  ptrs.sky_ms_luts          = (cudaTextureObject_t*) device_buffer_get_pointer(instance->sky_ms_luts);

  device_update_symbol(ptrs, ptrs);
  log_message("Updated device pointers.");
}

extern "C" void free_host_memory(void* ptr) {
  gpuErrchk(cudaFreeHost(ptr));
}
