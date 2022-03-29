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
#include "cuda/bvh.cuh"
#include "cuda/cloudgen.cuh"
#include "cuda/denoise.cuh"
#include "cuda/directives.cuh"
#include "cuda/kernels.cuh"
#include "cuda/light.cuh"
#include "cuda/math.cuh"
#include "cuda/random.cuh"
#include "cuda/sky.cuh"
#include "cuda/utils.cuh"
#include "utils.h"
#include "image.h"
#include "log.h"
#include "mesh.h"
#include "primitives.h"
#include "raytrace.h"
#include "scene.h"

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

  gpuErrchk(cudaMemcpyToSymbol(device_sun, &(sun), sizeof(vec3), 0, cudaMemcpyHostToDevice));

  vec3 moon              = angles_to_direction(scene.sky.moon_altitude, scene.sky.moon_azimuth);
  const float scale_moon = 1.0f / (sqrtf(moon.x * moon.x + moon.y * moon.y + moon.z * moon.z));
  moon.x *= scale_moon * SKY_MOON_DISTANCE;
  moon.y *= scale_moon * SKY_MOON_DISTANCE;
  moon.z *= scale_moon * SKY_MOON_DISTANCE;
  moon.y -= SKY_EARTH_RADIUS;
  moon.x -= scene.sky.geometry_offset.x;
  moon.y -= scene.sky.geometry_offset.y;
  moon.z -= scene.sky.geometry_offset.z;

  gpuErrchk(cudaMemcpyToSymbol(device_moon, &(moon), sizeof(vec3), 0, cudaMemcpyHostToDevice));
}

/*
 * Computes matrices used for temporal reprojection.
 * @param instance RaytraceInstance to be used.
 */
extern "C" void update_temporal_matrix(RaytraceInstance* instance) {
  Quaternion q;
  gpuErrchk(cudaMemcpyFromSymbol(&q, device_camera_rotation, sizeof(Quaternion), 0, cudaMemcpyDeviceToHost));

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

  gpuErrchk(cudaMemcpyToSymbol(device_view_space, &(mat), sizeof(Mat4x4), 0, cudaMemcpyHostToDevice));

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

  gpuErrchk(cudaMemcpyToSymbol(device_projection, &(mat), sizeof(Mat4x4), 0, cudaMemcpyHostToDevice));
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
  gpuErrchk(cudaMemcpyToSymbol(device_jitter, &(instance->jitter), sizeof(Jitter), 0, cudaMemcpyHostToDevice));
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
  gpuErrchk(cudaMemcpyToSymbol(device_camera_rotation, &(q), sizeof(Quaternion), 0, cudaMemcpyHostToDevice));

  const float step = 2.0f * (scene.camera.fov / width);
  const float vfov = step * height / 2.0f;

  gpuErrchk(cudaMemcpyToSymbol(device_step, &(step), sizeof(float), 0, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpyToSymbol(device_vfov, &(vfov), sizeof(float), 0, cudaMemcpyHostToDevice));
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
 * Allocates all pixel count related buffers.
 * @param instance RaytraceInstance to be used.
 */
extern "C" void allocate_buffers(RaytraceInstance* instance) {
  const unsigned int amount = instance->width * instance->height;

  gpuErrchk(cudaMemcpyToSymbol(device_width, &(instance->width), sizeof(unsigned int), 0, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpyToSymbol(device_height, &(instance->height), sizeof(unsigned int), 0, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpyToSymbol(device_max_ray_depth, &(instance->max_ray_depth), sizeof(int), 0, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpyToSymbol(device_reservoir_size, &(instance->reservoir_size), sizeof(int), 0, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpyToSymbol(device_amount, &(amount), sizeof(unsigned int), 0, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpyToSymbol(device_denoiser, &(instance->denoiser), sizeof(int), 0, cudaMemcpyHostToDevice));

  device_buffer_malloc(instance->frame_buffer, sizeof(RGBF), amount);
  device_buffer_malloc(instance->frame_temporal, sizeof(RGBF), amount);
  device_buffer_malloc(instance->frame_output, sizeof(RGBF), amount);
  device_buffer_malloc(instance->frame_variance, sizeof(RGBF), amount);
  device_buffer_malloc(instance->frame_bias_cache, sizeof(RGBF), amount);
  device_buffer_malloc(instance->light_records, sizeof(RGBF), amount);
  device_buffer_malloc(instance->bounce_records, sizeof(RGBF), amount);

  if (instance->denoiser) {
    device_buffer_malloc(instance->albedo_buffer, sizeof(RGBF), amount);
  }

  /*
   * Due to transparency causing light rays to create more light rays and the fact
   * that pixels may be assigned to different threads in each iteration
   * we have to be able to handle twice the pixels per thread than normally necessary
   */
  const int thread_count      = THREADS_PER_BLOCK * BLOCKS_PER_GRID;
  const int pixels_per_thread = 2 * (amount + thread_count - 1) / thread_count;
  const int max_task_count    = pixels_per_thread * thread_count;

  gpuErrchk(cudaMemcpyToSymbol(device_pixels_per_thread, &(pixels_per_thread), sizeof(int), 0, cudaMemcpyHostToDevice));

  device_buffer_malloc(instance->light_trace, sizeof(TraceTask), max_task_count);
  device_buffer_malloc(instance->bounce_trace, sizeof(TraceTask), max_task_count);
  device_buffer_malloc(instance->light_trace_count, sizeof(uint16_t), thread_count);
  device_buffer_malloc(instance->bounce_trace_count, sizeof(uint16_t), thread_count);

  device_buffer_malloc(instance->trace_results, sizeof(TraceResult), max_task_count);
  device_buffer_malloc(instance->task_counts, sizeof(uint16_t), 5 * thread_count);
  device_buffer_malloc(instance->task_offsets, sizeof(uint16_t), 5 * thread_count);
  device_buffer_malloc(instance->randoms, sizeof(curandStateXORWOW_t), thread_count);

  device_buffer_malloc(instance->light_sample_history, sizeof(uint32_t), amount);
  device_buffer_malloc(instance->raydir_buffer, sizeof(vec3), amount);
  device_buffer_malloc(instance->trace_result_buffer, sizeof(TraceResult), amount);
  device_buffer_malloc(instance->trace_result_temporal, sizeof(TraceResult), amount);
  device_buffer_malloc(instance->state_buffer, sizeof(uint8_t), amount);

  device_buffer_malloc(instance->light_samples_1, sizeof(LightSample), amount);
  device_buffer_malloc(instance->light_samples_2, sizeof(LightSample), amount);
  device_buffer_malloc(instance->light_eval_data, sizeof(LightEvalData), amount);

  cudaMemset(device_buffer_get_pointer(instance->trace_result_buffer), 0, sizeof(TraceResult) * amount);
}

void generate_clouds(RaytraceInstance* instance) {
  bench_tic();

  if (instance->scene_gpu.sky.cloud.initialized) {
    device_free(instance->scene_gpu.sky.cloud.shape_noise, CLOUD_SHAPE_RES * CLOUD_SHAPE_RES * CLOUD_SHAPE_RES * 4 * sizeof(uint8_t));
    device_free(instance->scene_gpu.sky.cloud.detail_noise, CLOUD_DETAIL_RES * CLOUD_DETAIL_RES * CLOUD_DETAIL_RES * 4 * sizeof(uint8_t));
    device_free(instance->scene_gpu.sky.cloud.weather_map, CLOUD_WEATHER_RES * CLOUD_WEATHER_RES * 4 * sizeof(uint8_t));
    device_free(instance->scene_gpu.sky.cloud.curl_noise, CLOUD_CURL_RES * CLOUD_CURL_RES * 4 * sizeof(uint8_t));
  }

  device_malloc(
    (void**) &instance->scene_gpu.sky.cloud.shape_noise, CLOUD_SHAPE_RES * CLOUD_SHAPE_RES * CLOUD_SHAPE_RES * 4 * sizeof(uint8_t));
  generate_shape_noise<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(CLOUD_SHAPE_RES, instance->scene_gpu.sky.cloud.shape_noise);

  device_malloc(
    (void**) &instance->scene_gpu.sky.cloud.detail_noise, CLOUD_DETAIL_RES * CLOUD_DETAIL_RES * CLOUD_DETAIL_RES * 4 * sizeof(uint8_t));
  generate_detail_noise<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(CLOUD_DETAIL_RES, instance->scene_gpu.sky.cloud.detail_noise);

  device_malloc((void**) &instance->scene_gpu.sky.cloud.weather_map, CLOUD_WEATHER_RES * CLOUD_WEATHER_RES * 4 * sizeof(uint8_t));
  generate_weather_map<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(
    CLOUD_WEATHER_RES, (float) instance->scene_gpu.sky.cloud.seed, instance->scene_gpu.sky.cloud.weather_map);

  device_malloc((void**) &instance->scene_gpu.sky.cloud.curl_noise, CLOUD_CURL_RES * CLOUD_CURL_RES * 4 * sizeof(uint8_t));
  generate_curl_noise<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(CLOUD_CURL_RES, instance->scene_gpu.sky.cloud.curl_noise);

  instance->scene_gpu.sky.cloud.initialized = 1;

  bench_toc((char*) "Cloud Noise Generation");
}

extern "C" RaytraceInstance* init_raytracing(
  General general, DeviceBuffer* albedo_atlas, int albedo_atlas_length, DeviceBuffer* illuminance_atlas, int illuminance_atlas_length,
  DeviceBuffer* material_atlas, int material_atlas_length, Scene scene) {
  RaytraceInstance* instance = (RaytraceInstance*) malloc(sizeof(RaytraceInstance));
  memset(instance, 0, sizeof(RaytraceInstance));

  instance->width  = general.width;
  instance->height = general.height;

  instance->max_ray_depth   = general.max_ray_depth;
  instance->offline_samples = general.samples;
  instance->denoiser        = general.denoiser;
  instance->reservoir_size  = general.reservoir_size;

  instance->albedo_atlas      = albedo_atlas;
  instance->illuminance_atlas = illuminance_atlas;
  instance->material_atlas    = material_atlas;

  instance->albedo_atlas_length      = albedo_atlas_length;
  instance->illuminance_atlas_length = illuminance_atlas_length;
  instance->material_atlas_length    = material_atlas_length;

  instance->scene_gpu          = scene;
  instance->settings           = general;
  instance->shading_mode       = 0;
  instance->accum_mode         = TEMPORAL_ACCUMULATION;
  instance->spatial_samples    = 5;
  instance->spatial_iterations = 2;

  device_buffer_init(&instance->light_trace);
  device_buffer_init(&instance->bounce_trace);
  device_buffer_init(&instance->light_trace_count);
  device_buffer_init(&instance->bounce_trace_count);
  device_buffer_init(&instance->trace_results);
  device_buffer_init(&instance->task_counts);
  device_buffer_init(&instance->task_offsets);
  device_buffer_init(&instance->light_sample_history);
  device_buffer_init(&instance->frame_output);
  device_buffer_init(&instance->frame_temporal);
  device_buffer_init(&instance->frame_buffer);
  device_buffer_init(&instance->frame_variance);
  device_buffer_init(&instance->frame_bias_cache);
  device_buffer_init(&instance->albedo_buffer);
  device_buffer_init(&instance->light_records);
  device_buffer_init(&instance->bounce_records);
  device_buffer_init(&instance->buffer_8bit);
  device_buffer_init(&instance->randoms);
  device_buffer_init(&instance->raydir_buffer);
  device_buffer_init(&instance->trace_result_buffer);
  device_buffer_init(&instance->trace_result_temporal);
  device_buffer_init(&instance->state_buffer);
  device_buffer_init(&instance->light_samples_1);
  device_buffer_init(&instance->light_samples_2);
  device_buffer_init(&instance->light_eval_data);

  device_buffer_malloc(instance->buffer_8bit, sizeof(XRGB8), instance->width * instance->height);

  device_malloc((void**) &(instance->scene_gpu.texture_assignments), sizeof(TextureAssignment) * scene.materials_length);
  device_malloc((void**) &(instance->scene_gpu.triangles), sizeof(Triangle) * instance->scene_gpu.triangles_length);
  device_malloc((void**) &(instance->scene_gpu.traversal_triangles), sizeof(TraversalTriangle) * instance->scene_gpu.triangles_length);
  device_malloc((void**) &(instance->scene_gpu.nodes), sizeof(Node8) * instance->scene_gpu.nodes_length);
  device_malloc((void**) &(instance->scene_gpu.triangle_lights), sizeof(TriangleLight) * instance->scene_gpu.triangle_lights_length);

  gpuErrchk(cudaMemcpy(
    instance->scene_gpu.texture_assignments, scene.texture_assignments, sizeof(TextureAssignment) * scene.materials_length,
    cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(instance->scene_gpu.triangles, scene.triangles, sizeof(Triangle) * scene.triangles_length, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(
    instance->scene_gpu.traversal_triangles, scene.traversal_triangles, sizeof(TraversalTriangle) * scene.triangles_length,
    cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(instance->scene_gpu.nodes, scene.nodes, sizeof(Node8) * scene.nodes_length, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(
    instance->scene_gpu.triangle_lights, scene.triangle_lights, sizeof(TriangleLight) * scene.triangle_lights_length,
    cudaMemcpyHostToDevice));

  gpuErrchk(cudaMemcpyToSymbol(
    device_texture_assignments, &(instance->scene_gpu.texture_assignments), sizeof(TextureAssignment*), 0, cudaMemcpyHostToDevice));

  allocate_buffers(instance);
  allocate_bloom_mips(instance);
  update_device_pointers(instance);
  initialize_randoms<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
  prepare_trace(instance);
  update_temporal_matrix(instance);

  instance->snap_resolution = SNAP_RESOLUTION_RENDER;

  return instance;
}

extern "C" void reset_raytracing(RaytraceInstance* instance) {
  free_bloom_mips(instance);

  if (instance->denoiser) {
    free_realtime_denoise(instance, instance->denoise_setup);
  }

  instance->width          = instance->settings.width;
  instance->height         = instance->settings.height;
  instance->max_ray_depth  = instance->settings.max_ray_depth;
  instance->denoiser       = instance->settings.denoiser;
  instance->reservoir_size = instance->settings.reservoir_size;

  allocate_buffers(instance);
  allocate_bloom_mips(instance);
  update_device_pointers(instance);
  initialize_randoms<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
  prepare_trace(instance);
  update_temporal_matrix(instance);

  if (instance->denoiser) {
    instance->denoise_setup = initialize_optix_denoise_for_realtime(instance);
  }

  log_message("Reset raytrace instance.");
}

extern "C" DeviceBuffer* initialize_textures(TextureRGBA* textures, const int textures_length) {
  cudaTextureObject_t* textures_cpu = (cudaTextureObject_t*) malloc(sizeof(cudaTextureObject_t) * textures_length);
  DeviceBuffer* textures_gpu;

  device_buffer_init(&textures_gpu);
  device_buffer_malloc(textures_gpu, sizeof(cudaTextureObject_t), textures_length);

  for (int i = 0; i < textures_length; i++) {
    TextureRGBA texture = textures[i];

    const size_t pixel_size = (texture.type == TexDataFP32) ? sizeof(RGBAF) : sizeof(RGBA8);

    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0]   = cudaAddressModeWrap;
    texDesc.addressMode[1]   = cudaAddressModeWrap;
    texDesc.filterMode       = cudaFilterModeLinear;
    texDesc.maxAnisotropy    = 16;
    texDesc.readMode         = (texture.type == TexDataFP32) ? cudaReadModeElementType : cudaReadModeNormalizedFloat;
    texDesc.normalizedCoords = 1;

    const int num_rows = texture.height;
    const int num_cols = texture.width;
    void* data         = (void*) texture.data;
    void* data_gpu;
    size_t pitch = device_malloc_pitch((void**) &data_gpu, num_cols * pixel_size, num_rows);
    gpuErrchk(cudaMemcpy2D(data_gpu, pitch, data, num_cols * pixel_size, num_cols * pixel_size, num_rows, cudaMemcpyHostToDevice));

    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType                  = cudaResourceTypePitch2D;
    resDesc.res.pitch2D.devPtr       = data_gpu;
    resDesc.res.pitch2D.width        = num_cols;
    resDesc.res.pitch2D.height       = num_rows;
    resDesc.res.pitch2D.desc         = cudaCreateChannelDesc<uchar4>();
    resDesc.res.pitch2D.pitchInBytes = pitch;

    gpuErrchk(cudaCreateTextureObject(textures_cpu + i, &resDesc, &texDesc, NULL));
  }

  device_buffer_upload(textures_gpu, textures_cpu);

  free(textures_cpu);

  return textures_gpu;
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
  print_info("Copyright (c) 2022 MilchRatchet");
}

extern "C" void free_textures_atlas(DeviceBuffer* texture_atlas, const int textures_length) {
  cudaTextureObject_t* textures_cpu = (cudaTextureObject_t*) malloc(device_buffer_get_size(texture_atlas));
  device_buffer_download_full(texture_atlas, textures_cpu);

  for (int i = 0; i < textures_length; i++) {
    gpuErrchk(cudaDestroyTextureObject(textures_cpu[i]));
  }

  device_buffer_destroy(texture_atlas);
  free(textures_cpu);
}

/*
 * Updates and uploades all dynamic data. Must be called at least once before using trace.
 * @param instance RaytraceInstance to be used.
 */
extern "C" void prepare_trace(RaytraceInstance* instance) {
  gpuErrchk(cudaMemcpyToSymbol(device_scene, &(instance->scene_gpu), sizeof(Scene), 0, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpyToSymbol(device_shading_mode, &(instance->shading_mode), sizeof(unsigned int), 0, cudaMemcpyHostToDevice));
  device_buffer_copy(instance->trace_result_buffer, instance->trace_result_temporal);
  update_special_lights(instance->scene_gpu);
  update_camera_pos(instance->scene_gpu, instance->width, instance->height);
  update_jitter(instance);
  gpuErrchk(cudaMemcpyToSymbol(device_accum_mode, &(instance->accum_mode), sizeof(int), 0, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpyToSymbol(device_spatial_samples, &(instance->spatial_samples), sizeof(int), 0, cudaMemcpyHostToDevice));
}

static void bind_type(RaytraceInstance* instance, int type) {
  gpuErrchk(cudaMemcpyToSymbol(device_iteration_type, &(type), sizeof(int), 0, cudaMemcpyHostToDevice));

  switch (type) {
    case TYPE_CAMERA:
    case TYPE_BOUNCE:
      gpuErrchk(
        cudaMemcpyToSymbol(device_trace_tasks, &(instance->bounce_trace->device_pointer), sizeof(TraceTask*), 0, cudaMemcpyHostToDevice));
      gpuErrchk(cudaMemcpyToSymbol(
        device_trace_count, &(instance->bounce_trace_count->device_pointer), sizeof(uint16_t*), 0, cudaMemcpyHostToDevice));
      gpuErrchk(
        cudaMemcpyToSymbol(device_records, &(instance->bounce_records->device_pointer), sizeof(uint16_t*), 0, cudaMemcpyHostToDevice));
      break;
    case TYPE_LIGHT:
      gpuErrchk(
        cudaMemcpyToSymbol(device_trace_tasks, &(instance->light_trace->device_pointer), sizeof(TraceTask*), 0, cudaMemcpyHostToDevice));
      gpuErrchk(cudaMemcpyToSymbol(
        device_trace_count, &(instance->light_trace_count->device_pointer), sizeof(uint16_t*), 0, cudaMemcpyHostToDevice));
      gpuErrchk(
        cudaMemcpyToSymbol(device_records, &(instance->light_records->device_pointer), sizeof(uint16_t*), 0, cudaMemcpyHostToDevice));
      break;
  }
}

static void execute_kernels(RaytraceInstance* instance, int type) {
  bind_type(instance, type);

  if (type != TYPE_CAMERA)
    balance_trace_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
  preprocess_trace_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
  process_trace_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
  process_volumetrics_trace_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
  postprocess_trace_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
  if (type != TYPE_LIGHT) {
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
  process_ocean_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
  process_sky_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
  process_toy_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
  if (type != TYPE_LIGHT) {
    process_fog_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
  }
}

static void execute_debug_kernels(RaytraceInstance* instance, int type) {
  bind_type(instance, type);

  preprocess_trace_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
  process_trace_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
  postprocess_trace_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
  process_debug_geometry_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
  process_debug_ocean_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
  process_debug_sky_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
  process_debug_toy_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
  process_debug_fog_tasks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
}

extern "C" void trace_scene(RaytraceInstance* instance) {
  gpuErrchk(cudaMemcpyToSymbol(device_temporal_frames, &(instance->temporal_frames), sizeof(int), 0, cudaMemcpyHostToDevice));

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
      update_temporal_matrix(instance);
      break;
    default:
      break;
  }

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
  device_buffer_free(instance->frame_bias_cache);
  device_buffer_free(instance->light_records);
  device_buffer_free(instance->bounce_records);
  device_buffer_free(instance->randoms);
  device_buffer_free(instance->light_sample_history);
  device_buffer_free(instance->raydir_buffer);
  device_buffer_free(instance->trace_result_buffer);
  device_buffer_free(instance->trace_result_temporal);
  device_buffer_free(instance->state_buffer);
  device_buffer_free(instance->light_samples_1);
  device_buffer_free(instance->light_samples_2);
  device_buffer_free(instance->light_eval_data);

  gpuErrchk(cudaDeviceSynchronize());
}

extern "C" void free_outputs(RaytraceInstance* instance) {
  device_buffer_free(instance->frame_output);

  if (instance->denoiser) {
    device_buffer_free(instance->albedo_buffer);
  }

  free_bloom_mips(instance);

  free(instance);
}

extern "C" void initialize_8bit_frame(RaytraceInstance* instance, const int width, const int height) {
  device_buffer_malloc(instance->buffer_8bit, sizeof(XRGB8), width * height);
  update_device_pointers(instance);
}

extern "C" void free_8bit_frame(RaytraceInstance* instance) {
  device_buffer_free(instance->buffer_8bit);
}

extern "C" void copy_framebuffer_to_8bit(XRGB8* buffer, const int width, const int height, RGBF* source, RaytraceInstance* instance) {
  convert_RGBF_to_XRGB8<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(width, height, source);
  device_buffer_download(instance->buffer_8bit, buffer, sizeof(XRGB8) * width * height);
}

extern "C" void* memcpy_gpu_to_cpu(void* gpu_ptr, size_t size) {
  void* cpu_ptr;
  gpuErrchk(cudaMallocHost((void**) &(cpu_ptr), size));
  gpuErrchk(cudaMemcpy(cpu_ptr, gpu_ptr, size, cudaMemcpyDeviceToHost));
  return cpu_ptr;
}

extern "C" void* memcpy_texture_to_cpu(void* textures_ptr, uint64_t* count) {
  const uint64_t tex_count           = *count;
  const uint32_t header_element_size = 32;
  const uint64_t header_size         = header_element_size * tex_count;

  uint8_t* header = (uint8_t*) malloc(header_size);

  cudaTextureObject_t* tex_objects;
  gpuErrchk(cudaMallocHost((void**) &(tex_objects), sizeof(cudaTextureObject_t) * tex_count));
  gpuErrchk(cudaMemcpy(tex_objects, textures_ptr, sizeof(cudaTextureObject_t) * tex_count, cudaMemcpyDeviceToHost));

  cudaResourceDesc resource;
  cudaTextureDesc texture_desc;
  uint64_t buffer_size = header_size;

  for (int i = 0; i < tex_count; i++) {
    cudaGetTextureObjectResourceDesc(&resource, tex_objects[i]);
    cudaGetTextureObjectTextureDesc(&texture_desc, tex_objects[i]);
    uint32_t width      = (uint32_t) resource.res.pitch2D.width;
    uint32_t height     = (uint32_t) resource.res.pitch2D.height;
    uint32_t pixel_size = (uint32_t) (texture_desc.readMode == cudaReadModeElementType) ? sizeof(RGBAF) : sizeof(RGBA8);
    memcpy(header + i * header_element_size, &buffer_size, 8);
    memcpy(header + i * header_element_size + 8, &width, 4);
    memcpy(header + i * header_element_size + 12, &height, 4);
    memcpy(header + i * header_element_size + 16, &pixel_size, 4);
    buffer_size += pixel_size * width * height;
  }

  uint8_t* cpu_ptr;
  gpuErrchk(cudaMallocHost((void**) &(cpu_ptr), buffer_size));

  memcpy(cpu_ptr, header, header_size);
  free(header);

  uint64_t offset = header_size;

  for (int i = 0; i < tex_count; i++) {
    cudaGetTextureObjectResourceDesc(&resource, tex_objects[i]);
    cudaGetTextureObjectTextureDesc(&texture_desc, tex_objects[i]);
    size_t pitch        = resource.res.pitch2D.pitchInBytes;
    size_t width        = resource.res.pitch2D.width;
    size_t height       = resource.res.pitch2D.height;
    uint32_t pixel_size = (uint32_t) (texture_desc.readMode == cudaReadModeElementType) ? sizeof(RGBAF) : sizeof(RGBA8);
    uint8_t* source     = (uint8_t*) resource.res.pitch2D.devPtr;
    for (int j = 0; j < height; j++) {
      gpuErrchk(cudaMemcpy(cpu_ptr + offset, source + j * pitch, pixel_size * width, cudaMemcpyDeviceToHost));
      offset += pixel_size * width;
    }
  }

  gpuErrchk(cudaFreeHost(tex_objects));

  *count = buffer_size;

  return cpu_ptr;
}

extern "C" void update_device_pointers(RaytraceInstance* instance) {
  DevicePointers ptrs;

  ptrs.light_trace           = (TraceTask*) device_buffer_get_pointer(instance->light_trace);
  ptrs.bounce_trace          = (TraceTask*) device_buffer_get_pointer(instance->bounce_trace);
  ptrs.light_trace_count     = (uint16_t*) device_buffer_get_pointer(instance->light_trace_count);
  ptrs.bounce_trace_count    = (uint16_t*) device_buffer_get_pointer(instance->bounce_trace_count);
  ptrs.trace_results         = (TraceResult*) device_buffer_get_pointer(instance->trace_results);
  ptrs.task_counts           = (uint16_t*) device_buffer_get_pointer(instance->task_counts);
  ptrs.task_offsets          = (uint16_t*) device_buffer_get_pointer(instance->task_offsets);
  ptrs.light_sample_history  = (uint32_t*) device_buffer_get_pointer(instance->light_sample_history);
  ptrs.frame_output          = (RGBF*) device_buffer_get_pointer(instance->frame_output);
  ptrs.frame_temporal        = (RGBF*) device_buffer_get_pointer(instance->frame_temporal);
  ptrs.frame_buffer          = (RGBF*) device_buffer_get_pointer(instance->frame_buffer);
  ptrs.frame_variance        = (RGBF*) device_buffer_get_pointer(instance->frame_variance);
  ptrs.frame_bias_cache      = (RGBF*) device_buffer_get_pointer(instance->frame_bias_cache);
  ptrs.albedo_buffer         = (RGBF*) device_buffer_get_pointer(instance->albedo_buffer);
  ptrs.light_records         = (RGBF*) device_buffer_get_pointer(instance->light_records);
  ptrs.bounce_records        = (RGBF*) device_buffer_get_pointer(instance->bounce_records);
  ptrs.buffer_8bit           = (XRGB8*) device_buffer_get_pointer(instance->buffer_8bit);
  ptrs.albedo_atlas          = (cudaTextureObject_t*) device_buffer_get_pointer(instance->albedo_atlas);
  ptrs.illuminance_atlas     = (cudaTextureObject_t*) device_buffer_get_pointer(instance->illuminance_atlas);
  ptrs.material_atlas        = (cudaTextureObject_t*) device_buffer_get_pointer(instance->material_atlas);
  ptrs.randoms               = (curandStateXORWOW_t*) device_buffer_get_pointer(instance->randoms);
  ptrs.raydir_buffer         = (vec3*) device_buffer_get_pointer(instance->raydir_buffer);
  ptrs.trace_result_buffer   = (TraceResult*) device_buffer_get_pointer(instance->trace_result_buffer);
  ptrs.trace_result_temporal = (TraceResult*) device_buffer_get_pointer(instance->trace_result_temporal);
  ptrs.state_buffer          = (uint8_t*) device_buffer_get_pointer(instance->state_buffer);
  ptrs.light_samples         = (LightSample*) device_buffer_get_pointer(instance->light_samples_1);
  ptrs.light_eval_data       = (LightEvalData*) device_buffer_get_pointer(instance->light_eval_data);

  gpuErrchk(cudaMemcpyToSymbol(device, &(ptrs), sizeof(DevicePointers), 0, cudaMemcpyHostToDevice));
  log_message("Updated device pointers.");
}

extern "C" void free_host_memory(void* ptr) {
  gpuErrchk(cudaFreeHost(ptr));
}
