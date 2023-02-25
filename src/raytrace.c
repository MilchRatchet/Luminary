#include "raytrace.h"

#include <cuda_runtime_api.h>
#include <math.h>
#include <string.h>

#include "buffer.h"
#include "denoise.h"
#include "device.h"
#include "sky_defines.h"
#include "utils.h"

////////////////////////////////////////////////////////////////////
// Math utility functions
////////////////////////////////////////////////////////////////////

static vec3 rotate_vector_by_quaternion(const vec3 v, const Quaternion q) {
  vec3 result;

  vec3 u;
  u.x = q.x;
  u.y = q.y;
  u.z = q.z;

  const float s = q.w;

  const float dot_uv = u.x * v.x + u.y * v.y + u.z * v.z;
  const float dot_uu = u.x * u.x + u.y * u.y + u.z * u.z;

  vec3 cross;

  cross.x = u.y * v.z - u.z * v.y;
  cross.y = u.z * v.x - u.x * v.z;
  cross.z = u.x * v.y - u.y * v.x;

  result.x = 2.0f * dot_uv * u.x + ((s * s) - dot_uu) * v.x + 2.0f * s * cross.x;
  result.y = 2.0f * dot_uv * u.y + ((s * s) - dot_uu) * v.y + 2.0f * s * cross.y;
  result.z = 2.0f * dot_uv * u.z + ((s * s) - dot_uu) * v.z + 2.0f * s * cross.z;

  return result;
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

static vec3 angles_to_direction(const float altitude, const float azimuth) {
  vec3 dir;
  dir.x = cosf(azimuth) * cosf(altitude);
  dir.y = sinf(altitude);
  dir.z = sinf(azimuth) * cosf(altitude);

  return dir;
}

////////////////////////////////////////////////////////////////////
// Scene content related functions
////////////////////////////////////////////////////////////////////

/*
 * Computes matrices used for temporal reprojection.
 * @param instance RaytraceInstance to be used.
 */
void raytrace_update_temporal_matrix(RaytraceInstance* instance) {
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
void raytrace_update_jitter(RaytraceInstance* instance) {
  Jitter jitter;

  jitter.prev_x = instance->jitter.x;
  jitter.prev_y = instance->jitter.y;
  jitter.x      = halton(instance->temporal_frames, 2);
  jitter.y      = halton(instance->temporal_frames, 3);

  instance->jitter = jitter;
  device_update_symbol(jitter, jitter);
}

/*
 * Computes and uploads to GPU camera rotation quaternions and fov step increments.
 * @param scene Scene which contains camera information.
 * @param width Number of pixel in horizontal in internal render buffer.
 * @param height Number of pixel in vertical in internal render buffer.
 */
void raytrace_update_camera_transform(const Scene scene, const unsigned int width, const unsigned int height) {
  const Quaternion q = get_rotation_quaternion(scene.camera.rotation);
  device_update_symbol(camera_rotation, q);

  const float step = 2.0f * (scene.camera.fov / width);
  const float vfov = step * height / 2.0f;

  device_update_symbol(step, step);
  device_update_symbol(vfov, vfov);
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

  instance->scene_gpu.toy.position.x = instance->scene_gpu.camera.pos.x + offset.x;
  instance->scene_gpu.toy.position.y = instance->scene_gpu.camera.pos.y + offset.y;
  instance->scene_gpu.toy.position.z = instance->scene_gpu.camera.pos.z + offset.z;
}

/*
 * Positions toy in front of the camera.
 * @param instance RaytraceInstance to be used.
 */
void raytrace_center_toy_at_camera(RaytraceInstance* instance) {
  const Quaternion q = get_rotation_quaternion(instance->scene_gpu.camera.rotation);

  vec3 offset;
  offset.x = 0.0f;
  offset.y = 0.0f;
  offset.z = -1.0f;

  offset = rotate_vector_by_quaternion(offset, q);

  offset.x *= 3.0f * instance->scene_gpu.toy.scale;
  offset.y *= 3.0f * instance->scene_gpu.toy.scale;
  offset.z *= 3.0f * instance->scene_gpu.toy.scale;

  instance->scene_gpu.toy.position.x = instance->scene_gpu.camera.pos.x + offset.x;
  instance->scene_gpu.toy.position.y = instance->scene_gpu.camera.pos.y + offset.y;
  instance->scene_gpu.toy.position.z = instance->scene_gpu.camera.pos.z + offset.z;
}

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

////////////////////////////////////////////////////////////////////
// Raytrace main functions
////////////////////////////////////////////////////////////////////

void raytrace_execute(RaytraceInstance* instance) {
  device_update_symbol(temporal_frames, instance->temporal_frames);

  raytrace_update_light_resampling_active(instance);

  device_generate_tasks();

  if (instance->shading_mode) {
    device_execute_debug_kernels(instance, TYPE_CAMERA);
  }
  else {
    device_execute_main_kernels(instance, TYPE_CAMERA);
    device_execute_main_kernels(instance, TYPE_LIGHT);
    for (int i = 0; i < instance->max_ray_depth; i++) {
      device_execute_main_kernels(instance, TYPE_BOUNCE);
      device_execute_main_kernels(instance, TYPE_LIGHT);
    }
  }

  device_handle_accumulation(instance);

  raytrace_update_temporal_matrix(instance);

  gpuErrchk(cudaDeviceSynchronize());
}

void raytrace_init(RaytraceInstance** _instance, General general, TextureAtlas tex_atlas, Scene* scene) {
  RaytraceInstance* instance = (RaytraceInstance*) calloc(1, sizeof(RaytraceInstance));

  instance->width         = general.width;
  instance->height        = general.height;
  instance->output_width  = general.width;
  instance->output_height = general.height;

  if (general.denoiser == DENOISING_UPSCALING) {
    if (instance->width * instance->height > 18144000) {
      error_message(
        "Internal resolution is too high for denoising with upscaling! The maximum is ~18144000 pixels. Upscaling is turned off!");
      general.denoiser = DENOISING_ON;
    }
    else {
      instance->output_width *= 2;
      instance->output_height *= 2;
    }
  }

  instance->max_ray_depth   = general.max_ray_depth;
  instance->offline_samples = general.samples;
  instance->denoiser        = general.denoiser;
  instance->reservoir_size  = general.reservoir_size;

  instance->tex_atlas = tex_atlas;

  instance->scene_gpu          = *scene;
  instance->settings           = general;
  instance->shading_mode       = 0;
  instance->accum_mode         = TEMPORAL_ACCUMULATION;
  instance->spatial_samples    = 5;
  instance->spatial_iterations = 2;

  instance->atmo_settings.base_density           = scene->sky.base_density;
  instance->atmo_settings.ground_visibility      = scene->sky.ground_visibility;
  instance->atmo_settings.mie_density            = scene->sky.mie_density;
  instance->atmo_settings.mie_falloff            = scene->sky.mie_falloff;
  instance->atmo_settings.mie_g                  = scene->sky.mie_g;
  instance->atmo_settings.ozone_absorption       = scene->sky.ozone_absorption;
  instance->atmo_settings.ozone_density          = scene->sky.ozone_density;
  instance->atmo_settings.ozone_layer_thickness  = scene->sky.ozone_layer_thickness;
  instance->atmo_settings.rayleigh_density       = scene->sky.rayleigh_density;
  instance->atmo_settings.rayleigh_falloff       = scene->sky.rayleigh_falloff;
  instance->atmo_settings.multiscattering_factor = scene->sky.multiscattering_factor;

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
  device_buffer_init(&instance->albedo_buffer);
  device_buffer_init(&instance->normal_buffer);
  device_buffer_init(&instance->light_records);
  device_buffer_init(&instance->bounce_records);
  device_buffer_init(&instance->buffer_8bit);
  device_buffer_init(&instance->randoms);
  device_buffer_init(&instance->raydir_buffer);
  device_buffer_init(&instance->trace_result_buffer);
  device_buffer_init(&instance->state_buffer);
  device_buffer_init(&instance->light_samples_1);
  device_buffer_init(&instance->light_samples_2);
  device_buffer_init(&instance->light_eval_data);
  device_buffer_init(&instance->cloud_noise);
  device_buffer_init(&instance->sky_ms_luts);
  device_buffer_init(&instance->sky_tm_luts);

  device_buffer_malloc(instance->buffer_8bit, sizeof(XRGB8), instance->width * instance->height);

  device_malloc((void**) &(instance->scene_gpu.texture_assignments), sizeof(TextureAssignment) * scene->materials_length);
  device_malloc((void**) &(instance->scene_gpu.triangles), sizeof(Triangle) * instance->scene_gpu.triangles_length);
  device_malloc((void**) &(instance->scene_gpu.traversal_triangles), sizeof(TraversalTriangle) * instance->scene_gpu.triangles_length);
  device_malloc((void**) &(instance->scene_gpu.nodes), sizeof(Node8) * instance->scene_gpu.nodes_length);
  device_malloc((void**) &(instance->scene_gpu.triangle_lights), sizeof(TriangleLight) * instance->scene_gpu.triangle_lights_length);

  gpuErrchk(cudaMemcpy(
    instance->scene_gpu.texture_assignments, scene->texture_assignments, sizeof(TextureAssignment) * scene->materials_length,
    cudaMemcpyHostToDevice));
  gpuErrchk(
    cudaMemcpy(instance->scene_gpu.triangles, scene->triangles, sizeof(Triangle) * scene->triangles_length, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(
    instance->scene_gpu.traversal_triangles, scene->traversal_triangles, sizeof(TraversalTriangle) * scene->triangles_length,
    cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(instance->scene_gpu.nodes, scene->nodes, sizeof(Node8) * scene->nodes_length, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(
    instance->scene_gpu.triangle_lights, scene->triangle_lights, sizeof(TriangleLight) * scene->triangle_lights_length,
    cudaMemcpyHostToDevice));

  device_update_symbol(texture_assignments, instance->scene_gpu.texture_assignments);

  device_sky_generate_LUTs(instance);
  device_cloud_noise_generate(instance);

  raytrace_update_light_resampling_active(instance);
  raytrace_allocate_buffers(instance);
  device_bloom_allocate_mips(instance);
  raytrace_update_device_pointers(instance);
  device_initialize_random_generators();
  raytrace_prepare(instance);
  raytrace_update_temporal_matrix(instance);

  instance->snap_resolution = SNAP_RESOLUTION_RENDER;

  *_instance = instance;
}

void raytrace_reset(RaytraceInstance* instance) {
  device_bloom_free_mips(instance);

  // If denoising was set to on but no denoise setup exists, then we do not create one either now
  const int allocate_denoise = instance->settings.denoiser && !(instance->denoiser && !instance->denoise_setup);

  if (instance->denoise_setup) {
    denoise_free(instance);
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
  raytrace_allocate_buffers(instance);
  device_bloom_allocate_mips(instance);
  raytrace_update_device_pointers(instance);
  device_initialize_random_generators();
  raytrace_prepare(instance);
  raytrace_update_temporal_matrix(instance);

  if (allocate_denoise) {
    denoise_create(instance);
  }

  log_message("Reset raytrace instance.");
}

/*
 * Updates and uploades all dynamic data. Must be called at least once before using trace.
 * @param instance RaytraceInstance to be used.
 */
void raytrace_prepare(RaytraceInstance* instance) {
  if (instance->scene_gpu.toy.flashlight_mode) {
    toy_flashlight_set_position(instance);
  }
  raytrace_update_device_scene(instance);
  device_update_symbol(shading_mode, instance->shading_mode);
  update_special_lights(instance->scene_gpu);
  raytrace_update_camera_transform(instance->scene_gpu, instance->width, instance->height);
  raytrace_update_jitter(instance);
  device_update_symbol(accum_mode, instance->accum_mode);
  device_update_symbol(spatial_samples, instance->spatial_samples);
}

/*
 * Allocates all pixel count related buffers.
 * @param instance RaytraceInstance to be used.
 */
void raytrace_allocate_buffers(RaytraceInstance* instance) {
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

  const int thread_count      = device_get_thread_count();
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
  device_buffer_malloc(instance->randoms, /*sizeof(curandStateXORWOW_t)*/ 44, thread_count);

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

void raytrace_update_device_pointers(RaytraceInstance* instance) {
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
  ptrs.randoms              = (void*) device_buffer_get_pointer(instance->randoms);
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

void raytrace_free_work_buffers(RaytraceInstance* instance) {
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

void raytrace_free_output_buffers(RaytraceInstance* instance) {
  device_buffer_free(instance->frame_output);

  if (instance->denoiser) {
    device_buffer_free(instance->albedo_buffer);
    device_buffer_free(instance->normal_buffer);
  }

  device_bloom_free_mips(instance);

  free(instance);
}

void raytrace_init_8bit_frame(RaytraceInstance* instance, const unsigned int width, const unsigned int height) {
  device_buffer_malloc(instance->buffer_8bit, sizeof(XRGB8), width * height);
  raytrace_update_device_pointers(instance);
}

void raytrace_free_8bit_frame(RaytraceInstance* instance) {
  device_buffer_free(instance->buffer_8bit);
}

/*
 * Computes whether light resampling is used.
 * @param instance RaytraceInstance
 */
void raytrace_update_light_resampling_active(RaytraceInstance* instance) {
  instance->light_resampling =
    instance->scene_gpu.material.lights_active || (instance->scene_gpu.toy.emissive && instance->scene_gpu.toy.active);

  device_update_symbol(light_resampling, instance->light_resampling);
}

void raytrace_update_device_scene(RaytraceInstance* instance) {
  device_update_symbol(scene, instance->scene_gpu);
}
